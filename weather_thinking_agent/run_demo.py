#!/usr/bin/env python3
"""
Weather Thinking Agent — Demo Runner

Demonstrates the full agent workflow:
  1. Tool discovery (no API key needed)
  2. Direct MCP tool calls (no API key needed)
  3. Full agent run with extended thinking (requires ANTHROPIC_API_KEY)
  4. HTML trace viewer generation

Usage:
    python run_demo.py                  # discovery + direct calls (no API key)
    python run_demo.py --discover       # tool discovery only
    python run_demo.py --agent          # full agent demo (needs API key)
    python run_demo.py --agent --html   # agent + HTML viewer output
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Resolve MCP server path
MCP_SERVER_SCRIPT = str(Path(__file__).parent.parent / "mock_weather_mcp" / "mcp_server.py")

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t: str) -> str:
    return _c("1", t)

def dim(t: str) -> str:
    return _c("2", t)

def green(t: str) -> str:
    return _c("32", t)

def cyan(t: str) -> str:
    return _c("36", t)

def yellow(t: str) -> str:
    return _c("33", t)

def red(t: str) -> str:
    return _c("31", t)

def magenta(t: str) -> str:
    return _c("35", t)


def banner(title: str) -> None:
    w = 68
    print(f"\n{'=' * w}")
    print(f"{' ' * ((w - len(title)) // 2)}{bold(title)}")
    print(f"{'=' * w}\n")


def section(title: str) -> None:
    print(f"\n{dim('-' * 68)}")
    print(f"  {cyan(title)}")
    print(f"{dim('-' * 68)}")


# ---------------------------------------------------------------------------
# Demo 1: Tool Discovery
# ---------------------------------------------------------------------------

async def demo_discover() -> None:
    """Connect to the weather MCP server and list all tools."""
    banner("MCP Tool Discovery")
    print(f"  Server: {dim(MCP_SERVER_SCRIPT)}\n")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

            print(f"  {green('Connected!')} Found {bold(str(len(result.tools)))} tools:\n")

            for tool in result.tools:
                schema_json = json.dumps(tool.inputSchema, indent=2)
                schema_bytes = len(schema_json)
                desc = (tool.description or "").split("\n")[0].strip()
                props = tool.inputSchema.get("properties", {})
                required = set(tool.inputSchema.get("required", []))

                params = []
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "any")
                    marker = " *" if pname in required else ""
                    params.append(f"{pname}: {ptype}{marker}")

                print(f"  {bold(tool.name)}")
                print(f"    {desc}")
                print(f"    Schema: {schema_bytes} bytes (~{schema_bytes // 4} tokens)")
                print(f"    Params: {', '.join(params)}")
                print()


# ---------------------------------------------------------------------------
# Demo 2: Direct Tool Calls
# ---------------------------------------------------------------------------

DEMO_CALLS = [
    {
        "label": "Current Weather — New York",
        "tool": "get_current_weather",
        "args": {"city": "New York", "units": "fahrenheit"},
    },
    {
        "label": "5-Day Forecast — Denver",
        "tool": "get_forecast",
        "args": {"city": "Denver", "days": 5, "units": "fahrenheit"},
    },
    {
        "label": "Weather Alerts — Miami",
        "tool": "get_weather_alerts",
        "args": {"city": "Miami"},
    },
]


async def demo_direct_calls() -> None:
    """Call weather tools directly via MCP (no LLM)."""
    banner("Direct MCP Tool Calls")
    print(f"  Calling tools directly through MCP (no LLM needed).\n")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            for i, call in enumerate(DEMO_CALLS, 1):
                section(f"[{i}/{len(DEMO_CALLS)}] {call['label']}")
                print(f"\n  {bold('Tool:')} {call['tool']}")
                print(f"  {bold('Args:')} {json.dumps(call['args'])}\n")

                result = await session.call_tool(call["tool"], call["args"])
                result_text = "".join(
                    b.text for b in result.content if hasattr(b, "text")
                )

                try:
                    parsed = json.loads(result_text)
                    formatted = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    formatted = result_text

                for line in formatted.splitlines():
                    print(f"    {line}")
                print()

    print(f"  {green('All direct calls completed.')}\n")


# ---------------------------------------------------------------------------
# Demo 3: Full Agent with Extended Thinking
# ---------------------------------------------------------------------------

async def demo_agent(generate_html: bool = False) -> None:
    """Run the full weather thinking agent."""
    from agent import WeatherThinkingAgent, banner as agent_banner, section as agent_section

    prompt = (
        "I'm planning a trip this week and considering New York, Miami, or Denver. "
        "Check the current weather, forecast, and any active alerts for all three cities. "
        "Then recommend which city I should visit and explain your reasoning."
    )

    banner("Weather Thinking Agent — Opus 4.7 + Extended Thinking")
    print(f"  {bold('Query:')} {prompt}\n")

    agent = WeatherThinkingAgent(
        budget_tokens=10_000,
        verbose=True,
    )

    trace = await agent.run(prompt)

    # Final response
    section("Final Response")
    print()
    for line in trace.final_response.strip().splitlines():
        print(f"    {line}")
    print()

    # Stats
    section("Agent Stats")
    print(f"    Model:            {trace.model}")
    print(f"    Thinking budget:  {trace.budget_tokens:,} tokens")
    print(f"    Thinking blocks:  {len(trace.thinking_blocks)}")
    print(f"    Tool calls:       {len(trace.tool_calls)}")
    print(f"    Turns:            {trace.total_turns}")
    print(f"    Input tokens:     {trace.total_input_tokens:,}")
    print(f"    Output tokens:    {trace.total_output_tokens:,}")
    print(f"    Wall time:        {trace.wall_time_s:.1f}s")
    print()

    # Save trace
    trace_path = Path(__file__).parent / "trace.json"
    trace_path.write_text(json.dumps(trace.to_dict(), indent=2))
    print(f"  Trace saved to: {trace_path}")

    # Generate HTML viewer
    if generate_html:
        from viewer import generate_html_viewer
        html_path = Path(__file__).parent / "trace_viewer.html"
        html_path.write_text(generate_html_viewer(trace.to_dict()))
        print(f"  HTML viewer saved to: {html_path}")
        print(f"  Open in browser: file://{html_path.resolve()}")

    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    if "--discover" in args:
        asyncio.run(demo_discover())
        return

    if "--agent" in args:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(f"\n  {red('Error:')} ANTHROPIC_API_KEY not set.")
            print(f"  Run with {bold('--discover')} instead (no API key needed).\n")
            sys.exit(1)
        asyncio.run(demo_discover())
        asyncio.run(demo_agent(generate_html="--html" in args))
        return

    # Default: discovery + direct calls (no API key)
    asyncio.run(demo_discover())
    asyncio.run(demo_direct_calls())


if __name__ == "__main__":
    main()
