#!/usr/bin/env python3
"""
Mock Weather MCP Server — Demo Runner

Connects to the weather MCP server, discovers tools, and optionally runs
queries through a Claude agent to demonstrate the full MCP workflow.

Usage:
    python run_demo.py                # tool discovery + direct tool calls (no API key)
    python run_demo.py --discover     # tool discovery only
    python run_demo.py --call         # discovery + direct MCP tool calls
    python run_demo.py --agent        # full agent demo (requires ANTHROPIC_API_KEY)

Requires:
    - Python 3.11+
    - pip install mcp anthropic
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")

# ---------------------------------------------------------------------------
# Formatting helpers
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


def banner(title: str) -> None:
    w = 64
    print(f"\n{'=' * w}")
    print(f"{' ' * ((w - len(title)) // 2)}{bold(title)}")
    print(f"{'=' * w}\n")


def section(title: str) -> None:
    print(f"\n{dim('-' * 64)}")
    print(f"  {cyan(title)}")
    print(f"{dim('-' * 64)}")


# ---------------------------------------------------------------------------
# Demo 1: Tool Discovery
# ---------------------------------------------------------------------------

async def demo_discover() -> None:
    """Connect to the MCP server and display all discovered tools."""
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

            total_bytes = 0
            for tool in result.tools:
                schema_json = json.dumps(tool.inputSchema, indent=2)
                schema_bytes = len(schema_json)
                total_bytes += schema_bytes

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

            print(f"  Total schema payload: {bold(str(total_bytes))} bytes (~{total_bytes // 4} tokens)")
            print(f"  These schemas define the MCP protocol contract for each tool.\n")


# ---------------------------------------------------------------------------
# Demo 2: Direct Tool Calls (no LLM needed)
# ---------------------------------------------------------------------------

DEMO_CALLS = [
    {
        "label": "Current Weather",
        "tool": "get_current_weather",
        "args": {"city": "New York", "units": "fahrenheit"},
    },
    {
        "label": "5-Day Forecast",
        "tool": "get_forecast",
        "args": {"city": "Denver", "days": 5, "units": "fahrenheit"},
    },
    {
        "label": "Weather Alerts",
        "tool": "get_weather_alerts",
        "args": {"city": "Miami"},
    },
    {
        "label": "City Not Found",
        "tool": "get_current_weather",
        "args": {"city": "Atlantis"},
    },
]


async def demo_direct_calls() -> None:
    """Call each tool directly via MCP and display results."""
    banner("Direct MCP Tool Calls")
    print(f"  Calling tools directly through the MCP protocol (no LLM).\n")

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

                result_text = ""
                for block in result.content:
                    if hasattr(block, "text"):
                        result_text += block.text

                # Pretty-print the JSON result
                try:
                    parsed = json.loads(result_text)
                    formatted = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    formatted = result_text

                for line in formatted.splitlines():
                    print(f"    {line}")
                print()

    print(f"  {green('All tool calls completed successfully.')}\n")


# ---------------------------------------------------------------------------
# Demo 3: Agent Loop (requires API key)
# ---------------------------------------------------------------------------

async def demo_agent() -> None:
    """Run a Claude agent that uses the weather MCP tools."""
    import anthropic

    banner("Claude Agent with Weather MCP Tools")

    model = "claude-sonnet-4-6-20250627"
    client = anthropic.AsyncAnthropic()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Discover tools and build schemas
            tools_result = await session.list_tools()
            claude_tools = []
            for tool in tools_result.tools:
                claude_tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                })

            print(f"  Model:  {dim(model)}")
            print(f"  Tools:  {', '.join(t['name'] for t in claude_tools)}")

            # Run a multi-step query
            prompt = (
                "I'm planning a trip and considering New York, Miami, or Denver. "
                "Check the current weather and any active alerts for all three cities, "
                "then give me a recommendation for which city to visit this week."
            )

            section("Agent Query")
            print(f"\n  {bold('Prompt:')} {prompt}\n")

            messages: list[dict] = [{"role": "user", "content": prompt}]
            system = (
                "You are a helpful travel weather assistant. Use the available "
                "weather tools to look up conditions and provide a clear, concise "
                "recommendation. Always check both current weather and alerts."
            )

            tool_calls_made = 0
            for turn in range(1, 16):
                response = await client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system,
                    tools=claude_tools,
                    messages=messages,
                )

                if response.stop_reason == "end_turn":
                    # Extract and display final text
                    text = "\n".join(
                        b.text for b in response.content if hasattr(b, "text")
                    )
                    section("Agent Response")
                    print()
                    for line in text.strip().splitlines():
                        print(f"    {line}")
                    print()
                    break

                # Process tool calls
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_calls_made += 1
                    print(f"  {dim(f'Turn {turn}:')} {green(block.name)}({dim(json.dumps(dict(block.input))[:80])})")

                    mcp_result = await session.call_tool(block.name, dict(block.input))
                    result_text = "".join(
                        b.text for b in mcp_result.content if hasattr(b, "text")
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

            print(f"  {bold('Stats:')}")
            print(f"    Tool calls made: {tool_calls_made}")
            print(f"    Turns used: {turn}")
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
            print(f"  Run with {bold('--discover')} or {bold('--call')} instead (no API key needed).\n")
            sys.exit(1)
        asyncio.run(demo_discover())
        asyncio.run(demo_agent())
        return

    # Default: discovery + direct calls (no API key needed)
    asyncio.run(demo_discover())
    asyncio.run(demo_direct_calls())


if __name__ == "__main__":
    main()
