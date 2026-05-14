#!/usr/bin/env python3
"""
End-to-End Demo: Mock Weather MCP Server + Thinking Agent

Demonstrates the complete flow of a user query being answered through
MCP tool interaction with Claude's extended thinking visible at each step.

Phases:
  1. MCP Server Boot & Tool Discovery
  2. Direct MCP Tool Calls (no LLM — verifies server is working)
  3. Full Agent Run with Extended Thinking (Claude reasons, calls tools, responds)
  4. HTML Trace Viewer Generation (self-contained visualization)

Usage:
    python run_e2e_demo.py                        # phases 1-2 (no API key)
    python run_e2e_demo.py --agent                # phases 1-4 (needs ANTHROPIC_API_KEY)
    python run_e2e_demo.py --agent --html         # phases 1-4 + HTML viewer
    python run_e2e_demo.py --agent --query "..."  # custom query
    python run_e2e_demo.py --discover             # phase 1 only
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
MCP_SERVER = str(ROOT / "mock_weather_mcp" / "mcp_server.py")
AGENT_DIR = ROOT / "weather_thinking_agent"

# ---------------------------------------------------------------------------
# Terminal formatting
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

def magenta(t: str) -> str:
    return _c("35", t)

def red(t: str) -> str:
    return _c("31", t)

def blue(t: str) -> str:
    return _c("34", t)


def phase_banner(num: int, title: str, desc: str) -> None:
    w = 72
    print(f"\n{'=' * w}")
    print(f"  {bold(cyan(f'PHASE {num}'))}  {bold(title)}")
    print(f"  {dim(desc)}")
    print(f"{'=' * w}\n")


def step(label: str) -> None:
    print(f"  {yellow('>')} {label}")


def ok(label: str) -> None:
    print(f"  {green('✓')} {label}")


def info(label: str, value: str) -> None:
    print(f"    {dim(label + ':')} {value}")


def divider() -> None:
    print(f"\n  {dim('─' * 60)}\n")


# ---------------------------------------------------------------------------
# Phase 1: MCP Server Boot & Tool Discovery
# ---------------------------------------------------------------------------

async def phase_discover() -> list[dict]:
    """Boot the MCP server, connect, and list available tools."""
    phase_banner(1, "MCP Server Boot & Tool Discovery",
                 "Start the mock weather server via stdio and discover its tools")

    step(f"Starting MCP server: {dim(MCP_SERVER)}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            t0 = time.monotonic()
            await session.initialize()
            elapsed = (time.monotonic() - t0) * 1000
            ok(f"Server connected ({elapsed:.0f}ms)")

            result = await session.list_tools()
            ok(f"Discovered {bold(str(len(result.tools)))} tools\n")

            tools_info = []
            for tool in result.tools:
                schema = tool.inputSchema
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                schema_bytes = len(json.dumps(schema))

                params = []
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "any")
                    req = " (required)" if pname in required else ""
                    params.append(f"{pname}: {ptype}{req}")

                desc = (tool.description or "").split("\n")[0].strip()

                print(f"  {bold(blue(tool.name))}")
                print(f"    {desc}")
                print(f"    {dim('Params:')} {', '.join(params)}")
                print(f"    {dim('Schema:')} {schema_bytes} bytes (~{schema_bytes // 4} tokens)")
                print()

                tools_info.append({
                    "name": tool.name,
                    "description": desc,
                    "inputSchema": schema,
                })

    return tools_info


# ---------------------------------------------------------------------------
# Phase 2: Direct MCP Tool Calls
# ---------------------------------------------------------------------------

SAMPLE_CALLS = [
    {
        "label": "Current Weather — New York",
        "tool": "get_current_weather",
        "args": {"city": "New York", "units": "fahrenheit"},
    },
    {
        "label": "5-Day Forecast — Denver",
        "tool": "get_forecast",
        "args": {"city": "Denver", "days": 5},
    },
    {
        "label": "Weather Alerts — Miami",
        "tool": "get_weather_alerts",
        "args": {"city": "Miami"},
    },
]


async def phase_direct_calls() -> None:
    """Call each weather tool directly via MCP (no LLM involved)."""
    phase_banner(2, "Direct MCP Tool Calls",
                 "Call tools directly through MCP to verify the server works (no LLM needed)")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            for i, call in enumerate(SAMPLE_CALLS, 1):
                step(f"[{i}/{len(SAMPLE_CALLS)}] {call['label']}")
                info("Tool", call["tool"])
                info("Args", json.dumps(call["args"]))

                t0 = time.monotonic()
                result = await session.call_tool(call["tool"], call["args"])
                elapsed = (time.monotonic() - t0) * 1000

                result_text = "".join(
                    b.text for b in result.content if hasattr(b, "text")
                )

                try:
                    parsed = json.loads(result_text)
                    formatted = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    formatted = result_text

                # Show a condensed preview
                lines = formatted.splitlines()
                preview_lines = lines[:12]
                print()
                for line in preview_lines:
                    print(f"      {dim(line)}")
                if len(lines) > 12:
                    print(f"      {dim(f'... ({len(lines) - 12} more lines)')}")

                ok(f"Completed in {elapsed:.0f}ms")
                divider()

    ok("All direct tool calls succeeded — MCP server is working correctly.\n")


# ---------------------------------------------------------------------------
# Phase 3: Full Agent Run with Extended Thinking
# ---------------------------------------------------------------------------

DEFAULT_QUERY = (
    "I'm planning a trip this week and considering New York, Miami, or Denver. "
    "Check the current weather, forecast, and any active alerts for all three cities. "
    "Then recommend which city I should visit and explain your reasoning."
)


async def phase_agent(query: str, generate_html: bool = False) -> None:
    """Run the full thinking agent with visible extended thinking."""

    phase_banner(3, "Agent Run — Extended Thinking + MCP Tools",
                 "Claude Opus 4.7 reasons through the query, calling MCP tools as needed")

    print(f"  {bold('Query:')}")
    for line in query.split(". "):
        line = line.strip()
        if line:
            print(f"    {line}{'.' if not line.endswith('.') else ''}")
    print()

    # Import the agent from the weather_thinking_agent package
    sys.path.insert(0, str(AGENT_DIR))
    from agent import WeatherThinkingAgent

    agent = WeatherThinkingAgent(
        budget_tokens=10_000,
        verbose=True,
    )

    trace = await agent.run(query)

    # ── Final Response ──
    divider()
    print(f"  {bold(cyan('FINAL RESPONSE'))}\n")
    for line in trace.final_response.strip().splitlines():
        print(f"    {line}")

    # ── Stats ──
    divider()
    print(f"  {bold(cyan('AGENT STATS'))}\n")
    stats = [
        ("Model", trace.model),
        ("Thinking budget", f"{trace.budget_tokens:,} tokens"),
        ("Thinking blocks", str(len(trace.thinking_blocks))),
        ("Tool calls", str(len(trace.tool_calls))),
        ("Turns", str(trace.total_turns)),
        ("Input tokens", f"{trace.total_input_tokens:,}"),
        ("Output tokens", f"{trace.total_output_tokens:,}"),
        ("Wall time", f"{trace.wall_time_s:.1f}s"),
    ]
    max_label = max(len(s[0]) for s in stats)
    for label, value in stats:
        print(f"    {label:<{max_label + 2}} {bold(value)}")

    # ── Extended Thinking Summary ──
    if trace.thinking_blocks:
        divider()
        print(f"  {bold(magenta('EXTENDED THINKING SUMMARY'))}\n")
        for i, tb in enumerate(trace.thinking_blocks, 1):
            chars = len(tb.text)
            preview = tb.text[:300].replace("\n", " ").strip()
            print(f"    {magenta(f'Block {i}')} (Turn {tb.turn}, {chars:,} chars):")
            print(f"      {dim(preview)}{'...' if chars > 300 else ''}")
            print()

    # ── Tool Call Summary ──
    if trace.tool_calls:
        divider()
        print(f"  {bold(yellow('TOOL CALL SUMMARY'))}\n")
        for i, tc in enumerate(trace.tool_calls, 1):
            args_str = json.dumps(tc.input)
            print(f"    {yellow(f'{i}.')} {bold(tc.name)}({dim(args_str[:60])}) "
                  f"→ {dim(f'{tc.duration_ms:.0f}ms')}")

    # ── Save trace JSON ──
    trace_path = AGENT_DIR / "trace.json"
    trace_path.write_text(json.dumps(trace.to_dict(), indent=2))
    print(f"\n  {green('✓')} Trace saved: {trace_path}")

    # ── Phase 4: HTML Viewer ──
    if generate_html:
        phase_banner(4, "HTML Trace Viewer",
                     "Generate a self-contained HTML page visualizing the agent's reasoning")

        from viewer import generate_html_viewer
        html_path = AGENT_DIR / "trace_viewer.html"
        html_path.write_text(generate_html_viewer(trace.to_dict()))
        ok(f"HTML viewer saved: {html_path}")
        ok(f"Open in browser: file://{html_path.resolve()}")

    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    # ── Title ──
    w = 72
    print(f"\n{'━' * w}")
    print(f"  {bold('End-to-End Demo: Weather MCP Server + Thinking Agent')}")
    print(f"  {dim('Claude Opus 4.7 · Extended Thinking · MCP Tool Use')}")
    print(f"{'━' * w}")

    # ── Discovery only ──
    if "--discover" in args:
        asyncio.run(phase_discover())
        return

    # ── Full agent run ──
    if "--agent" in args:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(f"\n  {red('Error:')} ANTHROPIC_API_KEY not set.\n")
            print(f"  To run without an API key (phases 1-2 only):")
            print(f"    python run_e2e_demo.py\n")
            print(f"  To run the full agent demo:")
            print(f"    export ANTHROPIC_API_KEY=sk-...")
            print(f"    python run_e2e_demo.py --agent\n")
            sys.exit(1)

        # Parse custom query
        query = DEFAULT_QUERY
        if "--query" in args:
            idx = args.index("--query")
            if idx + 1 < len(args):
                query = args[idx + 1]

        asyncio.run(phase_discover())
        asyncio.run(phase_direct_calls())
        asyncio.run(phase_agent(query, generate_html="--html" in args))
        return

    # ── Default: phases 1-2 (no API key) ──
    asyncio.run(phase_discover())
    asyncio.run(phase_direct_calls())

    print(f"  {dim('To run the full agent with extended thinking:')}")
    print(f"    export ANTHROPIC_API_KEY=sk-...")
    print(f"    python run_e2e_demo.py --agent --html\n")


if __name__ == "__main__":
    main()
