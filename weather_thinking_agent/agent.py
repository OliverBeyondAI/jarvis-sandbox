#!/usr/bin/env python3
"""
Weather Thinking Agent — Claude Opus 4.7 with extended thinking + MCP tool use.

Demonstrates:
  1. Connecting to a mock weather MCP server via stdio transport
  2. Discovering and using MCP tools (get_current_weather, get_forecast, get_weather_alerts)
  3. Extended thinking (chain-of-thought) with configurable budget_tokens
  4. Multi-step agentic loop: Claude reasons, calls tools, and synthesizes results
  5. Full trace capture for inspection (thinking blocks, tool calls, final response)

Usage:
    python agent.py                           # default travel-planning query
    python agent.py "Is Denver safe for hiking this week?"
    python agent.py --budget 8000 "Compare SF and Miami weather"
    python agent.py --trace trace.json        # save full trace to JSON
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-6-20250514"
MAX_TOKENS = 16_000
DEFAULT_BUDGET_TOKENS = 10_000
MAX_TURNS = 15

MCP_SERVER_SCRIPT = str(Path(__file__).parent.parent / "mock_weather_mcp" / "mcp_server.py")

# ---------------------------------------------------------------------------
# Trace data structures
# ---------------------------------------------------------------------------


@dataclass
class ThinkingBlock:
    """A single extended-thinking block from Claude's response."""
    text: str
    budget_tokens: int
    turn: int


@dataclass
class ToolCall:
    """A single tool invocation and its result."""
    name: str
    input: dict[str, Any]
    result: str
    turn: int
    duration_ms: float = 0.0


@dataclass
class AgentTrace:
    """Complete trace of an agent run — thinking, tool calls, and final answer."""
    prompt: str
    model: str
    budget_tokens: int
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_response: str = ""
    tools_discovered: list[dict[str, str]] = field(default_factory=list)
    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    wall_time_s: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model": self.model,
            "budget_tokens": self.budget_tokens,
            "thinking_blocks": [
                {"text": t.text, "budget_tokens": t.budget_tokens, "turn": t.turn}
                for t in self.thinking_blocks
            ],
            "tool_calls": [
                {
                    "name": tc.name,
                    "input": tc.input,
                    "result": tc.result,
                    "turn": tc.turn,
                    "duration_ms": tc.duration_ms,
                }
                for tc in self.tool_calls
            ],
            "final_response": self.final_response,
            "tools_discovered": self.tools_discovered,
            "total_turns": self.total_turns,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "wall_time_s": round(self.wall_time_s, 2),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Terminal formatting helpers
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
# Weather Thinking Agent
# ---------------------------------------------------------------------------

class WeatherThinkingAgent:
    """
    Agent that connects to a weather MCP server, uses Claude Opus 4.7's
    extended thinking to reason through weather queries, and calls MCP
    tools to fetch data.

    Extended thinking lets Claude "think out loud" before responding,
    producing a chain-of-thought trace that reveals its reasoning process.
    This is especially useful for multi-city comparisons and travel
    recommendations where the model needs to weigh multiple factors.
    """

    def __init__(
        self,
        server_script: str = MCP_SERVER_SCRIPT,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        verbose: bool = True,
    ):
        self.server_script = server_script
        self.model = model
        self.max_tokens = max_tokens
        self.budget_tokens = budget_tokens
        self.verbose = verbose
        self.client = anthropic.AsyncAnthropic()

    async def run(self, prompt: str) -> AgentTrace:
        """
        Full agent lifecycle:
          1. Connect to MCP server via stdio
          2. Discover available weather tools
          3. Run agentic loop with extended thinking
          4. Return complete trace
        """
        trace = AgentTrace(
            prompt=prompt,
            model=self.model,
            budget_tokens=self.budget_tokens,
        )
        t0 = time.monotonic()

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
        )

        self._log(f"  {dim('Server:')} {self.server_script}")
        self._log(f"  {dim('Model:')}  {self.model}")
        self._log(f"  {dim('Think:')}  budget_tokens={self.budget_tokens}")

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._log(f"  {green('Connected to MCP server')}")

                # Discover tools
                tools_result = await session.list_tools()
                claude_tools = []
                for tool in tools_result.tools:
                    claude_tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    })
                    trace.tools_discovered.append({
                        "name": tool.name,
                        "description": (tool.description or "").split("\n")[0],
                    })

                self._log(f"  {green('Discovered')} {bold(str(len(claude_tools)))} tools: "
                          f"{', '.join(t['name'] for t in claude_tools)}")

                # Run agentic loop
                await self._agentic_loop(session, prompt, claude_tools, trace)

        trace.wall_time_s = time.monotonic() - t0
        return trace

    async def _agentic_loop(
        self,
        session: ClientSession,
        prompt: str,
        claude_tools: list[dict[str, Any]],
        trace: AgentTrace,
    ) -> None:
        """Run the agentic loop with extended thinking enabled."""

        system = (
            "You are an expert travel weather analyst. Use the available weather "
            "tools to look up current conditions, forecasts, and alerts for cities. "
            "Think carefully about weather patterns, safety considerations, and "
            "travel logistics. Provide clear, actionable recommendations backed "
            "by the data you retrieve.\n\n"
            "When comparing cities, check weather data for each one before making "
            "a recommendation. Consider temperature, precipitation, severe weather "
            "alerts, and overall comfort."
        )

        messages: list[dict] = [{"role": "user", "content": prompt}]

        for turn in range(1, MAX_TURNS + 1):
            section(f"Turn {turn}")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                tools=claude_tools,
                messages=messages,
                thinking={
                    "type": "enabled",
                    "budget_tokens": self.budget_tokens,
                },
            )

            # Track token usage
            if hasattr(response, "usage"):
                trace.total_input_tokens += getattr(response.usage, "input_tokens", 0)
                trace.total_output_tokens += getattr(response.usage, "output_tokens", 0)

            trace.total_turns = turn

            # Process response content blocks
            has_tool_use = False
            assistant_content = []

            for block in response.content:
                assistant_content.append(block)

                if block.type == "thinking":
                    thinking = ThinkingBlock(
                        text=block.thinking,
                        budget_tokens=self.budget_tokens,
                        turn=turn,
                    )
                    trace.thinking_blocks.append(thinking)
                    preview = block.thinking[:200].replace("\n", " ")
                    self._log(f"  {magenta('Thinking:')} {dim(preview)}...")

                elif block.type == "text":
                    self._log(f"  {cyan('Text:')} {block.text[:150]}...")

                elif block.type == "tool_use":
                    has_tool_use = True

            # If no tool use, we're done
            if response.stop_reason == "end_turn" or not has_tool_use:
                text_parts = [
                    b.text for b in response.content if hasattr(b, "text") and b.type == "text"
                ]
                trace.final_response = "\n".join(text_parts)
                self._log(f"\n  {green('Agent finished')} after {turn} turn(s).")
                return

            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_input = dict(block.input)
                self._log(f"  {yellow('Tool call:')} {bold(block.name)}"
                          f"({dim(json.dumps(tool_input)[:80])})")

                t0 = time.monotonic()
                mcp_result = await session.call_tool(block.name, tool_input)
                elapsed_ms = (time.monotonic() - t0) * 1000

                result_text = "".join(
                    b.text for b in mcp_result.content if hasattr(b, "text")
                )

                tc = ToolCall(
                    name=block.name,
                    input=tool_input,
                    result=result_text,
                    turn=turn,
                    duration_ms=round(elapsed_ms, 1),
                )
                trace.tool_calls.append(tc)

                self._log(f"    {green('Result:')} {dim(result_text[:100])}... "
                          f"({elapsed_ms:.0f}ms)")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        self._log(f"  {red('Reached max turns')} ({MAX_TURNS})")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = (
    "I'm planning a trip this week and considering New York, Miami, or Denver. "
    "Check the current weather, forecast, and any active alerts for all three cities. "
    "Then recommend which city I should visit and explain your reasoning."
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Weather Thinking Agent — Opus 4.7 + extended thinking + MCP",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="The weather query to answer",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET_TOKENS,
        help=f"Extended thinking budget in tokens (default: {DEFAULT_BUDGET_TOKENS})",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Save full agent trace to this JSON file",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Generate an HTML trace viewer at this path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"\n  {red('Error:')} ANTHROPIC_API_KEY environment variable not set.")
        print(f"  Set it and try again.\n")
        sys.exit(1)

    banner("Weather Thinking Agent")
    print(f"  {bold('Query:')} {args.prompt}\n")

    agent = WeatherThinkingAgent(
        budget_tokens=args.budget,
        verbose=not args.quiet,
    )

    trace = asyncio.run(agent.run(args.prompt))

    # Print final response
    section("Final Response")
    print()
    for line in trace.final_response.strip().splitlines():
        print(f"    {line}")
    print()

    # Print stats
    section("Stats")
    print(f"    Model:            {trace.model}")
    print(f"    Thinking budget:  {trace.budget_tokens:,} tokens")
    print(f"    Thinking blocks:  {len(trace.thinking_blocks)}")
    print(f"    Tool calls:       {len(trace.tool_calls)}")
    print(f"    Turns:            {trace.total_turns}")
    print(f"    Input tokens:     {trace.total_input_tokens:,}")
    print(f"    Output tokens:    {trace.total_output_tokens:,}")
    print(f"    Wall time:        {trace.wall_time_s:.1f}s")
    print()

    # Save trace JSON
    if args.trace:
        trace_path = Path(args.trace)
        trace_path.write_text(json.dumps(trace.to_dict(), indent=2))
        print(f"  Trace saved to: {trace_path}")

    # Generate HTML viewer
    if args.html:
        from viewer import generate_html_viewer
        html_path = Path(args.html)
        html_path.write_text(generate_html_viewer(trace.to_dict()))
        print(f"  HTML viewer saved to: {html_path}")


if __name__ == "__main__":
    main()
