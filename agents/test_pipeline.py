#!/usr/bin/env python3
"""
End-to-End Pipeline Test — Agents Framework

Tests the full pipeline: CLI → agent → tool calls → result output.

Two modes:
  1. Offline (default): Uses mock data to validate agent structure, tool
     dispatch, and output formatting without requiring API keys or network.
  2. Live (--live): Runs the real agent against a sample research topic
     (requires ANTHROPIC_API_KEY and optionally TAVILY_API_KEY).

Usage:
    python -m agents.test_pipeline              # offline tests
    python -m agents.test_pipeline --live       # live end-to-end
    python -m agents.test_pipeline --demo       # show sample output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch

from .agent import AgentResult, Agent, LocalAgentRunner
from .tools import ALL_TOOLS, execute_tool, FETCH_URL_TOOL, TAVILY_SEARCH_TOOL
from .main import RESEARCH_PROMPT_TEMPLATE, RESEARCH_PROMPT_BRIEF


# ---------------------------------------------------------------------------
# Sample data for offline testing
# ---------------------------------------------------------------------------

SAMPLE_TOPIC = "frontier AI models and agentic frameworks"

SAMPLE_AGENT_RESULT = AgentResult(
    text=(
        "## Research Summary: Frontier AI Models and Agentic Frameworks\n\n"
        "### Overview\n"
        "Frontier AI models have advanced rapidly in 2025-2026, with Claude, GPT, "
        "and Gemini families pushing the boundaries of reasoning, tool use, and "
        "autonomous task completion. Agentic frameworks — systems that allow LLMs "
        "to plan, use tools, and execute multi-step tasks — have become a major "
        "focus of both research and product development.\n\n"
        "### Key Developments\n"
        "1. **Claude Agent SDK** — Anthropic released the Claude Agent SDK enabling "
        "managed agent sessions with custom tool integration and environment isolation.\n"
        "2. **Multi-agent orchestration** — Frameworks like CrewAI, AutoGen, and "
        "LangGraph enable coordinated multi-agent workflows.\n"
        "3. **Tool-use standardization** — The Model Context Protocol (MCP) is "
        "emerging as a standard for tool/resource definitions across providers.\n"
        "4. **Reasoning models** — Extended thinking capabilities allow frontier "
        "models to tackle complex multi-step problems.\n\n"
        "### Major Players\n"
        "- **Anthropic** — Claude Opus 4.7, Agent SDK, MCP\n"
        "- **OpenAI** — GPT-4o, Assistants API, function calling\n"
        "- **Google DeepMind** — Gemini Ultra, Vertex AI agents\n"
        "- **Meta** — Llama 3, open-source agent tooling\n\n"
        "### Challenges\n"
        "- Reliability and consistency of multi-step agentic workflows\n"
        "- Safety and alignment in autonomous agent scenarios\n"
        "- Cost and latency of extended reasoning loops\n"
        "- Evaluation and benchmarking of agentic capabilities\n\n"
        "### Future Outlook\n"
        "The next 1-2 years will likely see convergence around standard agent "
        "protocols (MCP), improved reliability through better planning algorithms, "
        "and increasing deployment of agents in production workflows.\n\n"
        "**Sources:**\n"
        "- https://docs.anthropic.com/en/docs/agents\n"
        "- https://arxiv.org/abs/2402.05120\n"
        "- https://modelcontextprotocol.io\n"
    ),
    tool_calls=[
        {"name": "tavily_search", "input": {"query": "frontier AI models 2025 2026 developments"}},
        {"name": "tavily_search", "input": {"query": "agentic AI frameworks Claude Agent SDK MCP"}},
        {"name": "fetch_url", "input": {"url": "https://docs.anthropic.com/en/docs/agents"}},
    ],
    timestamp="2026-05-02T10:00:00.000000",
)


# ---------------------------------------------------------------------------
# Offline tests
# ---------------------------------------------------------------------------

def test_tool_schemas() -> bool:
    """Validate tool schema structure matches Anthropic custom tool format."""
    checks = []

    for tool in ALL_TOOLS:
        name = tool.get("name", "unknown")
        checks.append((f"{name} has name", bool(tool.get("name"))))
        checks.append((f"{name} has type=custom", tool.get("type") == "custom"))
        checks.append((f"{name} has description", bool(tool.get("description"))))
        checks.append((f"{name} has input_schema", bool(tool.get("input_schema"))))
        schema = tool.get("input_schema", {})
        checks.append((f"{name} schema has properties", "properties" in schema))
        checks.append((f"{name} schema has required", "required" in schema))

    return _run_checks("Tool Schemas", checks)


def test_tool_dispatch() -> bool:
    """Test that execute_tool routes correctly and handles unknown tools."""
    checks = []

    # Test unknown tool returns error
    result_str = asyncio.run(execute_tool("nonexistent_tool", {"foo": "bar"}))
    result = json.loads(result_str)
    checks.append(("unknown tool returns error", "error" in result))
    checks.append(("error mentions tool name", "nonexistent_tool" in result.get("error", "")))

    # Test fetch_url with invalid URL returns error gracefully
    result_str = asyncio.run(execute_tool("fetch_url", {"url": "http://invalid.test.localhost.nowhere"}))
    result = json.loads(result_str)
    checks.append(("bad URL returns error", "error" in result))

    return _run_checks("Tool Dispatch", checks)


def test_agent_result_structure() -> bool:
    """Validate AgentResult dataclass behavior."""
    # Default construction
    r = AgentResult()
    checks = [
        ("default text empty", r.text == ""),
        ("default tool_calls empty", r.tool_calls == []),
        ("default has timestamp", bool(r.timestamp)),
    ]

    # With data
    r2 = AgentResult(text="hello", tool_calls=[{"name": "test", "input": {}}])
    checks.append(("text set", r2.text == "hello"))
    checks.append(("tool_calls set", len(r2.tool_calls) == 1))

    return _run_checks("AgentResult Structure", checks)


def test_prompt_templates() -> bool:
    """Validate that research prompt templates format correctly."""
    topic = "quantum computing advances"

    deep = RESEARCH_PROMPT_TEMPLATE.format(topic=topic)
    brief = RESEARCH_PROMPT_BRIEF.format(topic=topic)

    checks = [
        ("deep contains topic", topic in deep),
        ("brief contains topic", topic in brief),
        ("deep has sections", "Key Developments" in deep and "Future Outlook" in deep),
        ("brief is shorter", len(brief) < len(deep)),
        ("deep has 5 sections", deep.count("**") >= 5),
        ("brief mentions sources", "sources" in brief.lower() or "cite" in brief.lower()),
    ]

    return _run_checks("Prompt Templates", checks)


def test_sample_output_format() -> bool:
    """Validate the structure of a sample agent result."""
    result = SAMPLE_AGENT_RESULT

    checks = [
        ("has text content", len(result.text) > 200),
        ("has tool calls", len(result.tool_calls) == 3),
        ("has timestamp", bool(result.timestamp)),
        ("text has overview", "Overview" in result.text),
        ("text has sources", "Sources" in result.text),
        ("tool call has tavily_search", any(tc["name"] == "tavily_search" for tc in result.tool_calls)),
        ("tool call has fetch_url", any(tc["name"] == "fetch_url" for tc in result.tool_calls)),
    ]

    # Test JSON serialization
    json_output = json.dumps({
        "text": result.text,
        "tool_calls": result.tool_calls,
        "timestamp": result.timestamp,
    }, indent=2)
    parsed = json.loads(json_output)
    checks.append(("json serializable", bool(parsed.get("text"))))
    checks.append(("json has tool_calls", len(parsed.get("tool_calls", [])) == 3))

    return _run_checks("Sample Output Format", checks)


def test_cli_argument_parsing() -> bool:
    """Test that CLI arguments parse correctly without running the agent."""
    from .main import main
    import argparse

    # Test topic mode
    checks = []

    parser = _build_parser()

    # --topic sets the topic
    args = parser.parse_args(["--topic", "AI safety"])
    checks.append(("--topic parsed", args.topic == "AI safety"))
    checks.append(("default depth is deep", args.depth == "deep"))

    # --depth brief
    args = parser.parse_args(["--topic", "test", "--depth", "brief"])
    checks.append(("--depth brief", args.depth == "brief"))

    # --json flag
    args = parser.parse_args(["--topic", "test", "--json"])
    checks.append(("--json flag", args.json is True))

    # --local flag
    args = parser.parse_args(["--local", "hello"])
    checks.append(("--local flag", args.local is True))

    # positional prompt
    args = parser.parse_args(["What is AI?"])
    checks.append(("positional prompt", args.prompt == "What is AI?"))

    return _run_checks("CLI Argument Parsing", checks)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (mirrors main.py logic)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--topic", "-t", type=str, default=None)
    parser.add_argument("--depth", "-d", choices=["brief", "deep"], default="deep")
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--json", "-j", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# Live end-to-end test
# ---------------------------------------------------------------------------

async def test_live_pipeline() -> bool:
    """Run the real agent against a sample topic and validate output."""
    print("\n  Live Pipeline Test")
    print("  " + "-" * 50)
    print(f"  Topic: {SAMPLE_TOPIC}")
    print("  Running agent (this may take 30-60 seconds)...\n")

    agent = Agent(use_managed=True, verbose=True)
    prompt = RESEARCH_PROMPT_BRIEF.format(topic=SAMPLE_TOPIC)

    try:
        result = await agent.run(prompt)
    except Exception as e:
        _print_result("Live Agent Run", False, f"{type(e).__name__}: {e}")
        return False

    checks = [
        ("has text output", len(result.text) > 100),
        ("used tool calls", len(result.tool_calls) > 0),
        ("has timestamp", bool(result.timestamp)),
        ("text mentions AI or model", "AI" in result.text or "model" in result.text.lower()),
    ]

    agent_ok = _run_checks("Live Agent Output", checks)

    # Test JSON serialization of live result
    try:
        json_out = json.dumps({
            "text": result.text,
            "tool_calls": result.tool_calls,
            "timestamp": result.timestamp,
        }, indent=2, default=str)
        parsed = json.loads(json_out)
        json_ok = True
    except Exception:
        json_ok = False

    serial_checks = [
        ("json serializable", json_ok),
        ("json has text", bool(parsed.get("text")) if json_ok else False),
    ]
    serial_ok = _run_checks("Live Output Serialization", serial_checks)

    # Print a snippet of the output
    print("\n" + "=" * 60)
    print("  DEMO: Live Agent Output (first 500 chars)")
    print("=" * 60)
    print(result.text[:500])
    if len(result.text) > 500:
        print(f"\n  ... [{len(result.text) - 500} more characters]")
    print(f"\n  Tool calls made: {len(result.tool_calls)}")
    for tc in result.tool_calls:
        print(f"    - {tc['name']}({json.dumps(tc['input'], default=str)[:60]})")

    return agent_ok and serial_ok


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

def _run_checks(suite: str, checks: list[tuple[str, bool]]) -> bool:
    """Run a list of (name, passed) checks and print results."""
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    all_ok = passed == total

    status = "\033[38;5;42m PASS \033[0m" if all_ok else "\033[38;5;196m FAIL \033[0m"
    print(f"  [{status}] {suite} ({passed}/{total})")

    for name, ok in checks:
        icon = "\033[38;5;42m✓\033[0m" if ok else "\033[38;5;196m✗\033[0m"
        print(f"    {icon} {name}")

    return all_ok


def _print_result(suite: str, ok: bool, detail: str = "") -> None:
    """Print a single test result."""
    status = "\033[38;5;42m PASS \033[0m" if ok else "\033[38;5;196m FAIL \033[0m"
    msg = f"  [{status}] {suite}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agents-test",
        description="Agents Framework — Pipeline Tests",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run live end-to-end test (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Print sample agent output for the demo topic",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Agents Framework — Pipeline Tests")
    print("=" * 60 + "\n")

    if args.demo:
        print("=" * 60)
        print(f"  DEMO: Sample Research Output")
        print(f"  Topic: {SAMPLE_TOPIC}")
        print("=" * 60 + "\n")
        print(SAMPLE_AGENT_RESULT.text)
        print(f"\nTool calls: {len(SAMPLE_AGENT_RESULT.tool_calls)}")
        for tc in SAMPLE_AGENT_RESULT.tool_calls:
            print(f"  - {tc['name']}({json.dumps(tc['input'], default=str)[:60]})")
        return

    # Run offline tests
    results = [
        test_tool_schemas(),
        test_tool_dispatch(),
        test_agent_result_structure(),
        test_prompt_templates(),
        test_sample_output_format(),
        test_cli_argument_parsing(),
    ]

    # Run live test if requested
    if args.live:
        live_ok = asyncio.run(test_live_pipeline())
        results.append(live_ok)

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"  \033[38;5;42mAll {total} test suites passed.\033[0m")
    else:
        print(f"  \033[38;5;196m{total - passed}/{total} test suites failed.\033[0m")
    print(f"{'=' * 60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
