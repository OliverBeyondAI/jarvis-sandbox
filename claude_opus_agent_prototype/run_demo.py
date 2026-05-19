#!/usr/bin/env python3
"""
Run Demo — End-to-end demonstration of the Claude Opus agent prototype.

Usage:
    python -m claude_opus_agent_prototype.run_demo                  # full demo
    python -m claude_opus_agent_prototype.run_demo --query "..."    # custom query
    python -m claude_opus_agent_prototype.run_demo --dry-run        # config check only

Requires:
    - ANTHROPIC_API_KEY environment variable
    - Python 3.11+
    - pip install anthropic httpx
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claude_opus_agent_prototype.agent import OpusAgent, AgentResult
from claude_opus_agent_prototype.config import AgentConfig
from claude_opus_agent_prototype.logging_utils import get_logger, LogLevel


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t: str) -> str: return _c("1", t)
def dim(t: str) -> str: return _c("2", t)
def green(t: str) -> str: return _c("32", t)
def cyan(t: str) -> str: return _c("36", t)
def red(t: str) -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)


def banner(title: str) -> str:
    line = "=" * 60
    return f"\n{line}\n  {bold(title)}\n{line}"


# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    {
        "label": "Web Research",
        "query": (
            "Research the latest developments in agentic AI frameworks "
            "(Claude Agent SDK, OpenAI Agents SDK, LangGraph). "
            "Compare their approaches to tool use and multi-step reasoning. "
            "Write a brief summary to output/agentic_ai_comparison.md."
        ),
    },
    {
        "label": "Multi-Source Analysis",
        "query": (
            "Read the file data/brand_guide.md if it exists. "
            "Then search the web for current best practices in AI-powered "
            "content creation. Synthesize both into 3 actionable recommendations."
        ),
    },
]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

async def run_query(agent: OpusAgent, query: dict, index: int, total: int) -> None:
    """Run a single demo query and display results."""
    label = query.get("label", f"Query {index}")
    prompt = query["query"]

    print(f"\n{dim('-' * 60)}")
    print(f"  {cyan(f'[{index}/{total}]')} {bold(label)}")
    print(f"{dim('-' * 60)}")
    print(f"\n  {bold('Prompt:')} {prompt[:120]}{'...' if len(prompt) > 120 else ''}\n")

    t0 = time.time()
    try:
        result = await agent.run(prompt)
    except Exception as e:
        print(f"\n  {red('Error:')} {e}\n")
        return
    elapsed = time.time() - t0

    # Display response
    print(f"\n  {bold('Response:')}")
    for line in result.text.strip().splitlines():
        print(f"    {line}")

    # Stats
    print(f"\n  {bold('Stats:')}")
    print(f"    Turns used  : {result.turns_used}")
    print(f"    Tool calls  : {len(result.tool_calls)}")
    print(f"    Elapsed     : {elapsed:.1f}s")

    if result.tool_calls:
        print(f"\n  {bold('Tool Log:')}")
        for i, tc in enumerate(result.tool_calls, 1):
            inp = json.dumps(tc["input"], default=str)
            if len(inp) > 80:
                inp = inp[:77] + "..."
            print(f"    {i}. {green(tc['name'])}({dim(inp)})")

    print()


async def run_demo(queries: list[dict]) -> None:
    """Run all demo queries."""
    config = AgentConfig()
    print(banner("Claude Opus Agent Prototype"))
    print(f"\n  Model       : {dim(config.model)}")
    print(f"  Max tokens  : {dim(str(config.max_tokens))}")
    print(f"  Max turns   : {dim(str(config.max_turns))}")
    print(f"  Output dir  : {dim(str(config.output_dir))}")
    print(f"  Queries     : {len(queries)}")

    agent = OpusAgent(config=config)

    for i, q in enumerate(queries, 1):
        await run_query(agent, q, i, len(queries))

    print(banner("Demo Complete"))
    print()


# ---------------------------------------------------------------------------
# Dry-run: config validation
# ---------------------------------------------------------------------------

def dry_run() -> None:
    """Display configuration without making API calls."""
    config = AgentConfig()
    print(banner("Dry Run — Configuration Check"))
    print(f"\n  Model       : {config.model}")
    print(f"  Max tokens  : {config.max_tokens}")
    print(f"  Temperature : {config.temperature}")
    print(f"  Max turns   : {config.max_turns}")
    print(f"  Max retries : {config.max_retries}")
    print(f"  Output dir  : {config.output_dir}")
    print(f"  API key set : {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No'}")
    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    if "--dry-run" in args:
        dry_run()
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"\n  {red('Error:')} ANTHROPIC_API_KEY not set.")
        print(f"  Run with {bold('--dry-run')} to check configuration.\n")
        sys.exit(1)

    if "--query" in args:
        idx = args.index("--query")
        if idx + 1 >= len(args):
            print("Usage: run_demo.py --query \"your query\"")
            sys.exit(1)
        queries = [{"label": "Custom Query", "query": args[idx + 1]}]
    else:
        queries = DEMO_QUERIES

    asyncio.run(run_demo(queries))


if __name__ == "__main__":
    main()
