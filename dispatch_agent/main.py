#!/usr/bin/env python3
"""
Dispatch Agent — CLI Entry Point

Run a multi-agent research pipeline that decomposes a topic, dispatches
parallel sub-agents via channels, and synthesizes a unified report.

Usage:
    python -m dispatch_agent "Impact of AI on healthcare diagnostics"
    python -m dispatch_agent --max-agents 3 "Quantum computing advances 2025"
    python -m dispatch_agent --json "Latest developments in fusion energy"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap

from .dispatcher import Dispatcher


async def run_research(
    topic: str,
    max_agents: int = 5,
    output_json: bool = False,
    model: str | None = None,
) -> None:
    """Run the dispatch research pipeline and print results."""
    kwargs: dict = {
        "max_sub_agents": max_agents,
        "verbose": not output_json,
    }
    if model:
        kwargs["model"] = model

    dispatcher = Dispatcher(**kwargs)

    if not output_json:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  DISPATCH RESEARCH AGENT", file=sys.stderr)
        print(f"  Topic: {topic}", file=sys.stderr)
        print(f"  Max sub-agents: {max_agents}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

    result = await dispatcher.research(topic)

    if output_json:
        # Structured JSON output
        output = {
            "topic": result.topic,
            "report": result.report,
            "metadata": {
                "sub_queries": result.sub_queries,
                "total_tool_calls": result.total_tool_calls,
                "total_duration_seconds": round(result.total_duration_seconds, 2),
                "timestamp": result.timestamp,
                "sub_agents": [
                    {
                        "agent_id": r.agent_id,
                        "query": r.query,
                        "tool_calls": r.tool_calls_count,
                        "duration_seconds": round(r.duration_seconds, 2),
                        "sources_count": len(r.sources),
                        "error": r.error,
                    }
                    for r in result.sub_agent_results
                ],
            },
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        # Human-readable output
        _print_report(result)


def _print_report(result) -> None:
    """Print a formatted research report to stdout."""
    report = result.report

    if not report:
        print("\n--- Raw Report ---\n")
        print(result.raw_text)
        return

    title = report.get("title", f"Research Report: {result.topic}")
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

    # Executive summary
    summary = report.get("executive_summary", "")
    if summary:
        print("EXECUTIVE SUMMARY")
        print("-" * 40)
        print(textwrap.fill(summary, width=72))
        print()

    # Sections
    for section in report.get("sections", []):
        heading = section.get("heading", "Untitled Section")
        print(f"\n## {heading}")
        print("-" * 40)
        content = section.get("content", "")
        for paragraph in content.split("\n\n"):
            print(textwrap.fill(paragraph.strip(), width=72))
            print()
        sources = section.get("sources", [])
        if sources:
            print("  Sources:")
            for s in sources:
                print(f"    - {s.get('title', 'Untitled')}: {s.get('url', '')}")
            print()

    # Cross-cutting themes
    themes = report.get("cross_cutting_themes", [])
    if themes:
        print("\nCROSS-CUTTING THEMES")
        print("-" * 40)
        for t in themes:
            print(f"  * {t}")
        print()

    # Key takeaways
    takeaways = report.get("key_takeaways", [])
    if takeaways:
        print("KEY TAKEAWAYS")
        print("-" * 40)
        for i, t in enumerate(takeaways, 1):
            print(f"  {i}. {t}")
        print()

    # Gaps
    gaps = report.get("gaps_and_limitations", [])
    if gaps:
        print("GAPS & LIMITATIONS")
        print("-" * 40)
        for g in gaps:
            print(f"  - {g}")
        print()

    # Follow-ups
    followups = report.get("suggested_followups", [])
    if followups:
        print("SUGGESTED FOLLOW-UPS")
        print("-" * 40)
        for f in followups:
            print(f"  ? {f}")
        print()

    # Metadata footer
    print(f"{'='*60}")
    print(f"  Sub-agents: {len(result.sub_agent_results)}")
    print(f"  Total tool calls: {result.total_tool_calls}")
    print(f"  Duration: {result.total_duration_seconds:.1f}s")
    print(f"  Timestamp: {result.timestamp}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dispatch_agent",
        description="Multi-agent research pipeline with parallel dispatch via channels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m dispatch_agent "Impact of AI on healthcare"
              python -m dispatch_agent --max-agents 3 "Quantum computing 2025"
              python -m dispatch_agent --json "Fusion energy progress"
              python -m dispatch_agent --model claude-opus-4-7-20250501 "AI safety"
        """),
    )

    parser.add_argument(
        "topic",
        nargs="?",
        help="The research topic to investigate",
    )
    parser.add_argument(
        "--max-agents", "-n",
        type=int,
        default=5,
        help="Maximum number of parallel sub-agents (default: 5)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        dest="output_json",
        help="Output structured JSON instead of formatted text",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override the Claude model (default: claude-opus-4-7-20250501)",
    )

    args = parser.parse_args()

    if not args.topic:
        parser.print_help()
        print("\nError: No research topic provided.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(
        run_research(
            args.topic,
            max_agents=args.max_agents,
            output_json=args.output_json,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
