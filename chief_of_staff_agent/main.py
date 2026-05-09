#!/usr/bin/env python3
"""
Chief of Staff Agent — CLI Entry Point

Run multi-step research and produce executive briefings.

Usage:
    python -m chief_of_staff_agent "What is the current state of AI regulation in the EU and US?"
    python -m chief_of_staff_agent --brief "Competitive landscape for enterprise AI agents"
    python -m chief_of_staff_agent --brief "Impact of tariffs on semiconductor supply chain" --output ./reports
    python -m chief_of_staff_agent --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap


# ---------------------------------------------------------------------------
# Demo briefs — pre-built research scenarios for quick testing
# ---------------------------------------------------------------------------

DEMO_BRIEFS = {
    "ai_regulation": {
        "title": "AI Regulation Landscape",
        "brief": textwrap.dedent("""\
            Research the current state of AI regulation globally.

            Key questions:
            1. What is the status of the EU AI Act implementation?
            2. What AI-related executive orders or legislation exist in the US?
            3. How are China, UK, and other major economies approaching AI governance?
            4. What are the key compliance requirements for companies building AI products?
            5. What regulatory changes are expected in the next 12 months?
        """),
    },
    "ai_agents": {
        "title": "Enterprise AI Agents Market",
        "brief": textwrap.dedent("""\
            Research the competitive landscape for AI agents in the enterprise.

            Key questions:
            1. Who are the major players building AI agent platforms (Anthropic, OpenAI, Google, Microsoft, startups)?
            2. What are the key differentiators between agent frameworks and platforms?
            3. What enterprise use cases are seeing the most traction?
            4. What are the current limitations and failure modes of AI agents?
            5. What is the market size and growth trajectory for AI agent platforms?
        """),
    },
    "semiconductor_supply": {
        "title": "Semiconductor Supply Chain Analysis",
        "brief": textwrap.dedent("""\
            Research the current state of the global semiconductor supply chain.

            Key questions:
            1. What is the current status of chip supply/demand dynamics?
            2. How are TSMC, Samsung, and Intel fabs progressing on advanced nodes?
            3. What impact are tariffs and export controls having on the industry?
            4. What are the implications for AI hardware availability (GPUs, TPUs)?
            5. What risks should executives be tracking in the next 6-12 months?
        """),
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_research(
    brief: str,
    output_dir: str | None = None,
    model: str | None = None,
    output_json: bool = False,
) -> None:
    """Run the chief of staff agent on a research brief."""
    from .agent import ChiefOfStaffAgent
    from .config import Config

    config = Config.from_env()

    # Override config if needed
    overrides: dict = {}
    if output_dir:
        overrides["output_dir"] = output_dir
    if model:
        overrides["model"] = model
    if overrides:
        # Create new config with overrides (frozen dataclass workaround)
        config_dict = {
            f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()
        }
        config_dict.update(overrides)
        config = Config(**config_dict)

    # Validate config
    warnings = config.validate()
    for w in warnings:
        print(f"[warning] {w}", file=sys.stderr)

    agent = ChiefOfStaffAgent(config=config)

    print("\n" + "=" * 60, file=sys.stderr)
    print("  CHIEF OF STAFF RESEARCH AGENT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Brief: {brief[:80]}...", file=sys.stderr)
    print(f"  Model: {config.model}", file=sys.stderr)
    print(f"  Output: {config.output_dir}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    result = await agent.run(brief)

    if output_json:
        print(
            json.dumps(
                {
                    "text": result.text,
                    "report_path": result.report_path,
                    "search_count": result.search_count,
                    "fetch_count": result.fetch_count,
                    "tool_calls_count": len(result.tool_calls),
                    "duration_seconds": result.duration_seconds,
                    "timestamp": result.timestamp,
                },
                indent=2,
                default=str,
            )
        )
    else:
        if result.report_path:
            print(f"\nReport saved to: {result.report_path}")
        print(f"\nResearch completed in {result.duration_seconds:.1f}s")
        print(f"  Searches: {result.search_count} | Pages read: {result.fetch_count} | Total tool calls: {len(result.tool_calls)}")

        if result.text:
            print(f"\n{'='*60}")
            print(result.text)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chief_of_staff_agent",
        description="Chief of Staff Research Agent — autonomous multi-step research and executive briefings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m chief_of_staff_agent "What is the state of AI regulation?"
              python -m chief_of_staff_agent --brief "Enterprise AI agents competitive landscape"
              python -m chief_of_staff_agent --demo ai_agents
              python -m chief_of_staff_agent --demo ai_regulation --output ./my_reports
              python -m chief_of_staff_agent --list-demos

            Demo topics: ai_regulation, ai_agents, semiconductor_supply
        """),
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Free-form research topic or question",
    )
    parser.add_argument(
        "--brief", "-b",
        type=str,
        default=None,
        help="Research brief (topic + questions)",
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=list(DEMO_BRIEFS.keys()),
        default=None,
        help="Run a pre-built demo research scenario",
    )
    parser.add_argument(
        "--list-demos",
        action="store_true",
        help="List available demo scenarios and exit",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for reports (default: ./chief_of_staff_reports)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override Claude model (default: claude-opus-4-20250514)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result metadata as JSON",
    )

    args = parser.parse_args()

    # List demos
    if args.list_demos:
        print("\nAvailable demo scenarios:\n")
        for key, demo in DEMO_BRIEFS.items():
            print(f"  {key:25s} {demo['title']}")
        print(f"\nUsage: python -m chief_of_staff_agent --demo <name>\n")
        return

    # Determine the brief
    if args.demo:
        demo = DEMO_BRIEFS[args.demo]
        brief = demo["brief"]
        print(f"Running demo: {demo['title']}", file=sys.stderr)
    elif args.brief:
        brief = args.brief
    elif args.prompt:
        brief = args.prompt
    else:
        parser.print_help()
        print(
            "\nError: Provide a prompt, --brief, or --demo.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(
        run_research(
            brief=brief,
            output_dir=args.output,
            model=args.model,
            output_json=args.json,
        )
    )


if __name__ == "__main__":
    main()
