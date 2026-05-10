#!/usr/bin/env python3
"""
Autonomous Research Agent — CLI Entry Point

Run multi-step autonomous research with query decomposition and synthesis.

Usage:
    python -m autonomous_research_agent "What are the implications of AI agents for enterprise software?"
    python -m autonomous_research_agent --query "Impact of tariffs on semiconductor supply chain" --output ./reports
    python -m autonomous_research_agent --demo ai_agents
    python -m autonomous_research_agent --list-demos
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap


# ---------------------------------------------------------------------------
# Demo queries — pre-built research scenarios for quick testing
# ---------------------------------------------------------------------------

DEMO_QUERIES = {
    "ai_agents": {
        "title": "Enterprise AI Agents Landscape",
        "query": textwrap.dedent("""\
            Research the current landscape of autonomous AI agents for enterprise use.

            Key questions to investigate:
            1. Who are the major players building AI agent platforms (Anthropic, OpenAI, Google, Microsoft, startups)?
            2. What frameworks and SDKs are available for building AI agents?
            3. What enterprise use cases are seeing the most real-world traction?
            4. What are the current limitations, failure modes, and safety concerns?
            5. What is the market size, growth trajectory, and investment landscape?
            6. How are companies handling agent reliability, safety, and monitoring in production?
        """),
    },
    "ai_regulation": {
        "title": "Global AI Regulation Landscape 2025-2026",
        "query": textwrap.dedent("""\
            Research the current state of AI regulation globally.

            Key questions to investigate:
            1. What is the status of the EU AI Act implementation and enforcement?
            2. What AI-related legislation and executive orders exist in the US?
            3. How are China, UK, Japan, and other economies approaching AI governance?
            4. What compliance requirements do companies building AI products face?
            5. What regulatory changes are expected in the next 12 months?
            6. How are regulations affecting AI research and deployment in practice?
        """),
    },
    "quantum_computing": {
        "title": "Quantum Computing Progress & Timeline",
        "query": textwrap.dedent("""\
            Research the current state and near-term outlook for quantum computing.

            Key questions to investigate:
            1. What are the latest breakthroughs in quantum hardware (IBM, Google, IonQ, etc.)?
            2. What is the realistic timeline for quantum advantage in practical applications?
            3. What industries and use cases are most likely to benefit first?
            4. What is the state of quantum software, algorithms, and error correction?
            5. What are the implications for cryptography and cybersecurity?
            6. What should enterprise leaders be doing now to prepare?
        """),
    },
    "climate_tech": {
        "title": "Climate Tech Investment & Innovation",
        "query": textwrap.dedent("""\
            Research the current state of climate technology and clean energy innovation.

            Key questions to investigate:
            1. What are the most promising climate tech sectors attracting investment?
            2. What breakthroughs have occurred in energy storage, solar, and fusion?
            3. How is AI being applied to climate and sustainability challenges?
            4. What is the state of carbon capture and removal technologies?
            5. What policy incentives (IRA, EU Green Deal) are driving adoption?
            6. What are the biggest risks and obstacles to meeting climate targets?
        """),
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_research(
    query: str,
    output_dir: str | None = None,
    model: str | None = None,
    output_json: bool = False,
) -> None:
    """Run the autonomous research agent on a query."""
    from .agent import AutonomousResearchAgent
    from .config import Config

    config = Config.from_env()

    # Override config if needed
    overrides: dict = {}
    if output_dir:
        overrides["output_dir"] = output_dir
    if model:
        overrides["model"] = model
    if overrides:
        config_dict = {
            f.name: getattr(config, f.name)
            for f in config.__dataclass_fields__.values()
        }
        config_dict.update(overrides)
        config = Config(**config_dict)

    # Validate config
    warnings = config.validate()
    for w in warnings:
        print(f"[warning] {w}", file=sys.stderr)

    agent = AutonomousResearchAgent(config=config)

    print("\n" + "=" * 70, file=sys.stderr)
    print("  AUTONOMOUS RESEARCH AGENT", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Query: {query[:100]}...", file=sys.stderr)
    print(f"  Model: {config.model}", file=sys.stderr)
    print(f"  Output: {config.output_dir}", file=sys.stderr)
    print(
        f"  Min searches: {config.min_searches} | "
        f"Min deep dives: {config.min_deep_dives}",
        file=sys.stderr,
    )
    print("=" * 70 + "\n", file=sys.stderr)

    result = await agent.run(query)

    if output_json:
        print(
            json.dumps(
                {
                    "text": result.text,
                    "report_path": result.report_path,
                    "search_count": result.search_count,
                    "fetch_count": result.fetch_count,
                    "analyze_count": result.analyze_count,
                    "findings_count": len(result.findings),
                    "sources_count": len(result.sources),
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
        print(
            f"  Searches: {result.search_count} | "
            f"Pages read: {result.fetch_count} | "
            f"Analysis phases: {result.analyze_count} | "
            f"Findings: {len(result.findings)} | "
            f"Sources: {len(result.sources)}"
        )

        if result.text:
            print(f"\n{'=' * 70}")
            print(result.text)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autonomous_research_agent",
        description=(
            "Autonomous Research Agent — multi-step research with "
            "query decomposition, parallel search, finding synthesis, "
            "and actionable insights."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m autonomous_research_agent "What is the future of AI agents?"
              python -m autonomous_research_agent --query "Enterprise AI adoption trends"
              python -m autonomous_research_agent --demo ai_agents
              python -m autonomous_research_agent --demo ai_regulation --output ./my_reports
              python -m autonomous_research_agent --demo quantum_computing --json
              python -m autonomous_research_agent --list-demos

            Demo topics: ai_agents, ai_regulation, quantum_computing, climate_tech
        """),
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Free-form research query or topic",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Research query (alternative to positional prompt)",
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=list(DEMO_QUERIES.keys()),
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
        help="Output directory for reports (default: ./research_reports)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override Claude model (default: claude-sonnet-4-6-20250514)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result metadata as JSON",
    )

    args = parser.parse_args()

    # List demos
    if args.list_demos:
        print("\nAvailable demo research scenarios:\n")
        for key, demo in DEMO_QUERIES.items():
            print(f"  {key:25s} {demo['title']}")
        print(
            f"\nUsage: python -m autonomous_research_agent --demo <name>\n"
        )
        return

    # Determine the query
    if args.demo:
        demo = DEMO_QUERIES[args.demo]
        query = demo["query"]
        print(f"Running demo: {demo['title']}", file=sys.stderr)
    elif args.query:
        query = args.query
    elif args.prompt:
        query = args.prompt
    else:
        parser.print_help()
        print(
            "\nError: Provide a prompt, --query, or --demo.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(
        run_research(
            query=query,
            output_dir=args.output,
            model=args.model,
            output_json=args.json,
        )
    )


if __name__ == "__main__":
    main()
