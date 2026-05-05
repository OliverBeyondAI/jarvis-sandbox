"""
CLI entry point for the Orchestrator — chains all three agents end-to-end.

Usage:
    python -m agents.orchestrator "multimodal AI agents"
    python -m agents.orchestrator --topic "agentic healthcare AI" --output ./output
    python -m agents.orchestrator --topic "LLM scaling" --audience executive --json
    python -m agents.orchestrator --validate
    python -m agents.orchestrator --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap

from ..memo_generation.models import MemoAudience
from .pipeline import Orchestrator, PipelineResult, run_pipeline


# ---------------------------------------------------------------------------
# Demo mode (uses demo data from agents to show orchestration without API calls)
# ---------------------------------------------------------------------------


async def run_demo() -> None:
    """Run the pipeline with demo data to verify orchestration logic."""
    from ..memo_generation.main import _build_demo_research_report, _build_demo_synthesis_report
    from ..memo_generation.agent import MemoGenerationAgent
    from ..memo_generation.config import MemoConfig

    print("\n" + "=" * 60, file=sys.stderr)
    print("  Orchestrator — Demo Mode (no API calls)", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    # Simulate Stage 1 output
    print("[demo] Stage 1: Using built-in demo ResearchReport", file=sys.stderr)
    research_report = _build_demo_research_report()
    print(f"  Report: {research_report.title}", file=sys.stderr)
    print(f"  Trends: {len(research_report.trends)}", file=sys.stderr)

    # Simulate Stage 2 output
    print("[demo] Stage 2: Using built-in demo SynthesisReport", file=sys.stderr)
    synthesis_report = _build_demo_synthesis_report()
    print(f"  Report: {synthesis_report.title}", file=sys.stderr)
    print(f"  Applications: {sum(len(ts.applications) for ts in synthesis_report.trend_syntheses)}", file=sys.stderr)

    # Run Stage 3 for real (memo generation from structured data)
    print("[demo] Stage 3: Generating memo from demo data...", file=sys.stderr)
    config = MemoConfig.from_env()
    agent = MemoGenerationAgent(config)
    bundle = await agent.run_full_pipeline(
        research_report=research_report,
        synthesis_report=synthesis_report,
        audience=MemoAudience.PRODUCT,
    )

    print(f"\n[demo] Pipeline complete!", file=sys.stderr)
    print(f"[demo] Bundle ID: {bundle.bundle_id}", file=sys.stderr)
    print(f"[demo] Memo (MD): {bundle.memo_path}", file=sys.stderr)
    print(f"[demo] Memo (HTML): {bundle.memo_html_path}", file=sys.stderr)

    # Output bundle manifest
    print(json.dumps(bundle.model_dump(mode="json"), indent=2, default=str))


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="AI Trend Research Pipeline — Orchestrates Research → Synthesis → Memo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m agents.orchestrator "multimodal AI agents"
              python -m agents.orchestrator --topic "agentic healthcare AI"
              python -m agents.orchestrator --topic "LLM scaling" --output ./reports
              python -m agents.orchestrator --topic "AI safety" --audience executive
              python -m agents.orchestrator --demo
              python -m agents.orchestrator --validate
        """),
    )

    parser.add_argument(
        "topic",
        nargs="?",
        help="The AI trend topic to research (e.g. 'multimodal AI agents')",
    )
    parser.add_argument(
        "--topic", "-t",
        dest="topic_flag",
        type=str,
        default=None,
        help="Alternative way to specify topic (--topic 'agentic AI')",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./pipeline_output",
        help="Output directory for generated files (default: ./pipeline_output)",
    )
    parser.add_argument(
        "--audience", "-a",
        choices=["product", "executive", "engineering"],
        default="product",
        help="Target audience for the memo (default: product)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output full pipeline result as JSON",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all agent configurations and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo data (skips Research and Synthesis API calls)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output to stderr",
    )

    args = parser.parse_args()

    # Resolve topic from positional or flag
    topic = args.topic_flag or args.topic

    # Validate mode
    if args.validate:
        orchestrator = Orchestrator(verbose=True)
        warnings = orchestrator.validate()
        if warnings:
            print("Configuration warnings:", file=sys.stderr)
            for w in warnings:
                print(f"  WARNING: {w}", file=sys.stderr)
            sys.exit(1)
        else:
            print("[validate] All configurations OK", file=sys.stderr)
            sys.exit(0)

    # Demo mode
    if args.demo:
        asyncio.run(run_demo())
        return

    # Require a topic for live pipeline
    if not topic:
        parser.print_help()
        print("\nError: Provide a topic as positional arg or via --topic.", file=sys.stderr)
        sys.exit(1)

    # Map audience string to enum
    audience_map = {
        "product": MemoAudience.PRODUCT,
        "executive": MemoAudience.EXECUTIVE,
        "engineering": MemoAudience.ENGINEERING,
    }
    audience = audience_map[args.audience]

    # Run the pipeline
    verbose = not args.quiet
    result = asyncio.run(
        run_pipeline(
            topic=topic,
            output_dir=args.output,
            audience=audience,
            verbose=verbose,
        )
    )

    # Output
    if args.json:
        output = {
            "topic": result.topic,
            "success": result.success,
            "total_duration_seconds": result.total_duration_seconds,
            "started_at": result.started_at,
            "stages": [
                {
                    "name": s.name,
                    "success": s.success,
                    "duration_seconds": s.duration_seconds,
                    "error": s.error,
                }
                for s in result.stages
            ],
        }
        if result.artifact_bundle:
            output["artifact_bundle"] = result.artifact_bundle.model_dump(mode="json")
        print(json.dumps(output, indent=2, default=str))
    else:
        print("\n" + result.summary())

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
