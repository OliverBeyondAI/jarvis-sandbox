"""
CLI entry point for the Trend Research system.

Usage:
    python -m agents.trend_research [--domain DOMAIN] [--local] [--validate]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import Config
from .models import ResearchReport, Trend, TrendCategory
from .s3_storage import S3Storage
from .tavily_client import TavilyResearchClient


async def validate_setup(config: Config) -> bool:
    """Validate that all required services are accessible."""
    warnings = config.validate()
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}", file=sys.stderr)
        return False

    print("[validate] Configuration OK", file=sys.stderr)

    # Test storage
    storage = S3Storage(config)
    test_key = "_validation_test.json"
    path = await storage.store_json(test_key, {"test": True})
    data = await storage.load_json(test_key)
    assert data["test"] is True, "Storage round-trip failed"
    print(f"[validate] Storage OK ({path})", file=sys.stderr)

    print("[validate] All checks passed", file=sys.stderr)
    return True


async def run_demo(config: Config, domain: str) -> None:
    """Run a demo search-and-store cycle to verify the system works."""
    tavily = TavilyResearchClient(config)
    storage = S3Storage(config)

    print(f"\n--- Trend Research Demo: {domain} ---\n", file=sys.stderr)

    # Search for trends
    sources = await tavily.search_trends(domain, max_results=5)
    print(f"Found {len(sources)} sources", file=sys.stderr)

    # Build a report
    trends = [
        Trend(
            name=s.title,
            category=TrendCategory.OTHER,
            summary=s.snippet[:200] if s.snippet else "",
            sources=[s],
            confidence=s.relevance_score,
        )
        for s in sources[:5]
    ]

    report = ResearchReport(
        title=f"Emerging Trends in {domain.title()}",
        domain=domain,
        executive_summary=f"Automated scan of {len(sources)} sources identified {len(trends)} potential trends.",
        trends=trends,
    )

    # Store the report
    key = report.to_storage_key()
    path = await storage.store_json(key, report.model_dump(mode="json"))
    print(f"\nReport stored at: {path}", file=sys.stderr)

    # Output the report JSON to stdout
    print(json.dumps(report.model_dump(mode="json"), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend Research multi-agent system",
        prog="trend-research",
    )
    parser.add_argument(
        "--domain",
        default="artificial intelligence",
        help="Domain to research (default: 'artificial intelligence')",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=True,
        help="Use local filesystem storage (default: True)",
    )
    args = parser.parse_args()

    config = Config.from_env()

    if args.validate:
        ok = asyncio.run(validate_setup(config))
        sys.exit(0 if ok else 1)

    asyncio.run(run_demo(config, args.domain))


if __name__ == "__main__":
    main()
