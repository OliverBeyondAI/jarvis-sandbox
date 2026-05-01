#!/usr/bin/env python3
"""
Research Summarizer Agent — Main Entry Point

Accepts a list of URLs as input and produces structured research summaries
using Claude and Tavily.

Usage:
    # Summarize specific URLs:
    python -m research_summarizer.main \
        https://arxiv.org/abs/2301.00001 \
        https://example.com/article

    # Summarize URLs with a topic focus:
    python -m research_summarizer.main \
        --topic "transformer architectures" \
        https://arxiv.org/abs/2301.00001

    # Read URLs from a file (one per line):
    python -m research_summarizer.main --file urls.txt

    # Interactive mode:
    python -m research_summarizer.main --interactive

    # Output as JSON:
    python -m research_summarizer.main --json \
        https://example.com/article
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from typing import Any


def _import_agent():
    """Lazy import to keep --help fast."""
    from .agent import ResearchSummarizerAgent
    return ResearchSummarizerAgent


def _import_formatter():
    """Lazy import to keep --help fast."""
    from .formatter import format_output
    return format_output


async def run_summarize(
    urls: list[str],
    topic: str | None,
    fmt: str = "terminal",
    output_file: str | None = None,
    use_managed: bool = True,
) -> None:
    """Run the summarizer on the given URLs and print the result."""
    Agent = _import_agent()
    format_output = _import_formatter()
    agent = Agent(use_managed=use_managed)

    print(f"\nResearch Summarizer Agent")
    print(f"Analyzing {len(urls)} URL(s)...\n")

    result = await agent.summarize(urls, topic=topic)
    formatted = format_output(result, fmt=fmt)

    if output_file:
        with open(output_file, "w") as f:
            f.write(formatted)
        print(f"Output written to: {output_file}", file=sys.stderr)
    else:
        print(formatted)


async def run_interactive() -> None:
    """Launch the agent in interactive mode."""
    Agent = _import_agent()
    agent = Agent()
    await agent.run_interactive()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="research-summarizer",
        description="Research Summarizer Agent — AI-powered URL research and summarization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m research_summarizer.main https://example.com/article
              python -m research_summarizer.main --topic "AI safety" url1 url2
              python -m research_summarizer.main --file urls.txt
              python -m research_summarizer.main --interactive
              python -m research_summarizer.main --json https://example.com/article
        """),
    )

    parser.add_argument(
        "urls",
        nargs="*",
        help="URLs to fetch and summarize",
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help="Optional topic or focus area to guide summarization",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a file containing URLs (one per line)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch the agent in interactive mode",
    )
    parser.add_argument(
        "--format",
        choices=["terminal", "markdown", "json"],
        default="terminal",
        help="Output format: terminal (colored), markdown, or json (default: terminal)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Shorthand for --format json",
    )
    parser.add_argument(
        "--markdown", "--md",
        action="store_true",
        help="Shorthand for --format markdown",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write output to a file instead of stdout",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override the Claude model (default: claude-opus-4-7-20250501)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local agent loop instead of managed-agents API",
    )

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(run_interactive())
        return

    # Collect URLs from arguments and/or file
    urls: list[str] = list(args.urls) if args.urls else []

    if args.file:
        try:
            with open(args.file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urls.append(line)
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)

    if not urls:
        parser.print_help()
        print("\nError: No URLs provided. Pass URLs as arguments, use --file, or use --interactive.", file=sys.stderr)
        sys.exit(1)

    # Resolve output format
    fmt = args.format
    if args.json:
        fmt = "json"
    elif args.markdown:
        fmt = "markdown"

    use_managed = not args.local
    asyncio.run(run_summarize(urls, topic=args.topic, fmt=fmt, output_file=args.output, use_managed=use_managed))


if __name__ == "__main__":
    main()
