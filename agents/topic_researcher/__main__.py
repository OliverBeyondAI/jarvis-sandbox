"""
CLI entry point for the Topic Researcher agent.

Usage:
    python -m agents.topic_researcher "summarize the latest AI image generation models"
    python -m agents.topic_researcher --topic "quantum computing 2026" --depth brief
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import TopicResearcher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research a topic using Claude + Tavily web search.",
    )
    parser.add_argument(
        "topic_positional",
        nargs="?",
        default=None,
        help="The topic to research (positional).",
    )
    parser.add_argument(
        "--topic", "-t",
        default=None,
        help="The topic to research (named argument).",
    )
    parser.add_argument(
        "--depth", "-d",
        choices=["brief", "standard", "deep"],
        default="standard",
        help="Research depth (default: standard).",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-opus-4-7-20250501",
        help="Claude model to use.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of markdown.",
    )

    args = parser.parse_args()

    topic = args.topic or args.topic_positional
    if not topic:
        parser.error("Please provide a topic to research.")
        sys.exit(1)

    researcher = TopicResearcher(model=args.model)
    result = researcher.research(topic=topic, depth=args.depth)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.to_markdown())


if __name__ == "__main__":
    main()
