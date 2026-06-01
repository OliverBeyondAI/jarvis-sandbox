#!/usr/bin/env python3
"""
Agents — Main Entry Point

Run an AI agent with web search and URL fetching tools.

Usage:
    python -m agents "What are the latest developments in AI safety?"
    python -m agents --topic "frontier AI models" --depth deep
    python -m agents --local "Summarize this page"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap


def _import_agent():
    """Lazy import to keep --help fast."""
    from .agent import Agent
    return Agent


RESEARCH_PROMPT_TEMPLATE = textwrap.dedent("""\
    Research the following topic thoroughly using web search. Provide a
    comprehensive summary including:

    1. **Overview** — What is the current state of this topic?
    2. **Key Developments** — What are the most important recent advances?
    3. **Major Players** — Who are the key organizations and researchers?
    4. **Challenges** — What are the open problems and limitations?
    5. **Future Outlook** — Where is this heading in the next 1-2 years?

    Topic: {topic}

    Search for the latest information and cite your sources with URLs.
""")

RESEARCH_PROMPT_BRIEF = textwrap.dedent("""\
    Research the following topic using web search. Provide a concise summary
    covering the current state, key developments, and major players.

    Topic: {topic}

    Cite your sources with URLs.
""")


async def run_agent(
    prompt: str,
    use_managed: bool = True,
    model: str | None = None,
    output_json: bool = False,
) -> None:
    """Run the agent with the given prompt and print the result."""
    Agent = _import_agent()

    kwargs: dict = {"use_managed": use_managed}
    if model:
        kwargs["model"] = model

    agent = Agent(**kwargs)
    print(f"\nAgent running...\n", file=sys.stderr)

    result = await agent.run(prompt)

    if output_json:
        print(json.dumps({
            "text": result.text,
            "tool_calls": result.tool_calls,
            "timestamp": result.timestamp,
        }, indent=2, default=str))
    else:
        print(result.text)

    if result.tool_calls:
        print(
            f"\n[Used {len(result.tool_calls)} tool call(s)]",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agents",
        description="AI Agent with web search and URL fetching tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m agents "What is the latest on AI regulation?"
              python -m agents --topic "frontier AI models and agentic frameworks"
              python -m agents --topic "quantum computing" --depth brief
              python -m agents --local --topic "LLM scaling laws" --json
              python -m agents --model claude-opus-4-7-20250501 "Hello"
        """),
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send to the agent (mutually exclusive with --topic)",
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help="Research topic — generates a structured research prompt",
    )
    parser.add_argument(
        "--depth", "-d",
        choices=["brief", "deep"],
        default="deep",
        help="Research depth: brief (concise) or deep (comprehensive, default)",
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
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON (includes tool call metadata)",
    )

    args = parser.parse_args()

    # Build the prompt from either --topic or positional argument
    if args.topic:
        template = RESEARCH_PROMPT_BRIEF if args.depth == "brief" else RESEARCH_PROMPT_TEMPLATE
        prompt = template.format(topic=args.topic)
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        print("\nError: Provide a prompt or use --topic <research topic>.", file=sys.stderr)
        sys.exit(1)

    use_managed = not args.local
    asyncio.run(run_agent(prompt, use_managed=use_managed, model=args.model, output_json=args.json))


if __name__ == "__main__":
    main()
