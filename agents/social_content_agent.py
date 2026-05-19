#!/usr/bin/env python3
"""
Social Content Agent — Research a topic and draft brand-aligned social media posts.

Uses the Claude Agent SDK (via the agents.Agent facade) with web search, file I/O,
and URL fetching tools to:
  1. Read the brand guide from the local filesystem.
  2. Search the web for current information on the given topic.
  3. Draft a concise research summary.
  4. Produce ready-to-publish social media posts aligned to brand voice.

Usage:
    python -m agents.social_content_agent "AI in healthcare"
    python -m agents.social_content_agent --topic "quantum computing breakthroughs" --platform linkedin
    python -m agents.social_content_agent --topic "remote work trends" --platform twitter --output ./data/posts.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent import Agent, AgentResult
from .tools import DATA_DIR, read_file, write_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BRAND_GUIDE_PATH = DATA_DIR / "brand_guide.md"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Social Content Agent for Acme Corp. Your job is to research a
    given topic using web search, then produce:

    1. **Research Summary** — A concise 3-5 paragraph summary of the most
       important and current findings on the topic. Include key statistics,
       trends, and notable developments. Cite sources inline.

    2. **Social Media Posts** — Draft posts tailored to the requested platform(s),
       strictly following the brand guide loaded from the filesystem.

    ## Tools Available

    1. **tavily_search** — Search the web for current information. Use 2-4
       targeted queries to cover different angles of the topic.
    2. **fetch_url** — Fetch the full content of a web page for deeper reading.
    3. **read_file** — Read a file from the local filesystem. Use this to load
       the brand guide and any reference materials.
    4. **write_file** — Write content to a file. Use this to save your final
       output (summary + posts) to disk.

    ## Workflow

    1. First, use `read_file` to load the brand guide from the path provided
       in the user's message.
    2. Search the web for current information on the topic (2-4 queries).
    3. Optionally fetch 1-2 key URLs for deeper context.
    4. Synthesize your research into a structured summary.
    5. Draft social media posts following the brand guide's templates and voice.
    6. Use `write_file` to save the complete output.

    ## Output Format

    Your final output should follow this structure:

    ---
    # Topic Research & Social Content

    ## Research Summary
    [Your 3-5 paragraph summary with inline citations]

    ## Sources
    - [Source 1 title](url)
    - [Source 2 title](url)
    ...

    ## Social Media Posts

    ### LinkedIn
    [Post content following brand template]

    ### Twitter/X
    [Post content following brand template]
    ---

    Be specific, data-driven, and on-brand. Never fabricate statistics.
""")


# ---------------------------------------------------------------------------
# Social Content Agent
# ---------------------------------------------------------------------------

@dataclass
class ContentResult:
    """Structured output from the social content agent."""
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    output_path: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SocialContentAgent:
    """
    Agent that researches a topic and produces brand-aligned social content.

    Uses the Claude Agent SDK (via the agents.Agent facade) which provides
    managed-agent execution with automatic fallback to a local agentic loop.
    """

    def __init__(self, verbose: bool = True):
        self._agent = Agent(
            system_prompt=SYSTEM_PROMPT,
            verbose=verbose,
        )
        self.verbose = verbose

    async def run(
        self,
        topic: str,
        platform: str = "all",
        output_path: str | None = None,
    ) -> ContentResult:
        """
        Research topic and generate social content.

        Args:
            topic: The subject to research.
            platform: Target platform(s) — "linkedin", "twitter", or "all".
            output_path: Optional path to save the output file.
        """
        if output_path is None:
            slug = topic.lower().replace(" ", "_")[:30]
            output_path = str(DATA_DIR / f"social_content_{slug}.md")

        user_prompt = self._build_prompt(topic, platform, output_path)

        self._log(f"Topic: {topic}")
        self._log(f"Platform: {platform}")
        self._log(f"Output: {output_path}")
        self._log(f"Brand guide: {BRAND_GUIDE_PATH}")
        self._log("-" * 60)

        agent_result: AgentResult = await self._agent.run(user_prompt)

        return ContentResult(
            text=agent_result.text,
            tool_calls=agent_result.tool_calls,
            output_path=output_path,
        )

    def _build_prompt(self, topic: str, platform: str, output_path: str) -> str:
        """Construct the user prompt with context."""
        platform_instruction = {
            "all": "Draft posts for both LinkedIn and Twitter/X.",
            "linkedin": "Draft a post for LinkedIn only.",
            "twitter": "Draft a post for Twitter/X only.",
        }.get(platform, "Draft posts for both LinkedIn and Twitter/X.")

        return textwrap.dedent(f"""\
            ## Task

            Research the following topic and produce brand-aligned social media content.

            **Topic:** {topic}

            **Platform:** {platform_instruction}

            **Brand Guide Location:** {BRAND_GUIDE_PATH}

            **Output File:** {output_path}

            ## Instructions

            1. Load the brand guide from `{BRAND_GUIDE_PATH}`.
            2. Search the web for current, relevant information about: "{topic}"
            3. Synthesize your findings into a concise research summary.
            4. Draft social media posts following the brand guide's voice, templates, and hashtag strategy.
            5. Save the complete output (summary + posts) to `{output_path}`.
        """)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SocialContentAgent] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Demo mode (local fallback with pre-built scenario)
# ---------------------------------------------------------------------------

DEMO_RESEARCH = textwrap.dedent("""\
# Topic Research & Social Content

## Research Summary

Agentic AI development is accelerating rapidly in 2026, with major platforms
releasing frameworks that allow AI systems to autonomously plan, execute, and
iterate on complex tasks. Key developments include:

**Multi-agent orchestration** has become the dominant paradigm, with systems
like Claude Agent SDK, LangGraph, and CrewAI enabling developers to compose
specialized agents that collaborate on workflows. Enterprise adoption grew
47% year-over-year according to recent industry surveys.

**Tool use and function calling** capabilities have matured significantly.
Modern agents can reliably chain 10-20+ tool calls in sequence, handling
errors gracefully and adapting their strategies mid-execution. This has
unlocked production use cases in code generation, research synthesis, and
customer support automation.

**Safety and controllability** remain active research areas. New approaches
include constitutional AI for agent behavior, sandboxed execution environments,
and human-in-the-loop checkpoints for high-stakes decisions.

## Sources
- [State of AI Agents 2026 — Industry Report](https://example.com/ai-agents-2026)
- [Claude Agent SDK Documentation](https://docs.anthropic.com/agent-sdk)
- [Enterprise AI Adoption Survey Q1 2026](https://example.com/enterprise-ai-survey)

## Social Media Posts

### LinkedIn

The age of agentic AI isn't coming — it's here.

In the past year, we've seen a 47% increase in enterprise adoption of
multi-agent systems. These aren't simple chatbots. They're autonomous
workflows that plan, execute, and self-correct across dozens of tool calls.

At Acme Corp, we're building with these capabilities today. The question
isn't whether agents will transform knowledge work — it's how quickly your
team will adapt.

What's the most surprising agent use case you've encountered?

#BuildTheFuture #AIInsights #FutureOfWork

### Twitter/X

Agentic AI hit a tipping point in 2026: 47% more enterprises running
multi-agent workflows vs. last year.

These systems chain 10-20+ tool calls, self-correct on errors, and
collaborate across specialized roles.

We're not talking about chatbots anymore.

What are you building with agents?

#AcmeCorp #AIInsights
""")


async def run_demo(topic: str, platform: str, output_path: str) -> ContentResult:
    """Run a demo with pre-built content (no API calls required)."""
    print("[SocialContentAgent] Running in DEMO mode (no API key required)", file=sys.stderr)
    print(f"[SocialContentAgent] Topic: {topic}", file=sys.stderr)
    print(f"[SocialContentAgent] Platform: {platform}", file=sys.stderr)

    # Simulate reading the brand guide
    brand_guide = read_file(str(BRAND_GUIDE_PATH))
    if "error" in brand_guide:
        print(f"[SocialContentAgent] Warning: {brand_guide['error']}", file=sys.stderr)
    else:
        print(f"[SocialContentAgent] Brand guide loaded ({brand_guide['size']} chars)", file=sys.stderr)

    # Write demo output
    write_result = write_file(output_path, DEMO_RESEARCH)
    print(f"[SocialContentAgent] Output written to: {write_result['path']}", file=sys.stderr)

    return ContentResult(
        text=DEMO_RESEARCH,
        output_path=write_result["path"],
        tool_calls=[
            {"name": "read_file", "input": {"path": str(BRAND_GUIDE_PATH)}},
            {"name": "tavily_search", "input": {"query": f"{topic} 2026 trends"}},
            {"name": "tavily_search", "input": {"query": f"{topic} enterprise adoption statistics"}},
            {"name": "write_file", "input": {"path": output_path, "content": "(demo)"}},
        ],
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Social Content Agent — Research topics and draft brand-aligned posts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m agents.social_content_agent "AI in healthcare"
              python -m agents.social_content_agent --topic "quantum computing" --platform linkedin
              python -m agents.social_content_agent --topic "remote work" --demo
        """),
    )
    parser.add_argument("topic_positional", nargs="?", help="Topic to research (positional)")
    parser.add_argument("--topic", "-t", help="Topic to research")
    parser.add_argument(
        "--platform", "-p",
        choices=["all", "linkedin", "twitter"],
        default="all",
        help="Target platform (default: all)",
    )
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--demo", action="store_true", help="Run with pre-built demo content (no API key needed)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress logs")

    args = parser.parse_args()
    topic = args.topic or args.topic_positional

    if not topic:
        parser.error("Please provide a topic to research.")

    # Determine output path
    slug = topic.lower().replace(" ", "_")[:30]
    output_path = args.output or str(DATA_DIR / f"social_content_{slug}.md")

    # Run in demo mode or live mode
    if args.demo or not os.environ.get("ANTHROPIC_API_KEY"):
        result = await run_demo(topic, args.platform, output_path)
    else:
        agent = SocialContentAgent(verbose=not args.quiet)
        result = await agent.run(topic, args.platform, output_path)

    # Print final output
    print("\n" + "=" * 60)
    print("SOCIAL CONTENT AGENT — OUTPUT")
    print("=" * 60)
    print(result.text)
    print("=" * 60)
    print(f"\nTool calls made: {len(result.tool_calls)}")
    print(f"Output saved to: {result.output_path}")
    print(f"Timestamp: {result.timestamp}")


if __name__ == "__main__":
    asyncio.run(main())
