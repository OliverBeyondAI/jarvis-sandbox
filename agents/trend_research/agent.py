"""
Research Agent — Takes an AI trend as input and uses the Tavily API
to gather relevant articles, papers, and news, returning structured
research findings.

This is the core agent loop that orchestrates Claude + tools to produce
a structured TrendAnalysis from a given trend topic.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import anthropic

from .config import Config
from .models import ResearchReport, Source, Trend, TrendAnalysis, TrendCategory
from .tavily_client import TavilyResearchClient
from .tools import ALL_TOOLS, execute_tool


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

RESEARCH_AGENT_SYSTEM = """You are a research agent specializing in AI and technology trends.

Your task: Given an AI trend topic, use the available tools to gather comprehensive research findings including articles, papers, and news.

## Research Strategy

1. Start with `search_trends` to discover the landscape around the trend.
2. Use `deep_dive` to get in-depth research on specific aspects.
3. Use `search_news` to find the latest developments.
4. Optionally use `fetch_url` to get full content from the most relevant sources.

## Output Requirements

After researching, provide your findings in the following JSON format:

```json
{
  "trend_name": "Name of the trend",
  "category": "ai_ml|biotech|climate|computing|cybersecurity|energy|fintech|healthcare|robotics|space|other",
  "summary": "2-3 sentence summary of the trend",
  "significance": "Why this trend matters",
  "timeline": "Expected timeline for impact",
  "key_players": ["Company/Org 1", "Company/Org 2"],
  "current_state": "Description of where things stand today",
  "opportunities": ["Opportunity 1", "Opportunity 2"],
  "risks": ["Risk 1", "Risk 2"],
  "predictions": ["Prediction 1", "Prediction 2"],
  "confidence": 0.0-1.0,
  "sources": [
    {"title": "...", "url": "...", "snippet": "...", "relevance_score": 0.0-1.0}
  ]
}
```

Be thorough but focused. Gather 5-10 high-quality sources. Prioritize recency and relevance."""


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Agentic research loop that uses Claude to orchestrate Tavily searches
    and produce structured trend analysis.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    async def research_trend(self, trend_topic: str) -> dict[str, Any]:
        """
        Research a given AI trend topic and return structured findings.

        Args:
            trend_topic: The AI trend to research (e.g. "multimodal AI agents",
                         "neuromorphic computing", "AI safety alignment")

        Returns:
            Dictionary with structured research findings including sources,
            analysis, opportunities, and risks.
        """
        print(f"\n[agent] Researching trend: {trend_topic}", file=sys.stderr)

        # Build initial messages
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Research the following AI/technology trend thoroughly: **{trend_topic}**\n\n"
                    f"Use the search tools to gather articles, papers, and news about this trend. "
                    f"Then synthesize your findings into the structured JSON format described in your instructions."
                ),
            }
        ]

        # Convert tool schemas to Anthropic API format
        tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in ALL_TOOLS
        ]

        # Agent loop
        turn = 0
        max_turns = self.config.max_agent_turns

        while turn < max_turns:
            turn += 1
            print(f"[agent] Turn {turn}/{max_turns}", file=sys.stderr)

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=RESEARCH_AGENT_SYSTEM,
                tools=tools,
                messages=messages,
            )

            # Check if we're done (no tool use)
            if response.stop_reason == "end_turn":
                # Extract text response
                text_blocks = [b.text for b in response.content if b.type == "text"]
                final_text = "\n".join(text_blocks)
                return self._parse_findings(final_text, trend_topic)

            # Process tool calls
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                # No tool use and not end_turn — extract what we have
                text_blocks = [b.text for b in response.content if b.type == "text"]
                final_text = "\n".join(text_blocks)
                return self._parse_findings(final_text, trend_topic)

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute tools and collect results
            tool_results: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                print(f"[agent]   Tool call: {block.name}({json.dumps(block.input)[:80]}...)", file=sys.stderr)
                result_str = await execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        # Max turns reached — extract whatever we have
        print(f"[agent] Max turns ({max_turns}) reached", file=sys.stderr)
        return {"trend_name": trend_topic, "error": "Max agent turns reached", "sources": []}

    def _parse_findings(self, text: str, trend_topic: str) -> dict[str, Any]:
        """Parse the agent's final text response into structured findings."""
        # Try to extract JSON from the response
        # Look for JSON block in markdown code fence
        json_match = None
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            json_match = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            candidate = text[start:end].strip()
            if candidate.startswith("{"):
                json_match = candidate
        elif text.strip().startswith("{"):
            # Try raw JSON
            json_match = text.strip()

        if json_match:
            try:
                findings = json.loads(json_match)
                # Ensure trend_name is set
                findings.setdefault("trend_name", trend_topic)
                findings.setdefault("sources", [])
                return findings
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text as findings
        return {
            "trend_name": trend_topic,
            "summary": text[:500],
            "raw_research": text,
            "sources": [],
            "confidence": 0.3,
        }

    async def research_and_build_report(self, trend_topic: str) -> ResearchReport:
        """
        Full pipeline: research a trend and produce a complete ResearchReport.

        Args:
            trend_topic: The AI trend to research.

        Returns:
            A validated ResearchReport with trends, analyses, and sources.
        """
        findings = await self.research_trend(trend_topic)

        # Build Source objects
        sources = [
            Source(
                title=s.get("title", ""),
                url=s.get("url", ""),
                snippet=s.get("snippet", ""),
                relevance_score=min(1.0, max(0.0, float(s.get("relevance_score", 0.5)))),
            )
            for s in findings.get("sources", [])
        ]

        # Map category string to enum
        cat_str = findings.get("category", "other")
        try:
            category = TrendCategory(cat_str)
        except ValueError:
            category = TrendCategory.OTHER

        # Build Trend object
        trend = Trend(
            name=findings.get("trend_name", trend_topic),
            category=category,
            summary=findings.get("summary", ""),
            significance=findings.get("significance", ""),
            timeline=findings.get("timeline", ""),
            key_players=findings.get("key_players", []),
            sources=sources,
            confidence=min(1.0, max(0.0, float(findings.get("confidence", 0.5)))),
            tags=findings.get("tags", []),
        )

        # Build TrendAnalysis
        analysis = TrendAnalysis(
            trend_name=trend.name,
            current_state=findings.get("current_state", ""),
            opportunities=findings.get("opportunities", []),
            risks=findings.get("risks", []),
            predictions=findings.get("predictions", []),
            related_trends=findings.get("related_trends", []),
            raw_research=findings.get("raw_research", ""),
        )

        # Build the report
        report = ResearchReport(
            title=f"Research: {trend.name}",
            domain=findings.get("category", "artificial intelligence"),
            executive_summary=trend.summary,
            trends=[trend],
            analyses=[analysis],
        )

        return report


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

async def run_research(trend_topic: str, config: Config | None = None) -> dict[str, Any]:
    """
    Convenience function to research a trend and return findings.

    Args:
        trend_topic: The AI trend to research.
        config: Optional config override.

    Returns:
        Structured research findings dictionary.
    """
    agent = ResearchAgent(config)
    return await agent.research_trend(trend_topic)


async def run_full_pipeline(trend_topic: str, config: Config | None = None) -> ResearchReport:
    """
    Convenience function to research a trend and return a full report.

    Args:
        trend_topic: The AI trend to research.
        config: Optional config override.

    Returns:
        Complete ResearchReport object.
    """
    agent = ResearchAgent(config)
    return await agent.research_and_build_report(trend_topic)


def main() -> None:
    """CLI entry point for the research agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Research Agent — Investigate an AI trend using Tavily + Claude",
        prog="research-agent",
    )
    parser.add_argument(
        "trend",
        help="The AI trend to research (e.g. 'multimodal AI agents')",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Output a full ResearchReport instead of raw findings",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Store the report to local/S3 storage",
    )
    args = parser.parse_args()

    config = Config.from_env()
    warnings = config.validate()
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}", file=sys.stderr)

    if args.full_report or args.store:
        report = asyncio.run(run_full_pipeline(args.trend, config))

        if args.store:
            from .s3_storage import S3Storage

            storage = S3Storage(config)
            key = report.to_storage_key()
            path = asyncio.run(storage.store_json(key, report.model_dump(mode="json")))
            print(f"\nReport stored at: {path}", file=sys.stderr)

        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        findings = asyncio.run(run_research(args.trend, config))
        print(json.dumps(findings, indent=2, default=str))


if __name__ == "__main__":
    main()
