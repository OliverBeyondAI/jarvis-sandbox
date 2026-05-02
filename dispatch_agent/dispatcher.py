#!/usr/bin/env python3
"""
Dispatcher — Orchestrates parallel research sub-agents via channels.

The dispatcher implements a three-phase research pipeline:

  1. **Decompose** — Break the user's topic into 3-5 focused sub-queries
     using Claude for intelligent query decomposition.
  2. **Dispatch** — Fan out sub-queries to parallel research sub-agents
     via a FanOutChannel; each agent searches, reads, and synthesizes.
  3. **Synthesize** — Collect all sub-agent results and produce a
     unified research report with cross-referenced findings.

Supports both managed-agents API and local async runner modes,
with automatic failover (following the agents/ pattern).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anthropic

from .channels import FanOutChannel, SubAgentResult, SubAgentTask
from .sub_agents import run_sub_agent_worker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7-20250501"
MAX_TOKENS = 8192

DECOMPOSE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a research query decomposition specialist. Given a broad
    research topic, break it down into 3-5 focused, non-overlapping
    sub-queries that together provide comprehensive coverage of the topic.

    Each sub-query should:
    - Target a distinct aspect or angle of the topic
    - Be specific enough for a web search to return relevant results
    - Be self-contained (understandable without the other sub-queries)

    Respond with ONLY a JSON array of objects:

    ```json
    [
      {"id": "sub_1", "query": "Specific research question 1", "rationale": "Why this angle matters"},
      {"id": "sub_2", "query": "Specific research question 2", "rationale": "Why this angle matters"}
    ]
    ```
""")

SYNTHESIZE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a research synthesis specialist. You receive findings from
    multiple parallel research sub-agents, each investigating a different
    aspect of a topic. Your job is to produce a unified, well-structured
    research report.

    ## Output Format

    Produce a JSON object with this structure:

    ```json
    {
      "title": "Research Report: [Topic]",
      "executive_summary": "2-3 sentence overview of the key findings",
      "sections": [
        {
          "heading": "Section title",
          "content": "Detailed findings for this aspect (2-4 paragraphs)",
          "sources": [{"title": "...", "url": "..."}]
        }
      ],
      "cross_cutting_themes": ["Theme that appears across multiple sub-topics"],
      "key_takeaways": ["Takeaway 1", "Takeaway 2"],
      "gaps_and_limitations": ["What wasn't well-covered or needs more research"],
      "suggested_followups": ["Follow-up question 1", "Follow-up question 2"]
    }
    ```

    Guidelines:
    - Integrate findings across sub-agents — don't just concatenate them.
    - Highlight agreements and contradictions between sources.
    - Note where information is uncertain or contested.
    - Cite sources accurately using the URLs provided.
""")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Final output from the dispatch research pipeline."""
    topic: str
    report: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    sub_agent_results: list[SubAgentResult] = field(default_factory=list)
    sub_queries: list[dict[str, str]] = field(default_factory=list)
    total_tool_calls: int = 0
    total_duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class Dispatcher:
    """
    Coordinates parallel research sub-agents via channels.

    Usage:
        dispatcher = Dispatcher()
        result = await dispatcher.research("Impact of AI on healthcare")
    """

    def __init__(
        self,
        model: str = MODEL,
        max_sub_agents: int = 5,
        sub_agent_timeout: float = 120.0,
        verbose: bool = True,
    ):
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_sub_agents = max_sub_agents
        self.sub_agent_timeout = sub_agent_timeout
        self.verbose = verbose

    async def research(self, topic: str) -> DispatchResult:
        """
        Execute the full research pipeline:
        decompose → dispatch → synthesize.
        """
        start = time.monotonic()
        self._log(f"Starting research pipeline for: {topic}")

        # Phase 1: Decompose
        self._log("\n━━━ Phase 1: Query Decomposition ━━━")
        sub_queries = await self._decompose(topic)
        self._log(f"  Decomposed into {len(sub_queries)} sub-queries:")
        for sq in sub_queries:
            self._log(f"    • [{sq['id']}] {sq['query']}")

        # Phase 2: Dispatch parallel research
        self._log("\n━━━ Phase 2: Parallel Research Dispatch ━━━")
        sub_results = await self._dispatch(sub_queries)
        total_tools = sum(r.tool_calls_count for r in sub_results)
        self._log(f"  All {len(sub_results)} sub-agents completed ({total_tools} total tool calls)")

        # Phase 3: Synthesize
        self._log("\n━━━ Phase 3: Synthesis ━━━")
        report_text = await self._synthesize(topic, sub_queries, sub_results)

        duration = time.monotonic() - start
        self._log(f"\n━━━ Pipeline complete in {duration:.1f}s ━━━")

        # Parse report JSON
        report = self._extract_json(report_text) or {}

        return DispatchResult(
            topic=topic,
            report=report,
            raw_text=report_text,
            sub_agent_results=sub_results,
            sub_queries=sub_queries,
            total_tool_calls=total_tools,
            total_duration_seconds=duration,
        )

    # ----- Phase 1: Decompose -----

    async def _decompose(self, topic: str) -> list[dict[str, str]]:
        """Use Claude to decompose a topic into focused sub-queries."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=DECOMPOSE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Decompose this research topic into focused sub-queries:\n\n{topic}",
                }
            ],
        )

        text = self._response_text(response)
        queries = self._extract_json(text)

        if not isinstance(queries, list) or len(queries) == 0:
            # Fallback: generate simple sub-queries
            self._log("  Warning: Could not parse decomposition, using fallback")
            return [
                {"id": "sub_1", "query": f"Overview and current state of {topic}", "rationale": "Baseline understanding"},
                {"id": "sub_2", "query": f"Recent developments and trends in {topic}", "rationale": "Latest information"},
                {"id": "sub_3", "query": f"Key challenges and criticisms of {topic}", "rationale": "Balanced perspective"},
            ]

        # Cap at max_sub_agents
        return queries[: self.max_sub_agents]

    # ----- Phase 2: Dispatch -----

    async def _dispatch(
        self, sub_queries: list[dict[str, str]]
    ) -> list[SubAgentResult]:
        """
        Fan out sub-queries to parallel research sub-agents via channels.
        Each sub-agent runs concurrently and posts results back.
        """
        channel = FanOutChannel(name="research")

        # Build tasks
        tasks = [
            SubAgentTask(
                agent_id=sq["id"],
                query=sq["query"],
                context={"rationale": sq.get("rationale", "")},
            )
            for sq in sub_queries
        ]

        # Dispatch tasks into the channel
        await channel.dispatch(tasks)
        self._log(f"  Dispatched {len(tasks)} tasks to channel")

        # Launch parallel sub-agent workers
        workers = [
            run_sub_agent_worker(
                agent_id=sq["id"],
                channel=channel,
                model=self.model,
                verbose=self.verbose,
            )
            for sq in sub_queries
        ]

        self._log(f"  Launching {len(workers)} parallel sub-agents...")

        # Run workers concurrently
        await asyncio.gather(*workers)

        # Collect results
        results = await channel.collect_all(timeout=self.sub_agent_timeout)
        channel.close()

        # Log per-agent summaries
        for r in results:
            status = "✓" if not r.error else f"✗ ({r.error})"
            self._log(
                f"  [{r.agent_id}] {status} — "
                f"{len(r.findings)} chars, {r.tool_calls_count} tools, "
                f"{r.duration_seconds:.1f}s"
            )

        return results

    # ----- Phase 3: Synthesize -----

    async def _synthesize(
        self,
        topic: str,
        sub_queries: list[dict[str, str]],
        results: list[SubAgentResult],
    ) -> str:
        """
        Synthesize all sub-agent findings into a unified research report.
        """
        # Build the synthesis prompt with all findings
        findings_text = []
        for r in results:
            section = f"## Sub-Agent: {r.agent_id}\n"
            section += f"**Query:** {r.query}\n"
            if r.error:
                section += f"**Error:** {r.error}\n"
            section += f"**Findings:**\n{r.findings}\n"
            if r.sources:
                section += "**Sources:**\n"
                for s in r.sources:
                    section += f"- [{s.get('title', 'Untitled')}]({s.get('url', '')})\n"
            findings_text.append(section)

        prompt = (
            f"# Research Topic\n{topic}\n\n"
            f"# Sub-Agent Findings\n\n"
            + "\n---\n\n".join(findings_text)
            + "\n\n---\n\n"
            "Synthesize these findings into a comprehensive, unified research report. "
            "Integrate the information — don't just list each sub-agent's output separately."
        )

        self._log(f"  Synthesizing {len(results)} sub-agent reports...")

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYNTHESIZE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = self._response_text(response)
        self._log(f"  Synthesis complete ({len(text)} chars)")
        return text

    # ----- Helpers -----

    @staticmethod
    def _response_text(response: Any) -> str:
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract JSON from text (code fences or bare)."""
        # Code-fenced JSON
        match = re.search(r"```(?:json)?\s*\n?(\[.*?\]|\{.*?\})\s*\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Bare JSON array or object
        for pattern in [r"\[.*\]", r"\{.*\}"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        return None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)
