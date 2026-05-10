"""
Autonomous Research Agent — Core agent with multi-step research loop.

Conducts structured, multi-phase research using Claude + Tavily:
  1. Query Decomposition — Breaks complex questions into sub-queries
  2. Broad Search       — Parallel web searches across sub-queries
  3. Deep Dives         — Full-text reading of key sources
  4. Gap Analysis       — Identifies and fills research gaps
  5. Synthesis          — Cross-references findings, resolves contradictions
  6. Report Generation  — Produces structured report with actionable insights

Uses the Claude Messages API with a tool-use agentic loop.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic

from .config import Config
from .models import Source
from .tools import ALL_TOOLS, execute_tool, set_output_dir


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

RESEARCH_AGENT_SYSTEM = textwrap.dedent("""\
    You are an **Autonomous Research Agent** — an expert research system that
    conducts thorough, multi-step investigations on any topic and delivers
    structured reports with actionable insights.

    ## Your Mission

    Given a research query, you autonomously:
    1. Decompose it into targeted sub-questions
    2. Execute systematic web searches across multiple angles
    3. Deep-dive into authoritative sources
    4. Synthesize findings into coherent themes
    5. Generate actionable insights and recommendations
    6. Produce a polished, structured research report

    ## Tools Available

    1. **tavily_search** — Search the web for current information. Use specific,
       targeted queries. Run MULTIPLE searches (minimum 4) to cover different
       angles and sub-questions.

    2. **fetch_url** — Read the full content of a web page. Use this to
       deep-dive into the 2-3 most authoritative sources from search results.

    3. **analyze_findings** — Record and structure findings after each research
       phase. This helps you track progress and identify gaps. Use this after
       broad search and after deep dives.

    4. **save_report** — Save the final markdown report to disk. Call this
       exactly ONCE when your research and synthesis are complete.

    ## Research Protocol

    Follow this multi-phase process strictly:

    ### Phase 1: Query Decomposition
    - Analyze the research query carefully
    - Identify 3-6 key sub-questions or research angles
    - Plan your search strategy: what to search first, what depends on earlier results
    - State your plan explicitly before proceeding

    ### Phase 2: Broad Search (minimum 4 searches)
    - Execute targeted web searches for each sub-question
    - Use specific, well-crafted search queries (not the raw user query)
    - Vary your search terms to get diverse perspectives
    - After completing searches, use analyze_findings to record what you've learned

    ### Phase 3: Deep Dives (minimum 2 full-page reads)
    - Identify the 2-3 most authoritative or information-rich sources
    - Use fetch_url to read their full content
    - Extract detailed data, statistics, expert opinions, and nuance
    - Cross-reference information across sources
    - Use analyze_findings to record deep-dive insights

    ### Phase 4: Gap Analysis & Fill
    - Review all findings so far
    - Identify critical gaps: missing perspectives, unanswered sub-questions,
      contradictions that need resolution
    - Run 1-3 additional targeted searches to fill gaps
    - Resolve contradictions by finding authoritative sources

    ### Phase 5: Synthesis
    - Identify cross-cutting themes across all findings
    - Note where sources agree and disagree
    - Assess confidence levels for each major conclusion
    - Derive actionable insights from the evidence

    ### Phase 6: Report Generation
    - Write the complete report following the format below
    - Save it using save_report

    ## Report Format

    Your final report MUST follow this structure:

    ```markdown
    # [Report Title]

    *Generated: [Date]*
    *Query: [Original research query]*

    ---

    ## Executive Summary

    [2-3 paragraph overview for a busy reader. Lead with the most important
    finding. Include the "so what" — why this matters and what to do about it.]

    ---

    ## Key Takeaways

    1. [Most important insight — one sentence]
    2. [Second most important]
    3. [Third]
    4. [Aim for 4-6 takeaways]

    ---

    ## [Topical Section 1 Title]

    [Detailed analysis with specific data, quotes, and context]

    > **Finding Type** (Confidence): [Key finding with supporting evidence]

    ## [Topical Section 2 Title]

    [Continue with 3-5 topical sections covering the research angles]

    ---

    ## Cross-Cutting Themes

    ### [Theme 1]
    [Synthesis across findings]

    **Implications:**
    - [Implication 1]
    - [Implication 2]

    ---

    ## Actionable Insights & Recommendations

    ### 1. [Insight Title] [PRIORITY]

    [Description of the recommendation]

    **Rationale:** [Why this matters based on the evidence]

    **Next Steps:**
    - [Concrete step 1]
    - [Concrete step 2]

    **Risk if ignored:** [What happens if this is not acted on]

    ---

    ## Sources

    - [Source Title](URL)
    - [Continue for all sources]

    ---

    *Methodology: Multi-step autonomous research using Tavily web search + Claude analysis*
    *Searches: N | Pages read: N | Duration: Ns*
    ```

    ## Quality Standards

    - **Thoroughness**: Execute at least 4 searches and 2 deep dives before writing
    - **Accuracy**: Only state facts you can attribute to sources. Flag uncertainty.
    - **Recency**: Prefer recent sources (2025-2026). Note when data is older.
    - **Balance**: Present multiple perspectives on contested topics.
    - **Actionability**: End with concrete, prioritized recommendations.
    - **Evidence chains**: Connect insights to specific findings and sources.
    - **Contradiction handling**: When sources disagree, present both sides and
      assess which is more credible and why.

    ## Rules

    - Execute AT LEAST 4 web searches across different angles
    - Deep-dive into AT LEAST 2 full articles using fetch_url
    - Use analyze_findings at least twice (after broad search and after deep dives)
    - ALWAYS call save_report with the complete markdown as the final step
    - Do NOT ask the user questions — you have full autonomy to research
    - Think step-by-step but act decisively
    - Track your progress: after each phase, briefly summarize what you've
      learned and what gaps remain
""")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Structured output from an agent run."""

    text: str = ""
    report_path: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    search_count: int = 0
    fetch_count: int = 0
    analyze_count: int = 0
    duration_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Autonomous Research Agent
# ---------------------------------------------------------------------------


class AutonomousResearchAgent:
    """
    Multi-step research agent with query decomposition and finding synthesis.

    Uses the Claude Messages API with a tool-use agentic loop.
    The agent autonomously decomposes queries, searches, synthesizes, and reports.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Lazy-load async client."""
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self.config.anthropic_api_key:
                kwargs["api_key"] = self.config.anthropic_api_key
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def run(self, query: str) -> AgentResult:
        """
        Execute a full autonomous research cycle on the given query.

        Args:
            query: The research question or topic to investigate.

        Returns:
            AgentResult with the report text, file path, and metadata.
        """
        start_time = time.time()
        set_output_dir(self.config.output_dir)

        # Prepare tools for Messages API (strip 'type' key)
        tools = [
            {k: v for k, v in tool.items() if k != "type"}
            for tool in ALL_TOOLS
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": query},
        ]

        result = AgentResult()
        self._log(f"Starting research: {query[:120]}...")
        self._log(f"Model: {self.config.model}")
        self._log(f"Max turns: {self.config.max_agent_turns}")
        self._log("")

        for turn in range(1, self.config.max_agent_turns + 1):
            self._log(f"--- Turn {turn}/{self.config.max_agent_turns} ---")

            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=RESEARCH_AGENT_SYSTEM,
                tools=tools,
                messages=messages,
            )

            # Check for end of conversation
            if response.stop_reason == "end_turn":
                result.text = self._extract_text(response)
                self._log("Agent finished (end_turn).")
                break

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if hasattr(block, "text") and block.text:
                    thinking = block.text[:200]
                    if len(block.text) > 200:
                        thinking += "..."
                    self._log(f"  [thinking] {thinking}")

                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    input_summary = self._summarize_input(tool_name, tool_input)
                    self._log(f"  -> {tool_name}({input_summary})")

                    result.tool_calls.append(
                        {"name": tool_name, "input": tool_input}
                    )

                    # Track tool usage
                    if tool_name == "tavily_search":
                        result.search_count += 1
                    elif tool_name == "fetch_url":
                        result.fetch_count += 1
                    elif tool_name == "analyze_findings":
                        result.analyze_count += 1

                    result_str = await execute_tool(tool_name, tool_input)

                    # Collect sources from search results
                    if tool_name == "tavily_search":
                        self._collect_sources(result, result_str)

                    # Collect findings from analysis
                    if tool_name == "analyze_findings":
                        self._collect_findings(result, result_str)

                    # Track save_report path
                    if tool_name == "save_report":
                        self._track_report(result, result_str)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                result.text = self._extract_text(response)
                self._log(
                    f"  No tool calls, stop_reason={response.stop_reason}"
                )
                break

        result.duration_seconds = time.time() - start_time

        self._log("")
        self._log("=" * 60)
        self._log("RESEARCH COMPLETE")
        self._log(f"  Searches:        {result.search_count}")
        self._log(f"  Pages fetched:   {result.fetch_count}")
        self._log(f"  Analysis phases: {result.analyze_count}")
        self._log(f"  Total tool calls: {len(result.tool_calls)}")
        self._log(f"  Findings logged: {len(result.findings)}")
        self._log(f"  Sources found:   {len(result.sources)}")
        self._log(f"  Duration:        {result.duration_seconds:.1f}s")
        if result.report_path:
            self._log(f"  Report:          {result.report_path}")
        self._log("=" * 60)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract all text blocks from a Claude response."""
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _collect_sources(result: AgentResult, result_str: str) -> None:
        """Extract sources from a tavily_search result."""
        try:
            search_data = json.loads(result_str)
            for r in search_data.get("results", []):
                result.sources.append(
                    Source(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("snippet", ""),
                        relevance_score=min(
                            max(r.get("relevance_score", 0), 0.0), 1.0
                        ),
                    )
                )
        except (json.JSONDecodeError, KeyError):
            pass

    @staticmethod
    def _collect_findings(result: AgentResult, result_str: str) -> None:
        """Extract findings from an analyze_findings result."""
        try:
            data = json.loads(result_str)
            for f in data.get("findings", []):
                result.findings.append(f)
        except (json.JSONDecodeError, KeyError):
            pass

    @staticmethod
    def _track_report(result: AgentResult, result_str: str) -> None:
        """Track save_report output path."""
        try:
            save_result = json.loads(result_str)
            if save_result.get("saved"):
                result.report_path = save_result.get("path", "")
        except json.JSONDecodeError:
            pass

    @staticmethod
    def _summarize_input(tool_name: str, input_dict: dict[str, Any]) -> str:
        """Create a short log summary of tool input."""
        if tool_name == "tavily_search":
            q = input_dict.get("query", "")
            return f'"{q[:60]}..."' if len(q) > 60 else f'"{q}"'
        if tool_name == "fetch_url":
            url = input_dict.get("url", "")
            return f'"{url[:60]}..."' if len(url) > 60 else f'"{url}"'
        if tool_name == "save_report":
            fn = input_dict.get("filename", "")
            size = len(input_dict.get("content", ""))
            return f'"{fn}" ({size:,} chars)'
        if tool_name == "analyze_findings":
            phase = input_dict.get("phase", "?")
            count = len(input_dict.get("findings", []))
            return f'phase="{phase}", {count} findings'
        return json.dumps(input_dict, default=str)[:60]

    def _log(self, msg: str) -> None:
        """Print progress to stderr."""
        print(f"[research-agent] {msg}", file=sys.stderr)
