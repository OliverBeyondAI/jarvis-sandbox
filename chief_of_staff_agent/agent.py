"""
Chief of Staff Agent — Multi-step autonomous research agent.

Conducts structured, multi-phase research using Claude Opus + Tavily:
  1. Planning  — Decomposes the brief into research questions and search strategy
  2. Research  — Executes parallel web searches and deep-dives on key sources
  3. Analysis  — Synthesizes findings into structured insights
  4. Reporting — Produces an executive-ready markdown briefing

Uses the Claude messages API with tool-use loop (local runner pattern).
Falls back gracefully if managed-agents API is unavailable.
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
from .models import ResearchBrief, Source
from .tools import ALL_TOOLS, execute_tool, set_output_dir


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

CHIEF_OF_STAFF_SYSTEM = textwrap.dedent("""\
    You are a **Chief of Staff Research Agent** — an elite executive research
    assistant that conducts thorough, multi-step investigations and delivers
    polished executive briefings.

    ## Your Role

    You work like a chief of staff at a top-tier organization: when given a
    research topic or question, you autonomously plan your research strategy,
    execute multiple rounds of web searches, read key sources in depth, and
    synthesize everything into a clear, actionable briefing.

    ## Tools Available

    1. **tavily_search** — Search the web for current information. Use targeted,
       specific queries. Run multiple searches to cover different angles.
    2. **fetch_url** — Read the full content of a web page. Use this to deep-dive
       into the most relevant search results.
    3. **save_report** — Save the final markdown report to disk. Call this exactly
       once when your research is complete.

    ## Research Protocol

    Follow this multi-step process:

    ### Phase 1: Planning
    - Analyze the research topic/brief
    - Identify 3-5 key research angles or sub-questions
    - Plan your search strategy (what to search for first, second, etc.)

    ### Phase 2: Broad Research
    - Execute 3-5 targeted web searches covering different angles
    - Scan results to identify the most important sources
    - Note key facts, trends, and data points

    ### Phase 3: Deep Dives
    - Use fetch_url to read 2-3 of the most important/authoritative sources
    - Extract detailed information, statistics, and expert opinions
    - Cross-reference findings across sources

    ### Phase 4: Gap Analysis
    - Review what you've learned so far
    - Identify any critical gaps in your research
    - Run 1-2 additional targeted searches to fill gaps

    ### Phase 5: Report Generation
    - Synthesize all findings into a structured executive briefing
    - Use the save_report tool to save the final markdown report

    ## Report Format

    Your final report MUST follow this structure:

    ```markdown
    # [Report Title]

    *Generated: [Date]*

    ---

    ## Executive Summary

    [2-3 paragraph overview for a busy executive. Lead with the most important
    finding. Include the "so what" — why this matters.]

    ---

    ## Key Takeaways

    1. [Most important insight]
    2. [Second most important]
    3. [Third]
    4. [Fourth — aim for 4-6 takeaways]

    ---

    ## [Topical Section 1]

    [Detailed analysis with specific data, quotes, and context]

    > **Finding Type:** [Key finding with supporting evidence]

    ## [Topical Section 2]

    [Continue with additional sections as needed — typically 3-5 sections]

    ---

    ## Risks & Considerations

    [What could go wrong? What should the reader watch for?]

    ---

    ## Recommended Actions

    [Concrete, actionable next steps based on the research]

    ---

    ## Sources

    - [Source Title](URL)
    - [Continue for all sources cited]

    ---

    *Methodology: Multi-step autonomous research using Tavily web search + Claude Opus analysis*
    ```

    ## Quality Standards

    - **Accuracy**: Only state facts you can attribute to sources. Flag uncertainty.
    - **Recency**: Prefer recent sources (2025-2026). Note when data is older.
    - **Balance**: Present multiple perspectives on contested topics.
    - **Actionability**: End with concrete recommendations, not just observations.
    - **Concision**: Busy executives read this. Every sentence must earn its place.
    - **Citations**: Attribute specific claims to their sources inline.

    ## Important Rules

    - Execute AT LEAST 4 web searches before writing the report
    - Deep-dive into AT LEAST 2 full articles using fetch_url
    - ALWAYS call save_report with the complete markdown as the final step
    - Do NOT ask the user questions — you have full autonomy to research
    - Think step-by-step but act decisively
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
    search_count: int = 0
    fetch_count: int = 0
    duration_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Chief of Staff Agent
# ---------------------------------------------------------------------------


class ChiefOfStaffAgent:
    """
    Multi-step research agent that produces executive briefings.

    Uses the Claude messages API with a tool-use agentic loop.
    The agent autonomously plans, researches, analyzes, and reports.
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

    async def run(self, brief: str | ResearchBrief) -> AgentResult:
        """
        Execute a full research cycle on the given brief.

        Args:
            brief: The research topic as a string, or a ResearchBrief model.

        Returns:
            AgentResult with the report text, file path, and metadata.
        """
        if isinstance(brief, ResearchBrief):
            brief_text = f"Topic: {brief.topic}\nPriority: {brief.priority.value}"
            if brief.background:
                brief_text += f"\nBackground: {brief.background}"
            if brief.questions:
                brief_text += "\n\nKey Questions:\n" + "\n".join(
                    f"- {q.question}" for q in brief.questions
                )
            brief = brief_text
        start_time = time.time()
        set_output_dir(self.config.output_dir)

        # Prepare tools for messages API (strip 'type' key)
        tools = [
            {k: v for k, v in tool.items() if k != "type"}
            for tool in ALL_TOOLS
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": brief},
        ]

        result = AgentResult()
        self._log(f"Starting research on: {brief[:100]}...")
        self._log(f"Model: {self.config.model}")
        self._log(f"Max turns: {self.config.max_agent_turns}")
        self._log("")

        for turn in range(1, self.config.max_agent_turns + 1):
            self._log(f"--- Turn {turn}/{self.config.max_agent_turns} ---")

            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=CHIEF_OF_STAFF_SYSTEM,
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
                    # Log the agent's thinking (truncated)
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

                    if tool_name == "tavily_search":
                        result.search_count += 1
                    elif tool_name == "fetch_url":
                        result.fetch_count += 1

                    result_str = await execute_tool(tool_name, tool_input)

                    # Collect sources from search results
                    if tool_name == "tavily_search":
                        try:
                            search_data = json.loads(result_str)
                            for r in search_data.get("results", []):
                                result.sources.append(Source(
                                    title=r.get("title", ""),
                                    url=r.get("url", ""),
                                    snippet=r.get("snippet", ""),
                                    relevance_score=min(max(
                                        r.get("relevance_score", 0), 0.0
                                    ), 1.0),
                                ))
                        except (json.JSONDecodeError, KeyError):
                            pass

                    # Track save_report path
                    if tool_name == "save_report":
                        try:
                            save_result = json.loads(result_str)
                            if save_result.get("saved"):
                                result.report_path = save_result.get("path", "")
                                self._log(f"  Report saved: {result.report_path}")
                        except json.JSONDecodeError:
                            pass

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
                # No tool calls and not end_turn — extract what we have
                result.text = self._extract_text(response)
                self._log(f"  No tool calls, stop_reason={response.stop_reason}")
                break

        result.duration_seconds = time.time() - start_time

        self._log("")
        self._log("=" * 50)
        self._log("RESEARCH COMPLETE")
        self._log(f"  Searches: {result.search_count}")
        self._log(f"  Pages fetched: {result.fetch_count}")
        self._log(f"  Total tool calls: {len(result.tool_calls)}")
        self._log(f"  Duration: {result.duration_seconds:.1f}s")
        if result.report_path:
            self._log(f"  Report: {result.report_path}")
        self._log("=" * 50)

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
            return f'"{fn}" ({size} chars)'
        return json.dumps(input_dict, default=str)[:60]

    def _log(self, msg: str) -> None:
        """Print progress to stderr."""
        print(f"[chief-of-staff] {msg}", file=sys.stderr)
