#!/usr/bin/env python3
"""
Research Agent — Multi-step web research with Claude Opus 4.6.

A specialized agent that conducts autonomous research using a structured
workflow: search → extract → follow-up searches → synthesize. Uses Tavily
for web search, httpx for URL fetching, and an in-memory scratchpad to
accumulate findings across tool calls.

Built on the Claude Agent SDK (Anthropic Python SDK) with both managed-agents
and local fallback runners.

Usage:
    python -m agents.research_agent "What are the latest advances in quantum computing?"
    python -m agents.research_agent --topic "AI regulation" --depth deep
    python -m agents.research_agent --topic "mRNA vaccines" --output ./data/mrna_research.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anthropic
import httpx


# ---------------------------------------------------------------------------
# Summarization & Key-Takeaway Extraction
# ---------------------------------------------------------------------------

@dataclass
class KeyTakeaway:
    """A single key takeaway extracted from research."""
    point: str
    supporting_evidence: str = ""
    confidence: str = "medium"  # low, medium, high


@dataclass
class ResearchSummary:
    """Structured summary with key takeaways extracted from research results."""
    executive_summary: str = ""
    key_takeaways: list[KeyTakeaway] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    source_count: int = 0
    word_count: int = 0

    def format_text(self) -> str:
        """Format summary as readable text."""
        lines = []
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(self.executive_summary)
        lines.append("")

        if self.key_takeaways:
            lines.append("KEY TAKEAWAYS")
            lines.append("-" * 40)
            for i, t in enumerate(self.key_takeaways, 1):
                conf_marker = {"high": "+", "medium": "~", "low": "?"}.get(t.confidence, "~")
                lines.append(f"  [{conf_marker}] {i}. {t.point}")
                if t.supporting_evidence:
                    lines.append(f"      Evidence: {t.supporting_evidence}")
            lines.append("")

        if self.themes:
            lines.append("THEMES")
            lines.append("-" * 40)
            for theme in self.themes:
                lines.append(f"  - {theme}")
            lines.append("")

        if self.open_questions:
            lines.append("OPEN QUESTIONS")
            lines.append("-" * 40)
            for q in self.open_questions:
                lines.append(f"  ? {q}")
            lines.append("")

        lines.append(f"Sources: {self.source_count} | Words: {self.word_count}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "executive_summary": self.executive_summary,
            "key_takeaways": [
                {"point": t.point, "supporting_evidence": t.supporting_evidence, "confidence": t.confidence}
                for t in self.key_takeaways
            ],
            "themes": self.themes,
            "open_questions": self.open_questions,
            "source_count": self.source_count,
            "word_count": self.word_count,
        }


def extract_summary(result: "ResearchResult") -> ResearchSummary:
    """Extract a structured summary with key takeaways from a ResearchResult.

    Uses heuristic parsing of the synthesis text to identify:
    - Executive summary (first paragraph or explicit section)
    - Key takeaways (bulleted findings, key points sections)
    - Themes (section headings as topic themes)
    - Open questions (explicit questions or "further research" items)
    """
    text = result.synthesis.strip()
    if not text:
        return ResearchSummary()

    summary = ResearchSummary(
        source_count=result.sources_consulted,
        word_count=len(text.split()),
    )

    lines = text.split("\n")
    sections = _split_into_sections(lines)

    # Extract executive summary
    summary.executive_summary = _extract_executive_summary(sections, text)

    # Extract key takeaways
    summary.key_takeaways = _extract_key_takeaways(sections, text)

    # Extract themes from section headings
    summary.themes = _extract_themes(sections)

    # Extract open questions
    summary.open_questions = _extract_open_questions(sections, text)

    return summary


def _split_into_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split text lines into (heading, body_lines) sections."""
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Detect markdown headings
        if stripped.startswith("#"):
            if current_heading or current_body:
                sections.append((current_heading, current_body))
            current_heading = stripped.lstrip("#").strip()
            current_body = []
        else:
            current_body.append(line)

    if current_heading or current_body:
        sections.append((current_heading, current_body))

    return sections


def _extract_executive_summary(sections: list[tuple[str, list[str]]], full_text: str) -> str:
    """Extract executive summary from sections or first paragraph."""
    # Look for explicit executive/summary section
    for heading, body in sections:
        heading_lower = heading.lower()
        if any(kw in heading_lower for kw in ["executive summary", "summary", "overview", "tldr", "tl;dr"]):
            text = "\n".join(line for line in body if line.strip()).strip()
            if text:
                return text

    # Fall back to first non-empty paragraph
    paragraphs = full_text.split("\n\n")
    for para in paragraphs:
        cleaned = para.strip().lstrip("#").strip()
        if cleaned and len(cleaned) > 20 and not cleaned.startswith("-") and not cleaned.startswith("*"):
            # Limit to ~3 sentences
            sentences = re.split(r'(?<=[.!?])\s+', cleaned)
            return " ".join(sentences[:3])

    return full_text[:300].strip()


def _extract_key_takeaways(sections: list[tuple[str, list[str]]], full_text: str) -> list[KeyTakeaway]:
    """Extract key takeaways from bullet points in findings/takeaway sections."""
    takeaways: list[KeyTakeaway] = []

    # Priority sections for takeaways
    priority_keywords = ["key findings", "key takeaways", "takeaways", "findings", "highlights",
                         "key points", "main findings", "conclusions", "key insights"]

    target_bodies: list[list[str]] = []
    for heading, body in sections:
        if any(kw in heading.lower() for kw in priority_keywords):
            target_bodies.append(body)

    # If no specific section found, scan all bullet points
    if not target_bodies:
        all_body = [line for _, body in sections for line in body]
        target_bodies.append(all_body)

    for body in target_bodies:
        for line in body:
            stripped = line.strip()
            # Match bullet points: -, *, •, or numbered lists
            bullet_match = re.match(r'^(?:[-*•]|\d+[.)]\s)', stripped)
            if bullet_match:
                point = re.sub(r'^(?:[-*•]|\d+[.)]\s*)\s*', '', stripped).strip()
                if len(point) > 10:  # Skip trivially short bullets
                    # Strip bold markdown
                    point = re.sub(r'\*\*(.+?)\*\*', r'\1', point)
                    confidence = _infer_confidence(point)
                    takeaways.append(KeyTakeaway(point=point, confidence=confidence))

    # Deduplicate and limit
    seen: set[str] = set()
    unique: list[KeyTakeaway] = []
    for t in takeaways:
        normalized = t.point.lower()[:60]
        if normalized not in seen:
            seen.add(normalized)
            unique.append(t)

    return unique[:10]  # Cap at 10


def _infer_confidence(text: str) -> str:
    """Infer confidence level from language cues."""
    low_cues = ["may", "might", "could", "uncertain", "unclear", "preliminary", "early", "limited"]
    high_cues = ["clearly", "significant", "demonstrated", "proven", "established", "confirmed", "substantial"]

    text_lower = text.lower()
    low_score = sum(1 for cue in low_cues if cue in text_lower)
    high_score = sum(1 for cue in high_cues if cue in text_lower)

    if high_score > low_score:
        return "high"
    elif low_score > high_score:
        return "low"
    return "medium"


def _extract_themes(sections: list[tuple[str, list[str]]]) -> list[str]:
    """Extract thematic topics from section headings."""
    skip_headings = {"", "executive summary", "summary", "overview", "introduction",
                     "conclusion", "conclusions", "sources", "references", "further research",
                     "open questions", "key findings", "key takeaways", "takeaways",
                     "findings", "methodology", "appendix"}
    themes = []
    for heading, _ in sections:
        heading_clean = heading.strip()
        if heading_clean and heading_clean.lower() not in skip_headings:
            # Strip numbering
            theme = re.sub(r'^\d+[.)]\s*', '', heading_clean)
            if theme and len(theme) > 2:
                themes.append(theme)
    return themes[:8]


def _extract_open_questions(sections: list[tuple[str, list[str]]], full_text: str) -> list[str]:
    """Extract open questions or areas for further research."""
    questions: list[str] = []

    # Look for explicit sections
    question_keywords = ["open questions", "further research", "future research",
                         "areas for further", "limitations", "gaps"]
    for heading, body in sections:
        if any(kw in heading.lower() for kw in question_keywords):
            for line in body:
                stripped = line.strip()
                bullet_match = re.match(r'^(?:[-*•?]|\d+[.)]\s)', stripped)
                if bullet_match:
                    q = re.sub(r'^(?:[-*•?]|\d+[.)]\s*)\s*', '', stripped).strip()
                    if q and len(q) > 10:
                        questions.append(q)

    # Also find inline questions (single-line sentences ending with ?)
    for line in full_text.split("\n"):
        stripped_line = line.strip()
        # Skip headings, bullets, and empty lines
        if not stripped_line or stripped_line.startswith(("#", "-", "*", "•")):
            continue
        for sq in re.findall(r'([A-Z][^.!?\n]{15,}\?)', stripped_line):
            sq = sq.strip()
            # Skip if already captured
            if not any(sq in existing or existing in sq for existing in questions):
                questions.append(sq)

    return questions[:5]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-6-20250501"
MAX_TOKENS = 8192
MAX_TURNS = 30

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autonomous research agent powered by Claude Opus 4.6. Your role
    is to conduct thorough, multi-step research on any topic and produce a
    well-structured synthesis.

    ## Tools Available

    1. **tavily_search** — Search the web for current information. Use multiple
       queries to cover different angles of the topic.
    2. **fetch_url** — Retrieve full content from a specific URL for deeper reading.
    3. **save_notes** — Save research notes to your scratchpad. Use this to
       accumulate key findings, quotes, and source info as you research.
       Notes persist across all your tool calls and are available for synthesis.
    4. **save_report** — Save the final research report to the filesystem
       (data/ directory). Use this only for the final output.

    ## Research Protocol

    Follow this structured multi-step approach:

    ### Phase 1: Discovery (search)
    Issue 3-5 diverse search queries to map the landscape. Vary query phrasing
    to capture different perspectives and source types. After each search,
    use save_notes to record the most important findings.

    ### Phase 2: Deep Extraction (extract)
    Select the 2-3 most promising sources from your search results and fetch
    their full content with fetch_url. Read carefully and save_notes with
    key quotes, data points, and expert opinions.

    ### Phase 3: Follow-up Searches (follow-up)
    Based on what you learned in Phase 2, identify gaps or new questions.
    Run 2-3 follow-up searches targeting these gaps. Save notes on anything new.

    ### Phase 4: Synthesis (synthesize)
    Review all your accumulated notes and produce a comprehensive synthesis.
    Save the final report using save_report with a clear filename. The report
    should include:
    - Executive summary (2-3 sentences)
    - Key findings (bulleted)
    - Detailed analysis (organized by theme)
    - Sources cited with URLs
    - Open questions / areas for further research

    ## Guidelines

    - Always cite sources with URLs.
    - Prefer recent sources (last 12 months when relevant).
    - Acknowledge uncertainty and conflicting information.
    - Save notes frequently — your scratchpad is your working memory.
    - Be thorough but concise — aim for actionable insights over exhaustive detail.
""")

BRIEF_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a research agent with web search, URL fetching, and note-taking tools.
    Conduct focused research on the given topic: search for key information, read
    1-2 important sources, and produce a concise summary. Save notes as you go
    using save_notes, then save your final report with save_report.
    Always cite sources.
""")


# ---------------------------------------------------------------------------
# Scratchpad — Accumulator for multi-step research
# ---------------------------------------------------------------------------

@dataclass
class Scratchpad:
    """In-memory scratchpad that accumulates research notes across tool calls.

    The scratchpad preserves intermediate findings so the agent can reference
    them during synthesis without re-fetching sources.
    """

    entries: list[dict[str, str]] = field(default_factory=list)

    def add(self, content: str, label: str = "notes") -> str:
        """Add a note to the scratchpad and return confirmation."""
        self.entries.append({"label": label, "content": content})
        return f"Saved notes under '{label}' ({len(self.entries)} total entries)."

    def dump(self) -> str:
        """Dump all scratchpad entries as formatted text."""
        if not self.entries:
            return "(no notes saved yet)"
        parts = []
        for i, entry in enumerate(self.entries, 1):
            parts.append(f"[{i}] {entry['label']}:\n{entry['content']}")
        return "\n\n".join(parts)

    def count(self) -> int:
        """Return number of entries."""
        return len(self.entries)


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic custom tool format for managed agents)
# ---------------------------------------------------------------------------

TAVILY_SEARCH_TOOL: dict[str, Any] = {
    "name": "tavily_search",
    "type": "custom",
    "description": (
        "Search the web using Tavily for current information on any topic. "
        "Returns search results with titles, URLs, and content snippets. "
        "Use varied queries to get comprehensive coverage."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (1-10, default 5).",
                "default": 5,
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": (
                    "Search depth — 'basic' is faster, 'advanced' is more thorough. "
                    "Default: 'basic'."
                ),
                "default": "basic",
            },
        },
        "required": ["query"],
    },
}

FETCH_URL_TOOL: dict[str, Any] = {
    "name": "fetch_url",
    "type": "custom",
    "description": (
        "Fetch the text content of a web page at the given URL. "
        "Returns extracted text with HTML stripped. "
        "Use this to read full articles or documents found via search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch content from.",
            },
        },
        "required": ["url"],
    },
}

SAVE_NOTES_TOOL: dict[str, Any] = {
    "name": "save_notes",
    "type": "custom",
    "description": (
        "Save research notes to your scratchpad. Notes accumulate across calls "
        "and are available for reference during synthesis. Use this frequently "
        "to record key findings, quotes, and source information as you research."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The research notes to save.",
            },
            "label": {
                "type": "string",
                "description": (
                    "A short label for this set of notes (e.g., 'discovery_search_1', "
                    "'deep_dive_arxiv', 'follow_up_gaps')."
                ),
                "default": "notes",
            },
        },
        "required": ["content"],
    },
}

SAVE_REPORT_TOOL: dict[str, Any] = {
    "name": "save_report",
    "type": "custom",
    "description": (
        "Save the final research report to the local filesystem (data/ directory). "
        "Use this only for the completed synthesis, not for intermediate notes. "
        "Files are saved as markdown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": (
                    "Name of the file to save (e.g., 'research_quantum_2026-05-24.md'). "
                    "Will be saved inside the data/ directory."
                ),
            },
            "content": {
                "type": "string",
                "description": "The markdown content of the final report.",
            },
        },
        "required": ["filename", "content"],
    },
}

ALL_TOOLS: list[dict[str, Any]] = [
    TAVILY_SEARCH_TOOL,
    FETCH_URL_TOOL,
    SAVE_NOTES_TOOL,
    SAVE_REPORT_TOOL,
]


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

async def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict[str, Any]:
    """Search the web using the Tavily API."""
    def _sync_search() -> dict[str, Any]:
        try:
            from tavily import TavilyClient
            client = TavilyClient()  # Uses TAVILY_API_KEY env var
            results = client.search(
                query=query,
                max_results=max(1, min(max_results, 10)),
                search_depth=search_depth,
            )
            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0),
                    }
                    for r in results.get("results", [])
                ],
            }
        except ImportError:
            return {"query": query, "error": "tavily-python not installed. Run: pip install tavily-python"}
        except Exception as e:
            return {"query": query, "error": f"Search failed: {e}"}

    return await asyncio.to_thread(_sync_search)


async def fetch_url(url: str) -> dict[str, Any]:
    """Fetch and extract text content from a URL."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "ResearchAgent/1.0",
                "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            text = response.text
            if "text/html" in content_type:
                text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
                text = re.sub(r"\s+", " ", text).strip()

            max_chars = 50_000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Content truncated at 50,000 characters]"

            return {
                "url": url,
                "status": response.status_code,
                "content_type": content_type.split(";")[0].strip(),
                "content": text,
                "length": len(text),
            }
    except httpx.HTTPStatusError as e:
        return {"url": url, "error": f"HTTP {e.response.status_code}: {e}"}
    except httpx.TimeoutException:
        return {"url": url, "error": "Request timed out after 30 seconds"}
    except Exception as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}


def save_notes(scratchpad: Scratchpad, content: str, label: str = "notes") -> dict[str, Any]:
    """Save research notes to the scratchpad."""
    msg = scratchpad.add(content, label)
    return {"saved": True, "message": msg, "total_entries": scratchpad.count()}


async def save_report(filename: str, content: str) -> dict[str, Any]:
    """Save the final research report to the data/ directory."""
    try:
        safe_name = Path(filename).name
        if not safe_name:
            return {"error": "Invalid filename"}

        filepath = DATA_DIR / safe_name
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        def _write():
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

        await asyncio.to_thread(_write)

        return {
            "saved": True,
            "path": str(filepath),
            "filename": safe_name,
            "size_bytes": filepath.stat().st_size,
        }
    except Exception as e:
        return {"error": f"Failed to save: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(
    name: str,
    input_dict: dict[str, Any],
    scratchpad: Scratchpad,
) -> str:
    """Execute a tool by name and return JSON string result."""
    try:
        if name == "tavily_search":
            result = await tavily_search(
                query=input_dict["query"],
                max_results=input_dict.get("max_results", 5),
                search_depth=input_dict.get("search_depth", "basic"),
            )
        elif name == "fetch_url":
            result = await fetch_url(input_dict["url"])
        elif name == "save_notes":
            result = save_notes(
                scratchpad,
                content=input_dict["content"],
                label=input_dict.get("label", "notes"),
            )
        elif name == "save_report":
            result = await save_report(
                filename=input_dict["filename"],
                content=input_dict["content"],
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ResearchResult:
    """Structured output from a research agent run."""
    synthesis: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    files_saved: list[str] = field(default_factory=list)
    sources_consulted: int = 0
    scratchpad_entries: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def summary(self) -> str:
        """One-line summary of the research run."""
        return (
            f"Research complete: {self.sources_consulted} sources consulted, "
            f"{self.scratchpad_entries} notes accumulated, "
            f"{len(self.files_saved)} file(s) saved, "
            f"{len(self.tool_calls)} tool calls made."
        )

    def summarize(self) -> "ResearchSummary":
        """Extract structured summary with key takeaways from the synthesis."""
        return extract_summary(self)


# ---------------------------------------------------------------------------
# Managed Agent Runner (Claude Agent SDK)
# ---------------------------------------------------------------------------

class ManagedResearchRunner:
    """Runs the research agent via Claude Agent SDK managed-agents API."""

    def __init__(
        self,
        model: str = MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        verbose: bool = True,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt
        self.verbose = verbose

    async def run(self, prompt: str) -> ResearchResult:
        """Execute a full research session via managed-agents."""
        agent = None
        environment = None
        result = ResearchResult()
        scratchpad = Scratchpad()

        try:
            self._log("Creating managed research agent...")
            agent = self.client.beta.agents.create(
                model=self.model,
                name="research-agent",
                description="Autonomous research agent with multi-step web research workflow.",
                system=self.system_prompt,
                tools=ALL_TOOLS,
            )
            self._log(f"  Agent: {agent.id}")

            environment = self.client.beta.environments.create(name="research-env")
            self._log(f"  Environment: {environment.id}")

            session = self.client.beta.sessions.create(
                agent=agent.id,
                environment_id=environment.id,
            )
            self._log(f"  Session: {session.id}")

            self._log("Sending research prompt...")
            self.client.beta.sessions.events.send(
                session_id=session.id,
                events=[{"type": "user.message", "content": [{"type": "text", "text": prompt}]}],
            )

            final_text = ""
            for round_num in range(1, MAX_TURNS + 1):
                self._log(f"  Round {round_num}...")
                pending_tool_results = []

                events = self.client.beta.sessions.events.list(session_id=session.id, order="asc")
                for event in events.data:
                    if event.type == "agent.message":
                        for block in event.content:
                            if hasattr(block, "text"):
                                final_text = block.text

                    elif event.type == "agent.custom_tool_use":
                        tool_name = event.name
                        tool_input = dict(event.input)
                        self._log(f"    [{tool_name}] {_summarize(tool_input)}")

                        result.tool_calls.append({"name": tool_name, "input": tool_input})
                        tool_result = await execute_tool(tool_name, tool_input, scratchpad)

                        # Track files saved and sources
                        _track_result(result, tool_name, tool_result)

                        pending_tool_results.append({
                            "type": "user.custom_tool_result",
                            "custom_tool_use_id": event.id,
                            "content": [{"type": "text", "text": tool_result}],
                        })

                    elif event.type == "session.status_idle":
                        stop = event.stop_reason
                        if hasattr(stop, "type") and stop.type == "end_turn":
                            self._log("  Research complete.")
                            result.synthesis = final_text
                            result.scratchpad_entries = scratchpad.count()
                            return result

                if pending_tool_results:
                    self._log(f"    Sending {len(pending_tool_results)} result(s)...")
                    self.client.beta.sessions.events.send(
                        session_id=session.id,
                        events=pending_tool_results,
                    )
                else:
                    if final_text:
                        result.synthesis = final_text
                        result.scratchpad_entries = scratchpad.count()
                        return result
                    break

            result.synthesis = final_text or "Research agent reached maximum rounds."
            result.scratchpad_entries = scratchpad.count()
            return result

        finally:
            if agent:
                try:
                    self.client.beta.agents.archive(agent_id=agent.id)
                except Exception:
                    pass
            if environment:
                try:
                    self.client.beta.environments.delete(environment_id=environment.id)
                except Exception:
                    pass

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[research] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Local Agent Runner (Fallback)
# ---------------------------------------------------------------------------

class LocalResearchRunner:
    """Runs the research agent locally using AsyncAnthropic with tool loop.

    Drives the multi-step research workflow client-side: the system prompt
    guides the model through search → extract → follow-up → synthesize,
    while the scratchpad accumulates findings across turns.
    """

    def __init__(
        self,
        model: str = MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        verbose: bool = True,
    ):
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.system_prompt = system_prompt
        self.verbose = verbose

    def _messages_api_tools(self) -> list[dict[str, Any]]:
        """Convert tool schemas for the messages API (strip 'type' key)."""
        return [{k: v for k, v in tool.items() if k != "type"} for tool in ALL_TOOLS]

    async def run(self, prompt: str) -> ResearchResult:
        """Run the local agentic loop with scratchpad accumulation."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        result = ResearchResult()
        scratchpad = Scratchpad()

        self._log("Starting local research loop...")

        for turn in range(1, MAX_TURNS + 1):
            self._log(f"  Turn {turn}/{MAX_TURNS}")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=self._messages_api_tools(),
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                result.synthesis = _extract_text(response)
                result.scratchpad_entries = scratchpad.count()
                self._log(f"  Research complete. ({scratchpad.count()} scratchpad entries)")
                return result

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type == "tool_use":
                    self._log(f"    [{block.name}] {_summarize(block.input)}")
                    result.tool_calls.append({"name": block.name, "input": block.input})

                    tool_result = await execute_tool(block.name, block.input, scratchpad)
                    _track_result(result, block.name, tool_result)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                result.synthesis = _extract_text(response)
                result.scratchpad_entries = scratchpad.count()
                return result

        result.synthesis = "Research agent reached maximum turns."
        result.scratchpad_entries = scratchpad.count()
        return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[research] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API — ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Autonomous research agent with multi-step web research workflow.

    Uses Claude Opus 4.6 via the Claude Agent SDK (managed-agents API) with
    automatic fallback to a local async runner. Conducts research through
    four phases — discovery, extraction, follow-up, synthesis — accumulating
    findings in a scratchpad across tool calls.

    Example:
        agent = ResearchAgent()
        result = await agent.research("What are the latest advances in quantum computing?")
        print(result.synthesis)
        print(result.summary)
    """

    def __init__(
        self,
        model: str = MODEL,
        depth: str = "deep",
        verbose: bool = True,
        use_managed: bool = True,
        output_dir: Optional[Path] = None,
    ):
        self.model = model
        self.depth = depth
        self.verbose = verbose
        self.use_managed = use_managed

        if output_dir:
            global DATA_DIR
            DATA_DIR = Path(output_dir)

        self.system_prompt = SYSTEM_PROMPT if depth == "deep" else BRIEF_SYSTEM_PROMPT

    async def research(self, topic: str) -> ResearchResult:
        """
        Conduct autonomous multi-step research on the given topic.

        The agent follows a structured workflow:
        1. Discovery — broad web searches to map the landscape
        2. Extraction — deep reading of promising sources
        3. Follow-up — targeted searches to fill gaps
        4. Synthesis — structured report from accumulated notes

        Args:
            topic: The research question or topic to investigate.

        Returns:
            ResearchResult with synthesis text, files saved, and metadata.
        """
        prompt = self._build_prompt(topic)

        if self.use_managed:
            try:
                runner = ManagedResearchRunner(
                    model=self.model,
                    system_prompt=self.system_prompt,
                    verbose=self.verbose,
                )
                return await runner.run(prompt)
            except (anthropic.APIError, anthropic.APIConnectionError) as exc:
                if self.verbose:
                    print(
                        f"[research] Managed API unavailable ({type(exc).__name__}), "
                        "falling back to local runner...",
                        file=sys.stderr,
                    )

        runner = LocalResearchRunner(
            model=self.model,
            system_prompt=self.system_prompt,
            verbose=self.verbose,
        )
        return await runner.run(prompt)

    def _build_prompt(self, topic: str) -> str:
        """Build the research prompt with topic and date context."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.depth == "deep":
            return textwrap.dedent(f"""\
                Research Topic: {topic}

                Today's date: {today}

                Please conduct thorough research on this topic following your research
                protocol. Execute all four phases — discovery, deep extraction, follow-up
                searches, and synthesis. Use save_notes frequently to accumulate findings,
                then save the final report with save_report.
            """)
        else:
            return textwrap.dedent(f"""\
                Research Topic: {topic}
                Date: {today}

                Conduct focused research on this topic. Search for the most important
                recent information, save key notes, then produce a concise final report
                with save_report.
            """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(response: Any) -> str:
    """Extract all text content blocks from a Claude response."""
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


def _summarize(input_dict: dict[str, Any]) -> str:
    """Create a short summary of tool input for logging."""
    if "query" in input_dict:
        q = input_dict["query"]
        return f'query="{q[:50]}..."' if len(q) > 50 else f'query="{q}"'
    if "url" in input_dict:
        u = input_dict["url"]
        return f'url="{u[:60]}..."' if len(u) > 60 else f'url="{u}"'
    if "filename" in input_dict:
        return f'file="{input_dict["filename"]}"'
    if "label" in input_dict:
        return f'notes: {input_dict["label"]}'
    return json.dumps(input_dict, default=str)[:60]


def _track_result(result: ResearchResult, tool_name: str, tool_result_json: str) -> None:
    """Update ResearchResult tracking fields based on tool execution."""
    try:
        parsed = json.loads(tool_result_json)
    except (json.JSONDecodeError, TypeError):
        return

    if tool_name == "save_report" and parsed.get("saved"):
        result.files_saved.append(parsed.get("path", "unknown"))
    elif tool_name in ("tavily_search", "fetch_url") and "error" not in parsed:
        result.sources_consulted += 1


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """CLI interface for the research agent."""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent — Claude Opus 4.6 + Tavily Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m agents.research_agent "quantum computing breakthroughs 2026"
              python -m agents.research_agent --topic "AI regulation EU" --depth brief
              python -m agents.research_agent --topic "mRNA therapeutics" --local
        """),
    )
    parser.add_argument("query", nargs="?", help="Research topic (positional)")
    parser.add_argument("--topic", "-t", help="Research topic (named)")
    parser.add_argument(
        "--depth", "-d", choices=["deep", "brief"], default="deep",
        help="Research depth: 'deep' (multi-phase) or 'brief' (focused). Default: deep",
    )
    parser.add_argument("--local", "-l", action="store_true", help="Force local runner (skip managed API)")
    parser.add_argument("--model", "-m", default=MODEL, help=f"Model to use. Default: {MODEL}")
    parser.add_argument("--output", "-o", help="Output directory (default: ./data/)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress logging")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument(
        "--summarize", "-s", action="store_true",
        help="Extract and display structured summary with key takeaways",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run end-to-end demo with mock data (no API keys needed)",
    )

    args = parser.parse_args()

    # Demo mode: run end-to-end verification with mock data
    if args.demo:
        _run_demo(verbose=not args.quiet, as_json=args.json)
        return

    topic = args.query or args.topic

    if not topic:
        parser.error("Please provide a research topic via positional arg or --topic")

    output_dir = Path(args.output) if args.output else None

    agent = ResearchAgent(
        model=args.model,
        depth=args.depth,
        verbose=not args.quiet,
        use_managed=not args.local,
        output_dir=output_dir,
    )

    result = asyncio.run(agent.research(topic))

    if args.json:
        output: dict[str, Any] = {
            "synthesis": result.synthesis,
            "files_saved": result.files_saved,
            "sources_consulted": result.sources_consulted,
            "scratchpad_entries": result.scratchpad_entries,
            "tool_calls_count": len(result.tool_calls),
            "timestamp": result.timestamp,
        }
        if args.summarize:
            output["summary"] = result.summarize().to_dict()
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 72)
        print("RESEARCH SYNTHESIS")
        print("=" * 72 + "\n")
        print(result.synthesis)
        print("\n" + "-" * 72)
        print(result.summary)
        if result.files_saved:
            print("\nFiles saved:")
            for f in result.files_saved:
                print(f"  - {f}")

        if args.summarize:
            research_summary = result.summarize()
            print("\n" + "=" * 72)
            print("STRUCTURED SUMMARY & KEY TAKEAWAYS")
            print("=" * 72 + "\n")
            print(research_summary.format_text())

        print()


def _run_demo(verbose: bool = True, as_json: bool = False) -> None:
    """Run an end-to-end demo with sample data to verify the pipeline.

    This exercises the full pipeline — Scratchpad, tool dispatch, ResearchResult,
    and summarization — without requiring API keys or network access.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(f"[demo] {msg}", file=sys.stderr)

    _log("Running end-to-end pipeline verification...")

    # 1. Scratchpad accumulation
    _log("Phase 1: Scratchpad accumulation")
    pad = Scratchpad()
    pad.add("Found 3 major papers on quantum error correction published in 2026.", "discovery_search")
    pad.add("IBM achieved 1000+ qubit processor; Google demonstrated quantum advantage in materials simulation.", "deep_dive")
    pad.add("Key gap: limited real-world commercial applications beyond simulation.", "follow_up")
    assert pad.count() == 3, "Scratchpad should have 3 entries"
    _log(f"  Scratchpad: {pad.count()} entries accumulated")

    # 2. Tool dispatch (save_notes)
    _log("Phase 2: Tool dispatch verification")
    result_json = asyncio.run(execute_tool(
        "save_notes",
        {"content": "Synthesis note: quantum computing is advancing rapidly", "label": "synthesis"},
        pad,
    ))
    parsed = json.loads(result_json)
    assert parsed["saved"] is True, "save_notes should succeed"
    assert pad.count() == 4
    _log(f"  Tool dispatch: save_notes OK ({pad.count()} entries)")

    # 3. Build a ResearchResult with realistic synthesis
    _log("Phase 3: ResearchResult construction")
    sample_synthesis = textwrap.dedent("""\
        # Quantum Computing: State of the Field (2026)

        ## Executive Summary

        Quantum computing has reached a critical inflection point in 2026, with major
        hardware milestones from IBM and Google, significant advances in error correction,
        and the first commercially viable quantum applications emerging in drug discovery
        and materials science.

        ## Key Findings

        - IBM's 1,121-qubit Condor processor demonstrated significant error-corrected computation
        - Google achieved quantum advantage in materials simulation, clearly outperforming classical methods
        - Quantum error correction has improved 10x, with new surface code implementations
        - The quantum software ecosystem is maturing, with Qiskit and Cirq seeing major updates
        - Early commercial applications in pharmaceutical molecular simulation are emerging
        - Cloud quantum access (IBM Quantum, Amazon Braket) has expanded to 15+ providers
        - Funding for quantum startups reached $4.2B in 2025, though growth may be slowing

        ## Hardware Advances

        The race for quantum supremacy continues, with IBM, Google, and new entrants
        like PsiQuantum pushing boundaries. IBM's Condor processor represents a substantial
        leap, while photonic quantum computing approaches could disrupt the field.

        ## Software & Algorithms

        New variational algorithms have demonstrated proven speedups for optimization
        problems. The development of quantum machine learning libraries has accelerated,
        though practical advantages over classical ML remain uncertain.

        ## Commercial Applications

        Drug discovery companies like Zapata and QC Ware have reported preliminary
        results showing quantum-enhanced molecular simulations outperforming classical
        methods for specific use cases. Financial services firms might adopt quantum
        risk modeling by 2027.

        ## Open Questions

        - When will fault-tolerant quantum computing become practical?
        - Can quantum advantage be maintained as classical algorithms improve?
        - What is the timeline for quantum cryptography to threaten current encryption?

        ## Sources

        - https://example.com/ibm-condor-2026
        - https://example.com/google-quantum-materials
        - https://example.com/quantum-market-report-2026
    """)

    result = ResearchResult(
        synthesis=sample_synthesis,
        tool_calls=[
            {"name": "tavily_search", "input": {"query": "quantum computing 2026"}},
            {"name": "tavily_search", "input": {"query": "quantum error correction advances"}},
            {"name": "fetch_url", "input": {"url": "https://example.com/ibm-condor-2026"}},
            {"name": "save_notes", "input": {"content": "IBM Condor findings", "label": "deep_dive"}},
            {"name": "tavily_search", "input": {"query": "quantum computing commercial applications"}},
            {"name": "save_report", "input": {"filename": "quantum_2026.md", "content": sample_synthesis}},
        ],
        files_saved=["/data/quantum_2026.md"],
        sources_consulted=5,
        scratchpad_entries=4,
    )
    _log(f"  ResearchResult: {result.summary}")

    # 4. Summarization and key-takeaway extraction
    _log("Phase 4: Summarization & key-takeaway extraction")
    research_summary = result.summarize()

    assert research_summary.executive_summary, "Should extract executive summary"
    assert len(research_summary.key_takeaways) >= 3, f"Should extract >=3 takeaways, got {len(research_summary.key_takeaways)}"
    assert len(research_summary.themes) >= 1, f"Should extract >=1 theme, got {len(research_summary.themes)}"
    assert len(research_summary.open_questions) >= 1, f"Should extract >=1 open question, got {len(research_summary.open_questions)}"
    assert research_summary.source_count == 5
    assert research_summary.word_count > 50

    _log(f"  Executive summary: {len(research_summary.executive_summary)} chars")
    _log(f"  Key takeaways: {len(research_summary.key_takeaways)}")
    _log(f"  Themes: {research_summary.themes}")
    _log(f"  Open questions: {len(research_summary.open_questions)}")

    # 5. Output
    if as_json:
        output = {
            "demo": True,
            "pipeline_status": "OK",
            "synthesis_word_count": research_summary.word_count,
            "summary": research_summary.to_dict(),
            "result_metadata": {
                "sources_consulted": result.sources_consulted,
                "scratchpad_entries": result.scratchpad_entries,
                "files_saved": result.files_saved,
                "tool_calls_count": len(result.tool_calls),
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 72)
        print("END-TO-END DEMO — PIPELINE VERIFICATION")
        print("=" * 72 + "\n")
        print(research_summary.format_text())
        print("\n" + "-" * 72)
        print(result.summary)
        print()

    _log("All assertions passed. Pipeline verified end-to-end.")


if __name__ == "__main__":
    main()
