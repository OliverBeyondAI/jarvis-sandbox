"""
Core agent logic for the Topic Researcher.

Uses the Claude Agent SDK to orchestrate a research loop:
  1. Accept a topic / research question
  2. Search the web via Tavily
  3. Optionally fetch full pages
  4. Accumulate notes
  5. Produce a structured ResearchResult
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .tools import TOOLS, Scratchpad, execute_tool


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict | None:
    """Extract the first valid JSON object from *text*.

    Uses a bracket-counting approach instead of a greedy regex so that it
    correctly handles responses containing multiple JSON objects or trailing
    prose after the closing brace.
    """
    # Try parsing the whole string first (fast path)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Find the first '{' and use bracket counting to locate its match
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResearchResult:
    """Structured output from a research session."""
    topic: str
    summary: str
    key_findings: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    raw_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "sources": self.sources,
        }

    def to_markdown(self) -> str:
        lines = [f"# Research: {self.topic}", "", self.summary, ""]
        if self.key_findings:
            lines.append("## Key Findings")
            for finding in self.key_findings:
                lines.append(f"- {finding}")
            lines.append("")
        if self.sources:
            lines.append("## Sources")
            for src in self.sources:
                title = src.get("title", src.get("url", ""))
                url = src.get("url", "")
                lines.append(f"- [{title}]({url})")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant. Given a topic, use the available tools to search \
the web, read relevant pages, and compile a comprehensive summary.

Follow this workflow:
1. Use tavily_search to find relevant sources on the topic.
2. Use fetch_url to read the most promising pages (up to 3).
3. Use save_notes to record key findings as you go.
4. When you have enough information, respond with a final JSON object:

{
  "summary": "A concise 2-3 paragraph summary of the topic.",
  "key_findings": ["Finding 1", "Finding 2", ...],
  "sources": [{"title": "...", "url": "..."}, ...]
}

Be thorough but concise. Focus on factual, well-sourced information.\
"""

DEPTH_INSTRUCTIONS = {
    "brief": "Keep it short — 1 paragraph summary, top 3 findings.",
    "standard": "Provide a balanced summary — 2-3 paragraphs, 5-7 findings.",
    "deep": "Go deep — comprehensive summary, 10+ findings, read multiple sources.",
}


class TopicResearcher:
    """Agent that researches a topic using Claude + Tavily web search."""

    def __init__(
        self,
        model: str = "claude-opus-4-7-20250501",
        max_turns: int = 10,
    ):
        self.model = model
        self.max_turns = max_turns
        self.client = anthropic.Anthropic()

    def research(
        self,
        topic: str,
        depth: str = "standard",
    ) -> ResearchResult:
        """Run the research loop and return structured results."""
        scratchpad = Scratchpad()

        depth_hint = DEPTH_INSTRUCTIONS.get(depth, DEPTH_INSTRUCTIONS["standard"])
        user_message = f"Research the following topic: {topic}\n\nDepth: {depth_hint}"

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        for _turn in range(self.max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # Check if the model wants to use tools
            if response.stop_reason == "tool_use":
                # Process all tool calls in this response
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        result = execute_tool(
                            block.name, block.input, scratchpad=scratchpad
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

            else:
                # Model produced a final text response — extract the result
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text

                return self._parse_result(topic, final_text, scratchpad)

        # Fallback if we hit max turns
        return ResearchResult(
            topic=topic,
            summary="Research terminated after reaching maximum turns.",
            raw_notes=scratchpad.dump(),
        )

    def _parse_result(
        self, topic: str, text: str, scratchpad: Scratchpad
    ) -> ResearchResult:
        """Parse the model's final JSON response into a ResearchResult."""
        cleaned = text.strip()

        # Strip markdown fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        # Try direct parse first, then extract the first valid JSON object
        data = _extract_json_object(cleaned)
        if data is not None:
            return ResearchResult(
                topic=topic,
                summary=data.get("summary", cleaned),
                key_findings=data.get("key_findings", []),
                sources=data.get("sources", []),
                raw_notes=scratchpad.dump(),
            )

        # Fallback: use the raw text as the summary
        return ResearchResult(
            topic=topic,
            summary=text.strip(),
            raw_notes=scratchpad.dump(),
        )
