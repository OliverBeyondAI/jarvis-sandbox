"""
Tool definitions and execution for the Topic Researcher agent.

Provides web search via Tavily, URL fetching, and note-saving capabilities
as Claude Agent SDK tool schemas.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tavily client configuration
# ---------------------------------------------------------------------------

def _get_tavily_api_key() -> str | None:
    """Resolve the Tavily API key from environment variables.

    Checks (in order):
      1. TAVILY_API_KEY
      2. TAVILY_KEY  (common alias)
    """
    return os.environ.get("TAVILY_API_KEY") or os.environ.get("TAVILY_KEY") or None


def _get_tavily_client():
    """Create a configured TavilyClient, raising a clear error if unavailable."""
    api_key = _get_tavily_api_key()
    if not api_key:
        raise TavilyConfigError(
            "Tavily API key not found. Set the TAVILY_API_KEY environment variable. "
            "Get a key at https://tavily.com"
        )
    try:
        from tavily import TavilyClient
    except ImportError:
        raise TavilyConfigError(
            "tavily-python package is not installed. "
            "Install it with: pip install tavily-python"
        )
    return TavilyClient(api_key=api_key)


class TavilyConfigError(Exception):
    """Raised when the Tavily client cannot be configured."""


class TavilySearchError(Exception):
    """Raised when a Tavily search fails."""


# ---------------------------------------------------------------------------
# Tool schemas (Claude Agent SDK format)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "tavily_search",
        "description": (
            "Search the web for a given query using Tavily. Returns a list of "
            "results with title, URL, and content snippet. Supports different "
            "search depths and topic categories."
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
                    "description": "Maximum number of results to return (1-10, default 5).",
                    "default": 5,
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": (
                        "Search depth — 'basic' is faster, 'advanced' is more thorough "
                        "and includes additional content extraction. Default: 'basic'."
                    ),
                    "default": "basic",
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news"],
                    "description": (
                        "Topic category to focus the search. "
                        "'news' returns recent articles. Default: 'general'."
                    ),
                    "default": "general",
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of domains to restrict search to (e.g. ['arxiv.org', 'nature.com']).",
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of domains to exclude from search.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the text content of a URL. Useful for reading full articles "
            "or pages discovered via search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "save_notes",
        "description": (
            "Save research notes to an internal scratchpad. Use this to accumulate "
            "findings before producing the final summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The notes to save.",
                },
                "label": {
                    "type": "string",
                    "description": "A short label for this set of notes.",
                },
            },
            "required": ["content"],
        },
    },
]


# ---------------------------------------------------------------------------
# Scratchpad for accumulated notes
# ---------------------------------------------------------------------------

@dataclass
class Scratchpad:
    """In-memory scratchpad for research notes."""
    entries: list[dict[str, str]] = field(default_factory=list)

    def add(self, content: str, label: str = "notes") -> str:
        self.entries.append({"label": label, "content": content})
        return f"Saved notes under '{label}' ({len(self.entries)} total entries)."

    def dump(self) -> str:
        if not self.entries:
            return "(no notes saved yet)"
        parts = []
        for i, entry in enumerate(self.entries, 1):
            parts.append(f"[{i}] {entry['label']}:\n{entry['content']}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> list[dict[str, str]]:
    """Run a Tavily web search and return structured results.

    Returns a list of dicts, each with keys: title, url, content, score.
    On error, returns a single-element list with an 'error' key.
    """
    max_results = max(1, min(max_results, 10))

    try:
        client = _get_tavily_client()
    except TavilyConfigError as exc:
        return [{"error": str(exc)}]

    search_kwargs: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
    }
    if include_domains:
        search_kwargs["include_domains"] = include_domains
    if exclude_domains:
        search_kwargs["exclude_domains"] = exclude_domains

    try:
        response = client.search(**search_kwargs)
    except Exception as exc:
        error_type = type(exc).__name__
        return [{"error": f"Tavily search failed ({error_type}): {exc}"}]

    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": str(item.get("score", "")),
        })

    if not results:
        return [{"error": f"No results found for query: {query}"}]

    return results


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace to extract readable text."""
    import re

    # Remove script and style blocks entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        text = text.replace(entity, char)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_url(url: str) -> str:
    """Fetch text content from a URL with robust error handling."""
    import httpx

    if not url.startswith(("http://", "https://")):
        return "Error: Invalid URL scheme. URL must start with http:// or https://"

    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "TopicResearcher/0.1"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            raw = resp.text
            if "html" in content_type.lower():
                raw = _strip_html(raw)
            max_chars = 15_000
            if len(raw) > max_chars:
                return (
                    raw[:max_chars]
                    + f"\n\n[Content truncated: showing {max_chars:,} of {len(raw):,} characters]"
                )
            return raw
    except httpx.TimeoutException:
        return f"Error: Request timed out fetching {url}"
    except httpx.HTTPStatusError as exc:
        return f"Error: HTTP {exc.response.status_code} fetching {url}"
    except httpx.HTTPError as exc:
        return f"Error fetching URL: {exc}"


def save_notes(scratchpad: Scratchpad, content: str, label: str = "notes") -> str:
    """Save notes to the scratchpad."""
    return scratchpad.add(content, label)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    scratchpad: Scratchpad | None = None,
) -> str:
    """Execute a tool by name and return the result as a JSON string."""
    if tool_name == "tavily_search":
        results = tavily_search(
            query=tool_input["query"],
            max_results=tool_input.get("max_results", 5),
            search_depth=tool_input.get("search_depth", "basic"),
            topic=tool_input.get("topic", "general"),
            include_domains=tool_input.get("include_domains"),
            exclude_domains=tool_input.get("exclude_domains"),
        )
        return json.dumps(results, indent=2)

    elif tool_name == "fetch_url":
        return fetch_url(tool_input["url"])

    elif tool_name == "save_notes":
        pad = scratchpad or Scratchpad()
        return save_notes(
            pad,
            content=tool_input["content"],
            label=tool_input.get("label", "notes"),
        )

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
