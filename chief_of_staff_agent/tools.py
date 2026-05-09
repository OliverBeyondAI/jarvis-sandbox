"""
Tool Definitions & Implementations — Web search, URL fetching, and report saving.

Tools are defined as MCP-compatible JSON schemas and dispatched by name.
All implementations are async.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .models import Source


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic custom tool format)
# ---------------------------------------------------------------------------

TAVILY_SEARCH_TOOL: dict[str, Any] = {
    "name": "tavily_search",
    "type": "custom",
    "description": (
        "Search the web using Tavily for current information on any topic. "
        "Returns search results with titles, URLs, and content snippets. "
        "Use this to discover facts, recent developments, and expert opinions."
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
                "description": "Maximum number of results to return (1-10).",
                "default": 5,
            },
            "search_depth": {
                "type": "string",
                "description": "Search depth: 'basic' for quick results, 'advanced' for thorough research.",
                "enum": ["basic", "advanced"],
                "default": "advanced",
            },
        },
        "required": ["query"],
    },
}

FETCH_URL_TOOL: dict[str, Any] = {
    "name": "fetch_url",
    "type": "custom",
    "description": (
        "Fetch the full text content of a web page at the given URL. "
        "Returns extracted text with HTML stripped. Use this to read "
        "full articles, reports, or documentation pages."
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

SAVE_REPORT_TOOL: dict[str, Any] = {
    "name": "save_report",
    "type": "custom",
    "description": (
        "Save the final research report as a markdown file. "
        "Call this once when your research is complete to persist "
        "the executive briefing to disk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The filename for the report (e.g. 'ai_regulation_briefing.md').",
            },
            "content": {
                "type": "string",
                "description": "The full markdown content of the report.",
            },
        },
        "required": ["filename", "content"],
    },
}

ALL_TOOLS: list[dict[str, Any]] = [
    TAVILY_SEARCH_TOOL,
    FETCH_URL_TOOL,
    SAVE_REPORT_TOOL,
]


# ---------------------------------------------------------------------------
# Tool Implementations (all async)
# ---------------------------------------------------------------------------


async def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> dict[str, Any]:
    """Search the web using the Tavily API."""

    def _sync_search() -> dict[str, Any]:
        try:
            from tavily import TavilyClient

            client = TavilyClient()
            results = client.search(
                query=query,
                max_results=min(max_results, 10),
                search_depth=search_depth,
            )
            sources = [
                Source(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    relevance_score=min(max(r.get("score", 0), 0.0), 1.0),
                )
                for r in results.get("results", [])
            ]
            return {
                "query": query,
                "results": [s.model_dump(mode="json") for s in sources],
            }
        except ImportError:
            return {
                "query": query,
                "error": "Tavily not installed. Install with: pip install tavily-python",
            }
        except Exception as e:
            return {"query": query, "error": f"Search failed: {e}"}

    return await asyncio.to_thread(_sync_search)


async def fetch_url(url: str) -> dict[str, Any]:
    """Fetch text content from a URL with HTML stripping."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "ChiefOfStaffAgent/1.0",
                "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            text = response.text
            if "text/html" in content_type:
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)

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


async def save_report(
    filename: str,
    content: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Save a markdown report to disk."""
    target_dir = output_dir or os.environ.get(
        "COS_OUTPUT_DIR", "./chief_of_staff_reports"
    )
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".md"):
        filename = f"{filename}.md"

    filepath = path / filename
    filepath.write_text(content, encoding="utf-8")

    return {
        "saved": True,
        "path": str(filepath.resolve()),
        "size_bytes": len(content.encode("utf-8")),
    }


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

# Stash output_dir so save_report can use it (set by agent before run)
_output_dir: str | None = None


def set_output_dir(d: str) -> None:
    """Configure the output directory for save_report."""
    global _output_dir
    _output_dir = d


async def execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """Execute a tool by name and return a JSON string result."""
    try:
        if name == "tavily_search":
            result = await tavily_search(**input_dict)
        elif name == "fetch_url":
            result = await fetch_url(**input_dict)
        elif name == "save_report":
            result = await save_report(output_dir=_output_dir, **input_dict)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})
