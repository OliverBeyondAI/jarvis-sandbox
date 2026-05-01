#!/usr/bin/env python3
"""
MCP Tool Integration — Tavily Web Search and URL Content Fetching

Defines tools in MCP-compatible format with schemas and async implementations
that can be wired into the Claude Agent SDK managed-agents loop.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic custom tool format for managed agents)
# ---------------------------------------------------------------------------

FETCH_URL_TOOL = {
    "name": "fetch_url",
    "type": "custom",
    "description": (
        "Fetch the text content of a web page at the given URL. "
        "Returns the extracted text content, stripping HTML tags. "
        "Use this to retrieve and read the full content of each research URL."
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

TAVILY_SEARCH_TOOL = {
    "name": "tavily_search",
    "type": "custom",
    "description": (
        "Search the web using Tavily for additional context, background "
        "information, or related work on a research topic. Returns search "
        "results with titles, URLs, and content snippets. Use this when you "
        "need to supplement your understanding of a source with broader context."
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
        },
        "required": ["query"],
    },
}

# All tools available to the agent
ALL_TOOLS: list[dict[str, Any]] = [FETCH_URL_TOOL, TAVILY_SEARCH_TOOL]


# ---------------------------------------------------------------------------
# Tool Implementations (all async)
# ---------------------------------------------------------------------------

async def fetch_url(url: str) -> dict[str, Any]:
    """Fetch text content from a URL using httpx with HTML stripping."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "ResearchSummarizerAgent/0.1",
                "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            text = response.text
            if "text/html" in content_type:
                # Remove script and style blocks
                text = re.sub(
                    r"<(script|style|noscript)[^>]*>.*?</\1>",
                    "", text, flags=re.DOTALL | re.IGNORECASE,
                )
                # Remove HTML comments
                text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
                # Remove HTML tags
                text = re.sub(r"<[^>]+>", " ", text)
                # Decode common HTML entities
                text = (
                    text.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&quot;", '"')
                    .replace("&#39;", "'")
                    .replace("&nbsp;", " ")
                )
                # Collapse whitespace
                text = re.sub(r"\s+", " ", text).strip()

            # Truncate very long content to avoid token overflow
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


async def tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web using the Tavily API (async-safe via thread executor)."""
    def _sync_search() -> dict[str, Any]:
        try:
            from tavily import TavilyClient

            client = TavilyClient()
            results = client.search(query=query, max_results=min(max_results, 10))
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
            return {
                "query": query,
                "error": "Tavily client not installed. Install with: pip install tavily-python",
            }
        except Exception as e:
            return {"query": query, "error": f"Tavily search failed: {e}"}

    return await asyncio.to_thread(_sync_search)


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """
    Execute a tool by name and return a JSON string result.

    This is the central dispatch function that routes tool calls from the
    agent loop to the appropriate async implementation.
    """
    try:
        if name == "fetch_url":
            result = await fetch_url(**input_dict)
        elif name == "tavily_search":
            result = await tavily_search(**input_dict)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})
