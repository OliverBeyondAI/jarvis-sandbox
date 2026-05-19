#!/usr/bin/env python3
"""
MCP Tool Integration — Tavily Web Search, URL Fetching, and File I/O.

Defines tools in MCP-compatible format with schemas and async implementations
that can be wired into the Claude Agent SDK managed-agents loop.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# File I/O Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic custom tool format for managed agents)
# ---------------------------------------------------------------------------

FETCH_URL_TOOL: dict[str, Any] = {
    "name": "fetch_url",
    "type": "custom",
    "description": (
        "Fetch the text content of a web page at the given URL. "
        "Returns the extracted text content, stripping HTML tags. "
        "Use this to retrieve and read the full content of a web page."
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

TAVILY_SEARCH_TOOL: dict[str, Any] = {
    "name": "tavily_search",
    "type": "custom",
    "description": (
        "Search the web using Tavily for current information, context, "
        "or research on any topic. Returns search results with titles, "
        "URLs, and content snippets."
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

READ_FILE_TOOL: dict[str, Any] = {
    "name": "read_file",
    "type": "custom",
    "description": (
        "Read the contents of a file from the local filesystem. "
        "Use this to load brand guides, reference materials, or any text file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            },
        },
        "required": ["path"],
    },
}

WRITE_FILE_TOOL: dict[str, Any] = {
    "name": "write_file",
    "type": "custom",
    "description": (
        "Write content to a file on the local filesystem. "
        "Creates parent directories if they don't exist."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path where the file should be written.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
}

# All tools available to agents
ALL_TOOLS: list[dict[str, Any]] = [
    FETCH_URL_TOOL,
    TAVILY_SEARCH_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
]


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
                "User-Agent": "AgentsFramework/0.1",
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
# File I/O Implementations
# ---------------------------------------------------------------------------

def read_file(path: str) -> dict[str, Any]:
    """Read a file from the filesystem."""
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = DATA_DIR / file_path
        content = file_path.read_text(encoding="utf-8")
        return {
            "path": str(file_path),
            "content": content,
            "size": len(content),
        }
    except FileNotFoundError:
        return {"path": path, "error": f"File not found: {path}"}
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write content to a file on the filesystem."""
    try:
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = DATA_DIR / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return {
            "path": str(file_path),
            "size": len(content),
            "status": "written",
        }
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """
    Execute a tool by name and return a JSON string result.

    Central dispatch function that routes tool calls from the agent loop
    to the appropriate async implementation.
    """
    try:
        if name == "fetch_url":
            result = await fetch_url(**input_dict)
        elif name == "tavily_search":
            result = await tavily_search(**input_dict)
        elif name == "read_file":
            result = read_file(**input_dict)
        elif name == "write_file":
            result = write_file(**input_dict)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})
