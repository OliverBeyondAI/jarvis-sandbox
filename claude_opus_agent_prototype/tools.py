#!/usr/bin/env python3
"""
Tools — Web search, URL fetching, and file I/O for the Opus agent.

Reuses the tool infrastructure from agents.tools with Opus-specific
schema formatting. Each tool has an async implementation and a JSON
schema definition compatible with the Anthropic messages API.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx

from .config import DATA_DIR


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic messages API format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "tavily_search",
        "description": (
            "Search the web for current information on any topic. "
            "Returns results with titles, URLs, and content snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (1-10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the text content of a web page URL. "
            "Returns extracted text with HTML tags stripped."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

async def tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web via Tavily API."""
    def _sync() -> dict[str, Any]:
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
                    }
                    for r in results.get("results", [])
                ],
            }
        except ImportError:
            return {"query": query, "error": "tavily-python not installed"}
        except Exception as e:
            return {"query": query, "error": str(e)}

    return await asyncio.to_thread(_sync)


async def fetch_url(url: str) -> dict[str, Any]:
    """Fetch and extract text from a URL."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=30.0,
            headers={"User-Agent": "OpusAgentPrototype/0.1"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text
            if "text/html" in resp.headers.get("content-type", ""):
                text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 50_000:
                text = text[:50_000] + "\n\n[Truncated at 50,000 chars]"
            return {"url": url, "status": resp.status_code, "content": text}
    except Exception as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}


def read_file(path: str) -> dict[str, Any]:
    """Read a file, resolving relative paths against DATA_DIR."""
    try:
        fp = Path(path) if Path(path).is_absolute() else DATA_DIR / path
        return {"path": str(fp), "content": fp.read_text(encoding="utf-8")}
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write content to a file."""
    try:
        fp = Path(path) if Path(path).is_absolute() else DATA_DIR / path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return {"path": str(fp), "status": "written", "size": len(content)}
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

_TOOL_REGISTRY = {
    "tavily_search": tavily_search,
    "fetch_url": fetch_url,
    "read_file": read_file,
    "write_file": write_file,
}


async def execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """Dispatch a tool call by name. Returns JSON string."""
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        if asyncio.iscoroutinefunction(fn):
            result = await fn(**input_dict)
        else:
            result = fn(**input_dict)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})


def summarize_input(input_dict: dict[str, Any]) -> str:
    """Short summary of tool input for logging."""
    for key in ("url", "query", "path"):
        if key in input_dict:
            val = str(input_dict[key])
            return f'{key}="{val[:50]}..."' if len(val) > 50 else f'{key}="{val}"'
    return json.dumps(input_dict, default=str)[:60]
