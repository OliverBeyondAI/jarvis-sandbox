#!/usr/bin/env python3
"""
MCP Server — Topic Researcher tools exposed via the Model Context Protocol.

Wraps the existing tavily_search, fetch_url, and save_notes tools as a
standards-compliant MCP server using the `mcp` Python SDK.  This allows any
MCP-compatible client (Claude Desktop, Claude Agent SDK, etc.) to call these
tools over stdio or SSE transport.

Run with:
    python -m agents.topic_researcher.mcp_server                # stdio (default)
    python -m agents.topic_researcher.mcp_server --transport sse  # SSE on port 8000
"""

from __future__ import annotations

import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from .tools import (
    Scratchpad,
    fetch_url as _fetch_url,
    tavily_search as _tavily_search,
)

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="topic-researcher-tools",
    instructions=(
        "This server provides web search, URL fetching, and note-taking tools "
        "for the Topic Researcher agent.  Use tavily_search to discover sources, "
        "fetch_url to read full pages, and save_notes to accumulate findings."
    ),
)

# Module-level scratchpad shared across save_notes calls within a session.
_scratchpad = Scratchpad()


# ---------------------------------------------------------------------------
# Tool: tavily_search
# ---------------------------------------------------------------------------

@mcp.tool()
def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> str:
    """Search the web for current information on any topic using Tavily.

    Returns results with title, URL, and content snippet.

    Args:
        query: The search query to look up.
        max_results: Maximum number of results (1-10, default 5).
        search_depth: 'basic' is faster, 'advanced' is more thorough. Default: 'basic'.
        topic: 'general' or 'news'. 'news' returns recent articles. Default: 'general'.
        include_domains: Optional list of domains to restrict search to.
        exclude_domains: Optional list of domains to exclude from search.

    Returns:
        JSON string with a list of search results.
    """
    results = _tavily_search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    )
    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Tool: fetch_url
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_url(url: str) -> str:
    """Fetch and extract text content from a URL.

    Strips HTML tags and returns clean text.  Use this to read full articles
    or pages discovered via search.

    Args:
        url: The URL to fetch (must start with http:// or https://).

    Returns:
        The extracted text content of the page.
    """
    return _fetch_url(url)


# ---------------------------------------------------------------------------
# Tool: save_notes
# ---------------------------------------------------------------------------

@mcp.tool()
def save_notes(content: str, label: str = "notes") -> str:
    """Save research notes to an internal scratchpad.

    Use this to accumulate findings, observations, and analysis before
    producing the final summary.

    Args:
        content: The notes to save.
        label: A short label for this set of notes.

    Returns:
        Confirmation string with the number of entries saved.
    """
    return _scratchpad.add(content, label)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
