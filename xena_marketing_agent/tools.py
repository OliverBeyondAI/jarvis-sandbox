"""
Tool Definitions & Implementations — Market research, competitor analysis,
content drafting, and campaign saving.

Tools are defined as Anthropic custom tool schemas (with ``type: "custom"``)
for use with the beta managed-agents API.  Tool implementations are
synchronous so they can be called directly from the event loop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic custom tool format)
# ---------------------------------------------------------------------------

MARKET_RESEARCH_TOOL: dict[str, Any] = {
    "name": "market_research",
    "type": "custom",
    "description": (
        "Search the web for market intelligence — trends, competitor activity, "
        "audience insights, and industry data. Use targeted queries to gather "
        "information that will inform the marketing strategy and messaging."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Targeted search query for market research.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (1-10).",
                "default": 5,
            },
            "search_depth": {
                "type": "string",
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
        "Fetch the full text content of a web page. Use this to deep-dive "
        "into competitor websites, industry reports, or product pages to "
        "extract messaging patterns, positioning, and market data."
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

ANALYZE_MARKET_TOOL: dict[str, Any] = {
    "name": "analyze_market",
    "type": "custom",
    "description": (
        "Record structured market analysis findings — audience segments, "
        "competitor positioning, messaging opportunities, and market gaps. "
        "Use this after each research phase to organize insights before "
        "generating content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "description": "Current analysis phase.",
                "enum": [
                    "market_landscape",
                    "competitor_analysis",
                    "audience_research",
                    "messaging_strategy",
                    "content_planning",
                ],
            },
            "insights": {
                "type": "array",
                "description": "List of market insights from this phase.",
                "items": {
                    "type": "object",
                    "properties": {
                        "headline": {
                            "type": "string",
                            "description": "One-line insight summary.",
                        },
                        "detail": {
                            "type": "string",
                            "description": "Supporting detail or evidence.",
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "market_trend",
                                "competitor_intel",
                                "audience_insight",
                                "messaging_opportunity",
                                "positioning_gap",
                                "risk",
                            ],
                            "default": "market_trend",
                        },
                        "source_urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs supporting this insight.",
                        },
                    },
                    "required": ["headline"],
                },
            },
            "gaps": {
                "type": "array",
                "description": "Research gaps needing additional investigation.",
                "items": {"type": "string"},
            },
        },
        "required": ["phase", "insights"],
    },
}

DRAFT_CONTENT_TOOL: dict[str, Any] = {
    "name": "draft_content",
    "type": "custom",
    "description": (
        "Record a drafted piece of marketing content for a specific channel. "
        "Use this to save each content piece as you generate it — landing page "
        "copy, email sequences, social posts, blog outlines, ad copy, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Marketing channel for this content.",
                "enum": [
                    "landing_page",
                    "email_sequence",
                    "social_media",
                    "blog_post",
                    "ad_copy",
                    "press_release",
                    "product_description",
                ],
            },
            "title": {
                "type": "string",
                "description": "Title or headline for the content piece.",
            },
            "body": {
                "type": "string",
                "description": "The full content body (markdown formatted).",
            },
            "cta": {
                "type": "string",
                "description": "Call-to-action text.",
            },
            "notes": {
                "type": "string",
                "description": "Internal notes about this content piece.",
            },
        },
        "required": ["channel", "title", "body"],
    },
}

SAVE_CAMPAIGN_TOOL: dict[str, Any] = {
    "name": "save_campaign",
    "type": "custom",
    "description": (
        "Save the complete marketing campaign as a markdown file. "
        "Call this exactly ONCE when all content has been drafted, reviewed, "
        "and finalized. The content should be the full campaign document."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": (
                    "Filename for the campaign "
                    "(e.g. 'acme_launch_campaign.md')."
                ),
            },
            "content": {
                "type": "string",
                "description": "Full markdown content of the campaign document.",
            },
        },
        "required": ["filename", "content"],
    },
}

ALL_TOOLS: list[dict[str, Any]] = [
    MARKET_RESEARCH_TOOL,
    FETCH_URL_TOOL,
    ANALYZE_MARKET_TOOL,
    DRAFT_CONTENT_TOOL,
    SAVE_CAMPAIGN_TOOL,
]


# ---------------------------------------------------------------------------
# Tool Implementations (all async)
# ---------------------------------------------------------------------------


def market_research(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> dict[str, Any]:
    """Search the web for market intelligence using Tavily."""
    try:
        from tavily import TavilyClient

        client = TavilyClient()
        results = client.search(
            query=query,
            max_results=min(max_results, 10),
            search_depth=search_depth,
        )
        items = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "relevance_score": min(max(r.get("score", 0), 0.0), 1.0),
            }
            for r in results.get("results", [])
        ]
        return {
            "query": query,
            "results": items,
            "result_count": len(items),
        }
    except ImportError:
        return {
            "query": query,
            "error": "Tavily not installed. Install with: pip install tavily-python",
        }
    except Exception as e:
        return {"query": query, "error": f"Search failed: {e}"}


def fetch_url(url: str) -> dict[str, Any]:
    """Fetch text content from a URL with HTML stripping."""
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "XenaMarketingAgent/1.0",
                "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
            },
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            text = response.text
            if "text/html" in content_type:
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)

            max_chars = 50_000
            if len(text) > max_chars:
                text = (
                    text[:max_chars]
                    + "\n\n[Content truncated at 50,000 characters]"
                )

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


def analyze_market(
    phase: str,
    insights: list[dict[str, Any]],
    gaps: list[str] | None = None,
) -> dict[str, Any]:
    """Record structured market analysis findings."""
    return {
        "phase": phase,
        "insights_recorded": len(insights),
        "insights": insights,
        "gaps": gaps or [],
        "status": "recorded",
    }


def draft_content(
    channel: str,
    title: str,
    body: str,
    cta: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Record a drafted content piece."""
    return {
        "channel": channel,
        "title": title,
        "body_length": len(body),
        "cta": cta,
        "notes": notes,
        "status": "drafted",
    }


def save_campaign(
    filename: str,
    content: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Save a marketing campaign document to disk."""
    target_dir = output_dir or os.environ.get(
        "XENA_OUTPUT_DIR", "./marketing_output"
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

_output_dir: str | None = None


def set_output_dir(d: str) -> None:
    """Configure the output directory for save_campaign."""
    global _output_dir
    _output_dir = d


def execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """Execute a tool by name and return a JSON string result."""
    try:
        if name == "market_research":
            result = market_research(**input_dict)
        elif name == "fetch_url":
            result = fetch_url(**input_dict)
        elif name == "analyze_market":
            result = analyze_market(**input_dict)
        elif name == "draft_content":
            result = draft_content(**input_dict)
        elif name == "save_campaign":
            result = save_campaign(output_dir=_output_dir, **input_dict)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps(
            {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
        )
