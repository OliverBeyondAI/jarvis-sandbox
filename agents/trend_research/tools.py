"""
Tool definitions for the Trend Research multi-agent system.

Defines tools in MCP-compatible format (matching the pattern in agents/tools.py)
with specialized research and storage operations.
"""

from __future__ import annotations

import json
from typing import Any

from ..tools import fetch_url, tavily_search
from .config import Config
from .models import ResearchReport, Source
from .s3_storage import S3Storage
from .tavily_client import TavilyResearchClient


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic custom tool format)
# ---------------------------------------------------------------------------

SEARCH_TRENDS_TOOL: dict[str, Any] = {
    "name": "search_trends",
    "type": "custom",
    "description": (
        "Search for emerging trends in a given domain or industry. "
        "Returns a list of sources with titles, URLs, and content snippets "
        "relevant to current trends and developments."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "The domain or industry to search for trends in (e.g. 'artificial intelligence', 'biotech', 'climate tech').",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10).",
                "default": 10,
            },
        },
        "required": ["domain"],
    },
}

DEEP_DIVE_TOOL: dict[str, Any] = {
    "name": "deep_dive",
    "type": "custom",
    "description": (
        "Perform an in-depth research search on a specific topic or trend. "
        "Uses advanced search depth for more comprehensive results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The specific topic or trend to research in depth.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10).",
                "default": 10,
            },
        },
        "required": ["topic"],
    },
}

SEARCH_NEWS_TOOL: dict[str, Any] = {
    "name": "search_news",
    "type": "custom",
    "description": (
        "Search for recent news articles and developments about a topic. "
        "Optimized for finding the latest information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic to find recent news about.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-5).",
                "default": 5,
            },
        },
        "required": ["topic"],
    },
}

STORE_REPORT_TOOL: dict[str, Any] = {
    "name": "store_report",
    "type": "custom",
    "description": (
        "Store a completed research report to persistent storage (S3 or local). "
        "The report is saved as JSON with a generated storage key."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title of the research report.",
            },
            "domain": {
                "type": "string",
                "description": "Domain/topic area of the report.",
            },
            "executive_summary": {
                "type": "string",
                "description": "Executive summary of findings.",
            },
            "report_json": {
                "type": "string",
                "description": "Full report content as a JSON string.",
            },
        },
        "required": ["title", "domain", "executive_summary", "report_json"],
    },
}

FETCH_URL_TOOL: dict[str, Any] = {
    "name": "fetch_url",
    "type": "custom",
    "description": (
        "Fetch the text content of a web page at the given URL. "
        "Returns extracted text content with HTML tags stripped."
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

# All tools available to trend research agents
ALL_TOOLS: list[dict[str, Any]] = [
    SEARCH_TRENDS_TOOL,
    DEEP_DIVE_TOOL,
    SEARCH_NEWS_TOOL,
    STORE_REPORT_TOOL,
    FETCH_URL_TOOL,
]


# ---------------------------------------------------------------------------
# Shared instances (lazy-initialized per config)
# ---------------------------------------------------------------------------

_config: Config | None = None
_tavily_client: TavilyResearchClient | None = None
_storage: S3Storage | None = None


def _get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def _get_tavily() -> TavilyResearchClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyResearchClient(_get_config())
    return _tavily_client


def _get_storage() -> S3Storage:
    global _storage
    if _storage is None:
        _storage = S3Storage(_get_config())
    return _storage


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """
    Execute a tool by name and return a JSON string result.

    Routes tool calls to the appropriate async implementation.
    """
    try:
        if name == "search_trends":
            sources = await _get_tavily().search_trends(
                domain=input_dict["domain"],
                max_results=input_dict.get("max_results"),
            )
            result = {
                "domain": input_dict["domain"],
                "sources": [s.model_dump(mode="json") for s in sources],
                "count": len(sources),
            }

        elif name == "deep_dive":
            sources = await _get_tavily().deep_dive(
                topic=input_dict["topic"],
                max_results=input_dict.get("max_results"),
            )
            result = {
                "topic": input_dict["topic"],
                "sources": [s.model_dump(mode="json") for s in sources],
                "count": len(sources),
            }

        elif name == "search_news":
            sources = await _get_tavily().search_news(
                topic=input_dict["topic"],
                max_results=input_dict.get("max_results"),
            )
            result = {
                "topic": input_dict["topic"],
                "sources": [s.model_dump(mode="json") for s in sources],
                "count": len(sources),
            }

        elif name == "store_report":
            report_data = json.loads(input_dict["report_json"])
            report = ResearchReport(
                title=input_dict["title"],
                domain=input_dict["domain"],
                executive_summary=input_dict["executive_summary"],
                metadata=report_data,
            )
            storage = _get_storage()
            key = report.to_storage_key()
            path = await storage.store_json(key, report.model_dump(mode="json"))
            result = {"stored_at": path, "key": key, "title": report.title}

        elif name == "fetch_url":
            result = await fetch_url(input_dict["url"])

        else:
            result = {"error": f"Unknown tool: {name}"}

        return json.dumps(result, default=str)

    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})
