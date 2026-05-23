"""
Topic Researcher — Claude Agent SDK prototype with Tavily web search.

A focused agent that takes a natural language research request, searches the web
via Tavily, and produces a structured summary of findings. Demonstrates basic
tool use with the Claude Agent SDK.

Usage:
    python -m agents.topic_researcher "summarize the latest AI image generation models"
    python -m agents.topic_researcher --topic "quantum computing 2026" --depth brief
"""

from .agent import TopicResearcher, ResearchResult, _extract_json_object
from .tools import (
    TOOLS,
    Scratchpad,
    execute_tool,
    tavily_search,
    fetch_url,
    save_notes,
    TavilyConfigError,
    TavilySearchError,
)

__all__ = [
    "TopicResearcher",
    "ResearchResult",
    "_extract_json_object",
    "TOOLS",
    "Scratchpad",
    "execute_tool",
    "tavily_search",
    "fetch_url",
    "save_notes",
    "TavilyConfigError",
    "TavilySearchError",
]
