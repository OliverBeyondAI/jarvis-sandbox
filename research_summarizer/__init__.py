"""
Research Summarizer Agent — AI-powered URL research and summarization.

Built on the Claude Agent SDK (Anthropic managed-agents beta API) with
custom tool integration for Tavily web search and URL content fetching.
"""

from .agent import ResearchSummarizerAgent, SummaryResult
from .formatter import format_output, format_terminal, format_markdown, format_json
from .tools import ALL_TOOLS, execute_tool, fetch_url, tavily_search

__all__ = [
    "ResearchSummarizerAgent",
    "SummaryResult",
    "format_output",
    "format_terminal",
    "format_markdown",
    "format_json",
    "ALL_TOOLS",
    "execute_tool",
    "fetch_url",
    "tavily_search",
]
