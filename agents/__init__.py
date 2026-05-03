"""
Agents — Modular AI agent framework with tool integration.

Built on the Claude Agent SDK with custom tool integration for
Tavily web search and URL content fetching. Includes a specialized
Meeting Prep Agent for generating pre-meeting briefing documents.
"""

from .agent import Agent, AgentResult
from .meeting_prep_agent import (
    BriefingResult,
    MeetingPrepAgent,
    run_pipeline as run_meeting_prep_pipeline,
)
from .tools import ALL_TOOLS, execute_tool, fetch_url, tavily_search

__all__ = [
    "Agent",
    "AgentResult",
    "ALL_TOOLS",
    "BriefingResult",
    "MeetingPrepAgent",
    "execute_tool",
    "fetch_url",
    "run_meeting_prep_pipeline",
    "tavily_search",
]
