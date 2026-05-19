"""
Agents — Modular AI agent framework with tool integration.

Built on the Claude Agent SDK with custom tool integration for
Tavily web search, URL content fetching, and file I/O. Includes
specialized agents for meeting prep and social content generation.
"""

from .agent import Agent, AgentResult
from .meeting_prep_agent import (
    BriefingResult,
    MeetingPrepAgent,
    run_pipeline as run_meeting_prep_pipeline,
)
from .social_content_agent import ContentResult, SocialContentAgent
from .tools import ALL_TOOLS, execute_tool, fetch_url, read_file, tavily_search, write_file

__all__ = [
    "Agent",
    "AgentResult",
    "ALL_TOOLS",
    "BriefingResult",
    "ContentResult",
    "MeetingPrepAgent",
    "SocialContentAgent",
    "execute_tool",
    "fetch_url",
    "read_file",
    "run_meeting_prep_pipeline",
    "tavily_search",
    "write_file",
]
