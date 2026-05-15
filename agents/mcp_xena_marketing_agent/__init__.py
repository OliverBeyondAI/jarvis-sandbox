"""
MCP Xena Marketing Agent — Autonomous marketing campaign generator.

Combines:
  - Claude Agent SDK for the agentic loop
  - MCP servers for tool integration (research, analysis, content)
  - OpenAI SDK for image generation (ad creatives, social visuals)
  - Tavily API for market research and competitor intelligence
"""

from .agent import MCPXenaMarketingAgent
from .config import Config, ProductInfo
from .models import Campaign, MarketingContent, MarketResearch

__all__ = [
    "MCPXenaMarketingAgent",
    "Config",
    "ProductInfo",
    "Campaign",
    "MarketingContent",
    "MarketResearch",
]
