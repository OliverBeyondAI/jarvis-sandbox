"""
Trend Research — Multi-agent system for researching emerging trends.

Provides shared utilities (S3 storage, Tavily client, configuration)
and specialized agents for discovering, analyzing, and reporting on
trends across technology, science, and industry domains.
"""

from .agent import ResearchAgent, run_full_pipeline, run_research
from .config import Config
from .models import (
    ResearchReport,
    Source,
    Trend,
    TrendAnalysis,
    TrendCategory,
)
from .s3_storage import S3Storage
from .tavily_client import TavilyResearchClient
from .tools import ALL_TOOLS, execute_tool

__all__ = [
    "ALL_TOOLS",
    "Config",
    "ResearchAgent",
    "ResearchReport",
    "S3Storage",
    "Source",
    "TavilyResearchClient",
    "Trend",
    "TrendAnalysis",
    "TrendCategory",
    "execute_tool",
    "run_full_pipeline",
    "run_research",
]
