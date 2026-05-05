"""
Data models for the Trend Research system.

Uses Pydantic for validation and serialization of all data structures
flowing through the multi-agent pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TrendCategory(str, Enum):
    """High-level categories for trend classification."""

    AI_ML = "ai_ml"
    BIOTECH = "biotech"
    CLIMATE = "climate"
    COMPUTING = "computing"
    CYBERSECURITY = "cybersecurity"
    ENERGY = "energy"
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    ROBOTICS = "robotics"
    SPACE = "space"
    OTHER = "other"


class Source(BaseModel):
    """A web source referenced in trend research."""

    title: str
    url: str
    snippet: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Trend(BaseModel):
    """A single identified trend with supporting evidence."""

    name: str
    category: TrendCategory = TrendCategory.OTHER
    summary: str = ""
    significance: str = ""  # Why this trend matters
    timeline: str = ""  # Expected timeline for impact
    key_players: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class TrendAnalysis(BaseModel):
    """Deep analysis of a trend, produced by the analysis agent."""

    trend_name: str
    current_state: str = ""
    opportunities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    related_trends: list[str] = Field(default_factory=list)
    raw_research: str = ""  # Full text from research phase


class ResearchReport(BaseModel):
    """Complete research report aggregating trends and analyses."""

    title: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    domain: str = ""  # e.g. "AI and Machine Learning Q2 2026"
    executive_summary: str = ""
    trends: list[Trend] = Field(default_factory=list)
    analyses: list[TrendAnalysis] = Field(default_factory=list)
    methodology: str = "Multi-agent research pipeline using Tavily web search and Claude analysis"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_storage_key(self) -> str:
        """Generate a storage key based on domain and timestamp."""
        safe_domain = self.domain.lower().replace(" ", "_").replace("/", "-")[:50]
        ts = self.generated_at.strftime("%Y%m%d_%H%M%S")
        return f"{safe_domain}/{ts}/report.json"
