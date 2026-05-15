"""
Data Models — Pydantic models for the MCP Xena Marketing Agent.

Covers the full marketing content lifecycle:
  MarketInsight → CompetitorAnalysis → MessagingStrategy →
  MarketingContent → GeneratedImage → Campaign
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ContentChannel(str, Enum):
    """Marketing channel for content distribution."""

    LANDING_PAGE = "landing_page"
    EMAIL_SEQUENCE = "email_sequence"
    SOCIAL_MEDIA = "social_media"
    BLOG_POST = "blog_post"
    AD_COPY = "ad_copy"
    PRESS_RELEASE = "press_release"
    PRODUCT_DESCRIPTION = "product_description"


class InsightType(str, Enum):
    """Classification for market research insights."""

    MARKET_TREND = "market_trend"
    COMPETITOR_INTEL = "competitor_intel"
    AUDIENCE_INSIGHT = "audience_insight"
    MESSAGING_OPPORTUNITY = "messaging_opportunity"
    POSITIONING_GAP = "positioning_gap"
    RISK = "risk"


class AnalysisPhase(str, Enum):
    """Phase of market analysis."""

    MARKET_LANDSCAPE = "market_landscape"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    AUDIENCE_RESEARCH = "audience_research"
    MESSAGING_STRATEGY = "messaging_strategy"
    CONTENT_PLANNING = "content_planning"


# ---------------------------------------------------------------------------
# Research Models
# ---------------------------------------------------------------------------


class MarketInsight(BaseModel):
    """A single insight from market research."""

    headline: str
    detail: str = ""
    insight_type: InsightType = InsightType.MARKET_TREND
    source_urls: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CompetitorAnalysis(BaseModel):
    """Analysis of a single competitor."""

    name: str
    positioning: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    messaging_themes: list[str] = Field(default_factory=list)
    key_urls: list[str] = Field(default_factory=list)


class AudienceSegment(BaseModel):
    """A target audience segment with persona details."""

    name: str
    description: str = ""
    pain_points: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    preferred_channels: list[str] = Field(default_factory=list)
    messaging_angle: str = ""


class MarketResearch(BaseModel):
    """Aggregated market research findings from all phases."""

    insights: list[MarketInsight] = Field(default_factory=list)
    competitors: list[CompetitorAnalysis] = Field(default_factory=list)
    audience_segments: list[AudienceSegment] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Content Models
# ---------------------------------------------------------------------------


class MessagingStrategy(BaseModel):
    """Core messaging strategy derived from research."""

    positioning_statement: str = ""
    key_messages: list[str] = Field(default_factory=list)
    value_propositions: list[str] = Field(default_factory=list)
    emotional_appeals: list[str] = Field(default_factory=list)
    rational_appeals: list[str] = Field(default_factory=list)


class MarketingContent(BaseModel):
    """A single piece of generated marketing content."""

    channel: ContentChannel
    title: str
    body: str
    cta: str = ""
    notes: str = ""


class GeneratedImage(BaseModel):
    """An AI-generated marketing image with embedded text and branding."""

    concept: str = ""
    headline: str = ""
    tagline: str = ""
    brand_name: str = ""
    platform: str = "instagram_post"
    style: str = "natural"
    color_palette: str = ""
    channel: str = ""
    description: str = ""
    file_path: str = ""
    resolved_prompt: str = ""
    model: str = "gpt-image-1"


# ---------------------------------------------------------------------------
# Campaign (top-level output)
# ---------------------------------------------------------------------------


class Campaign(BaseModel):
    """A complete marketing campaign — the final structured output."""

    name: str
    product_name: str
    objective: str = ""
    brand_voice_preset: str = "professional"
    research: MarketResearch = Field(default_factory=MarketResearch)
    messaging: MessagingStrategy = Field(default_factory=MessagingStrategy)
    content: list[MarketingContent] = Field(default_factory=list)
    images: list[GeneratedImage] = Field(default_factory=list)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the campaign as a polished markdown document."""
        lines: list[str] = [
            f"# {self.name}",
            "",
            f"*Generated: {self.generated_at.strftime('%B %d, %Y at %H:%M UTC')}*",
            f"*Product: {self.product_name}*",
        ]
        if self.objective:
            lines.append(f"*Objective: {self.objective}*")
        lines += ["", "---", ""]

        # Messaging strategy
        if self.messaging.positioning_statement:
            lines += [
                "## Messaging Strategy",
                "",
                f"**Positioning:** {self.messaging.positioning_statement}",
                "",
            ]
            if self.messaging.key_messages:
                lines.append("**Key Messages:**")
                for msg in self.messaging.key_messages:
                    lines.append(f"- {msg}")
                lines.append("")
            if self.messaging.value_propositions:
                lines.append("**Value Propositions:**")
                for vp in self.messaging.value_propositions:
                    lines.append(f"- {vp}")
                lines += ["", "---", ""]

        # Content pieces
        for piece in self.content:
            channel_label = piece.channel.value.replace("_", " ").title()
            lines += [
                f"## {channel_label}: {piece.title}",
                "",
                piece.body,
                "",
            ]
            if piece.cta:
                lines += [f"**CTA:** {piece.cta}", ""]
            lines += ["---", ""]

        # Images
        if self.images:
            lines += ["## Generated Images", ""]
            for img in self.images:
                label = img.platform or img.channel or "General"
                lines.append(f"### {label}")
                if img.headline:
                    lines.append(f"- **Headline:** {img.headline}")
                if img.tagline:
                    lines.append(f"- **Tagline:** {img.tagline}")
                if img.concept:
                    lines.append(f"- **Concept:** {img.concept}")
                if img.style:
                    lines.append(f"- **Style:** {img.style}")
                if img.description:
                    lines.append(f"- **Description:** {img.description}")
                if img.file_path:
                    lines.append(f"- **File:** `{img.file_path}`")
                lines.append("")
            lines += ["---", ""]

        # Metadata
        if self.metadata:
            lines.append(
                f"*Research queries: {self.metadata.get('search_count', 'N/A')} | "
                f"Content pieces: {len(self.content)} | "
                f"Images: {len(self.images)} | "
                f"Duration: {self.metadata.get('duration_seconds', 'N/A')}s*"
            )
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return self.model_dump_json(indent=2)
