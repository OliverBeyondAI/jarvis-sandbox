"""
Data Models — Pydantic models for research briefs, findings, and reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResearchPriority(str, Enum):
    """Priority level for a research brief."""

    URGENT = "urgent"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"


class FindingType(str, Enum):
    """Classification of a research finding."""

    FACT = "fact"
    TREND = "trend"
    RISK = "risk"
    OPPORTUNITY = "opportunity"
    COMPETITOR_MOVE = "competitor_move"
    REGULATORY = "regulatory"
    OPINION = "opinion"


# ---------------------------------------------------------------------------
# Source & Finding
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A web source referenced in research."""

    title: str
    url: str
    snippet: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Finding(BaseModel):
    """A single research finding with provenance."""

    headline: str
    detail: str = ""
    finding_type: FindingType = FindingType.FACT
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    sources: list[Source] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Research Brief (input)
# ---------------------------------------------------------------------------


class ResearchQuestion(BaseModel):
    """A specific question within a research brief."""

    question: str
    context: str = ""


class ResearchBrief(BaseModel):
    """The input specification for a research task."""

    topic: str
    questions: list[ResearchQuestion] = Field(default_factory=list)
    priority: ResearchPriority = ResearchPriority.STANDARD
    background: str = ""
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Executive Report (output)
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    """A section within the executive report."""

    title: str
    content: str
    findings: list[Finding] = Field(default_factory=list)


class ExecutiveReport(BaseModel):
    """Complete executive research report — the final output."""

    title: str
    topic: str
    executive_summary: str = ""
    key_takeaways: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    all_sources: list[Source] = Field(default_factory=list)
    methodology: str = (
        "Multi-step autonomous research using Tavily web search + Claude Opus analysis"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the report as a polished markdown document."""
        lines: list[str] = []

        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(
            f"*Generated: {self.generated_at.strftime('%B %d, %Y at %H:%M UTC')}*"
        )
        lines.append("")

        # Executive Summary
        if self.executive_summary:
            lines.append("---")
            lines.append("")
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(self.executive_summary)
            lines.append("")

        # Key Takeaways
        if self.key_takeaways:
            lines.append("---")
            lines.append("")
            lines.append("## Key Takeaways")
            lines.append("")
            for i, takeaway in enumerate(self.key_takeaways, 1):
                lines.append(f"{i}. {takeaway}")
            lines.append("")

        # Sections
        for section in self.sections:
            lines.append("---")
            lines.append("")
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

            if section.findings:
                for finding in section.findings:
                    icon = {
                        FindingType.FACT: "**Fact:**",
                        FindingType.TREND: "**Trend:**",
                        FindingType.RISK: "**Risk:**",
                        FindingType.OPPORTUNITY: "**Opportunity:**",
                        FindingType.COMPETITOR_MOVE: "**Competitor Move:**",
                        FindingType.REGULATORY: "**Regulatory:**",
                        FindingType.OPINION: "**Opinion:**",
                    }.get(finding.finding_type, "**Finding:**")

                    lines.append(f"> {icon} {finding.headline}")
                    if finding.detail:
                        lines.append(f"> {finding.detail}")
                    lines.append("")

        # Sources
        if self.all_sources:
            lines.append("---")
            lines.append("")
            lines.append("## Sources")
            lines.append("")
            seen_urls: set[str] = set()
            for source in self.all_sources:
                if source.url not in seen_urls:
                    seen_urls.add(source.url)
                    lines.append(f"- [{source.title}]({source.url})")
            lines.append("")

        # Methodology
        lines.append("---")
        lines.append("")
        lines.append(f"*Methodology: {self.methodology}*")
        lines.append("")

        return "\n".join(lines)
