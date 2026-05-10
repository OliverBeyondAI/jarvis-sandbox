"""
Data Models — Pydantic models for queries, findings, synthesis, and reports.

Covers the full research lifecycle:
  Query → SubQuery → SearchResult → Finding → Synthesis → Report
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FindingType(str, Enum):
    """Classification of a research finding."""

    FACT = "fact"
    TREND = "trend"
    RISK = "risk"
    OPPORTUNITY = "opportunity"
    INSIGHT = "insight"
    CONTRADICTION = "contradiction"
    DATA_POINT = "data_point"


class ConfidenceLevel(str, Enum):
    """Qualitative confidence in a finding."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


class InsightPriority(str, Enum):
    """Priority level for actionable insights."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Source & Evidence
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


class Evidence(BaseModel):
    """A piece of evidence supporting a finding."""

    claim: str
    source: Source
    verbatim_quote: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


# ---------------------------------------------------------------------------
# Query Decomposition
# ---------------------------------------------------------------------------


class SubQuery(BaseModel):
    """A decomposed sub-question from the main research query."""

    question: str
    rationale: str = ""
    search_terms: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=5)


class QueryPlan(BaseModel):
    """The decomposition of a complex query into actionable sub-queries."""

    original_query: str
    sub_queries: list[SubQuery] = Field(default_factory=list)
    research_angles: list[str] = Field(default_factory=list)
    estimated_searches: int = 0


# ---------------------------------------------------------------------------
# Findings & Synthesis
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A single research finding with provenance and confidence."""

    headline: str
    detail: str = ""
    finding_type: FindingType = FindingType.FACT
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    evidence: list[Evidence] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    """Cross-cutting synthesis across multiple findings."""

    theme: str
    summary: str
    supporting_findings: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    implications: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Actionable Insights
# ---------------------------------------------------------------------------


class ActionableInsight(BaseModel):
    """A concrete, actionable recommendation derived from research."""

    title: str
    description: str
    priority: InsightPriority = InsightPriority.MEDIUM
    rationale: str = ""
    next_steps: list[str] = Field(default_factory=list)
    risks_if_ignored: str = ""
    timeline: str = ""


# ---------------------------------------------------------------------------
# Research Report (final output)
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    """A section within the research report."""

    title: str
    content: str
    findings: list[Finding] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Complete autonomous research report — the final structured output."""

    title: str
    query: str
    executive_summary: str = ""
    key_takeaways: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    actionable_insights: list[ActionableInsight] = Field(default_factory=list)
    syntheses: list[Synthesis] = Field(default_factory=list)
    all_sources: list[Source] = Field(default_factory=list)
    methodology: str = (
        "Multi-step autonomous research using Tavily web search + Claude analysis "
        "with query decomposition, parallel search, finding synthesis, and insight generation"
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
        lines.append(f"*Query: {self.query}*")
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

        # Main Sections
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
                        FindingType.FACT: "Fact",
                        FindingType.TREND: "Trend",
                        FindingType.RISK: "Risk",
                        FindingType.OPPORTUNITY: "Opportunity",
                        FindingType.INSIGHT: "Insight",
                        FindingType.CONTRADICTION: "Contradiction",
                        FindingType.DATA_POINT: "Data Point",
                    }.get(finding.finding_type, "Finding")
                    conf = finding.confidence.value.capitalize()

                    lines.append(f"> **{icon}** ({conf} confidence): {finding.headline}")
                    if finding.detail:
                        lines.append(f"> {finding.detail}")
                    lines.append("")

        # Syntheses
        if self.syntheses:
            lines.append("---")
            lines.append("")
            lines.append("## Cross-Cutting Themes")
            lines.append("")
            for syn in self.syntheses:
                lines.append(f"### {syn.theme}")
                lines.append("")
                lines.append(syn.summary)
                lines.append("")
                if syn.implications:
                    lines.append("**Implications:**")
                    for imp in syn.implications:
                        lines.append(f"- {imp}")
                    lines.append("")
                if syn.contradictions:
                    lines.append("**Contradictions to note:**")
                    for c in syn.contradictions:
                        lines.append(f"- {c}")
                    lines.append("")

        # Actionable Insights
        if self.actionable_insights:
            lines.append("---")
            lines.append("")
            lines.append("## Actionable Insights & Recommendations")
            lines.append("")
            for i, insight in enumerate(self.actionable_insights, 1):
                priority_badge = {
                    InsightPriority.CRITICAL: "CRITICAL",
                    InsightPriority.HIGH: "HIGH",
                    InsightPriority.MEDIUM: "MEDIUM",
                    InsightPriority.LOW: "LOW",
                }.get(insight.priority, "")
                lines.append(f"### {i}. {insight.title} [{priority_badge}]")
                lines.append("")
                lines.append(insight.description)
                lines.append("")
                if insight.rationale:
                    lines.append(f"**Rationale:** {insight.rationale}")
                    lines.append("")
                if insight.next_steps:
                    lines.append("**Next Steps:**")
                    for step in insight.next_steps:
                        lines.append(f"- {step}")
                    lines.append("")
                if insight.risks_if_ignored:
                    lines.append(f"**Risk if ignored:** {insight.risks_if_ignored}")
                    lines.append("")
                if insight.timeline:
                    lines.append(f"**Timeline:** {insight.timeline}")
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

        # Metadata
        if self.metadata:
            lines.append("")
            lines.append(
                f"*Searches: {self.metadata.get('search_count', 'N/A')} | "
                f"Pages read: {self.metadata.get('fetch_count', 'N/A')} | "
                f"Duration: {self.metadata.get('duration_seconds', 'N/A')}s*"
            )

        lines.append("")

        return "\n".join(lines)
