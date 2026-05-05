"""
Data models for the Synthesis Agent.

Defines structured output types for analyzing research trends and mapping
them to potential applications in OphthoFlow and Xena platforms.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Platform(str, Enum):
    """Target product platforms for application mapping."""

    OPHTHOFLOW = "ophthoflow"
    XENA = "xena"


class FitLevel(str, Enum):
    """How well a trend maps to a product application."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class EffortLevel(str, Enum):
    """Estimated implementation effort."""

    LOW = "low"       # Days to a week
    MEDIUM = "medium"  # Weeks to a month
    HIGH = "high"      # Months+


class ImpactLevel(str, Enum):
    """Expected business/clinical impact if implemented."""

    TRANSFORMATIVE = "transformative"  # Game-changing capability
    HIGH = "high"                      # Major improvement
    MODERATE = "moderate"              # Meaningful improvement
    INCREMENTAL = "incremental"        # Nice-to-have


class ApplicationIdea(BaseModel):
    """A concrete application idea mapping a trend to a product feature."""

    title: str
    description: str = ""
    platform: Platform
    fit_level: FitLevel = FitLevel.MEDIUM
    impact: ImpactLevel = ImpactLevel.MODERATE
    effort: EffortLevel = EffortLevel.MEDIUM
    use_case: str = ""           # Specific clinical/workflow use case
    user_benefit: str = ""       # How end users benefit
    technical_approach: str = "" # High-level implementation approach
    dependencies: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class TrendSynthesis(BaseModel):
    """Analysis of a single trend's relevance to OphthoFlow and Xena."""

    trend_name: str
    relevance_summary: str = ""  # Why this trend matters for our products
    maturity_assessment: str = ""  # How mature/ready is this trend for adoption
    competitive_landscape: str = ""  # Who else is applying this, what's the moat
    applications: list[ApplicationIdea] = Field(default_factory=list)
    cross_platform_synergies: list[str] = Field(default_factory=list)  # Benefits from applying to both platforms
    watch_items: list[str] = Field(default_factory=list)  # Things to monitor
    overall_priority: FitLevel = FitLevel.MEDIUM


class StrategicTheme(BaseModel):
    """A cross-cutting theme identified across multiple trends."""

    name: str
    description: str = ""
    contributing_trends: list[str] = Field(default_factory=list)
    strategic_implications: list[str] = Field(default_factory=list)


class SynthesisReport(BaseModel):
    """Complete synthesis report mapping research findings to product applications."""

    title: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    research_source: str = ""  # Reference to the input ResearchReport
    executive_summary: str = ""

    # Platform context
    ophthoflow_context: str = (
        "OphthoFlow is an ophthalmology workflow platform that automates prior "
        "authorization, clinical documentation, and practice management for "
        "eye care providers. Key capabilities include PA submission, procedure "
        "coding (CPT/ICD-10), payer portal integration, and clinical note parsing."
    )
    xena_context: str = (
        "Xena is a clinical platform providing care coordination, patient "
        "engagement, and clinical decision support across healthcare specialties. "
        "It serves as a hub for clinical workflows, data integration, and "
        "AI-assisted care delivery."
    )

    # Core analysis
    trend_syntheses: list[TrendSynthesis] = Field(default_factory=list)
    strategic_themes: list[StrategicTheme] = Field(default_factory=list)

    # Prioritized recommendations
    top_opportunities: list[ApplicationIdea] = Field(default_factory=list)
    quick_wins: list[ApplicationIdea] = Field(default_factory=list)  # High fit, low effort
    moonshots: list[ApplicationIdea] = Field(default_factory=list)   # High impact, high effort

    # Risk and readiness
    key_risks: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)

    methodology: str = "AI-powered synthesis of research findings using Claude analysis agent"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_storage_key(self) -> str:
        """Generate a storage key for this report."""
        ts = self.generated_at.strftime("%Y%m%d_%H%M%S")
        return f"synthesis/{ts}/synthesis_report.json"
