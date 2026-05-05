"""
Synthesis Agent — Maps research findings to product applications.

Takes structured research from the Trend Research Agent (Agent 1) and
identifies actionable applications for OphthoFlow (ophthalmology workflow)
and Xena (clinical platform).
"""

from .agent import SynthesisAgent, run_full_synthesis, run_synthesis
from .config import SynthesisConfig
from .models import (
    ApplicationIdea,
    EffortLevel,
    FitLevel,
    ImpactLevel,
    Platform,
    StrategicTheme,
    SynthesisReport,
    TrendSynthesis,
)

__all__ = [
    "ApplicationIdea",
    "EffortLevel",
    "FitLevel",
    "ImpactLevel",
    "Platform",
    "StrategicTheme",
    "SynthesisAgent",
    "SynthesisConfig",
    "SynthesisReport",
    "TrendSynthesis",
    "run_full_synthesis",
    "run_synthesis",
]
