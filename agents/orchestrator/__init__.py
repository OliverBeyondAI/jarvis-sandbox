"""
Orchestrator — Chains the three-agent pipeline end-to-end.

Pipeline: Trend Research → Synthesis → Memo Generation

Usage:
    python -m agents.orchestrator "multimodal AI agents"
    python -m agents.orchestrator --topic "agentic healthcare AI" --output ./output
"""

from .pipeline import Orchestrator, PipelineResult, run_pipeline

__all__ = ["Orchestrator", "PipelineResult", "run_pipeline"]
