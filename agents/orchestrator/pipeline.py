"""
Orchestrator Pipeline — Coordinates the three-agent research pipeline.

Stages:
  1. Trend Research Agent: Takes an AI trend topic → produces ResearchReport
  2. Synthesis Agent: Takes ResearchReport → produces SynthesisReport
  3. Memo Generation Agent: Takes both reports → produces formatted memo + HTML

Each stage is independent and can also be run standalone. The orchestrator
manages data flow, error handling, and progress reporting between stages.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..memo_generation.agent import MemoGenerationAgent
from ..memo_generation.config import MemoConfig
from ..memo_generation.models import ArtifactBundle, MemoAudience
from ..synthesis.agent import SynthesisAgent
from ..synthesis.config import SynthesisConfig
from ..synthesis.models import SynthesisReport
from ..trend_research.agent import ResearchAgent
from ..trend_research.config import Config as ResearchConfig
from ..trend_research.models import ResearchReport


# ---------------------------------------------------------------------------
# Pipeline Result
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Result from a single pipeline stage."""

    name: str
    success: bool
    duration_seconds: float
    error: str | None = None


@dataclass
class PipelineResult:
    """Complete result from the orchestrator pipeline."""

    topic: str
    success: bool
    stages: list[StageResult] = field(default_factory=list)
    research_report: ResearchReport | None = None
    synthesis_report: SynthesisReport | None = None
    artifact_bundle: ArtifactBundle | None = None
    total_duration_seconds: float = 0.0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        """Human-readable summary of the pipeline run."""
        lines = [
            f"Pipeline: {'SUCCESS' if self.success else 'FAILED'}",
            f"Topic: {self.topic}",
            f"Duration: {self.total_duration_seconds:.1f}s",
            "",
            "Stages:",
        ]
        for stage in self.stages:
            status = "OK" if stage.success else "FAIL"
            lines.append(f"  [{status}] {stage.name} ({stage.duration_seconds:.1f}s)")
            if stage.error:
                lines.append(f"         Error: {stage.error}")

        if self.artifact_bundle:
            lines.append("")
            lines.append("Outputs:")
            lines.append(f"  Research: {self.artifact_bundle.research_report_path}")
            lines.append(f"  Synthesis: {self.artifact_bundle.synthesis_report_path}")
            lines.append(f"  Memo (MD): {self.artifact_bundle.memo_path}")
            lines.append(f"  Memo (HTML): {self.artifact_bundle.memo_html_path}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Coordinates the three-agent pipeline: Research → Synthesis → Memo.

    Each stage runs sequentially, passing structured data to the next.
    The orchestrator handles configuration, progress reporting, and error
    recovery between stages.
    """

    def __init__(
        self,
        research_config: ResearchConfig | None = None,
        synthesis_config: SynthesisConfig | None = None,
        memo_config: MemoConfig | None = None,
        output_dir: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.research_config = research_config or ResearchConfig.from_env()
        self.synthesis_config = synthesis_config or SynthesisConfig.from_env()
        self.memo_config = memo_config or MemoConfig.from_env()
        self.verbose = verbose

        # Override output dir if specified
        if output_dir:
            object.__setattr__(self.memo_config, "local_storage_dir", output_dir)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[orchestrator] {msg}", file=sys.stderr)

    def validate(self) -> list[str]:
        """Validate all configurations and return warnings."""
        warnings: list[str] = []
        warnings.extend(self.research_config.validate())
        warnings.extend(self.synthesis_config.validate())
        warnings.extend(self.memo_config.validate())
        # Deduplicate (e.g. ANTHROPIC_API_KEY warning appears 3x)
        return list(dict.fromkeys(warnings))

    async def run(
        self,
        topic: str,
        audience: MemoAudience = MemoAudience.PRODUCT,
    ) -> PipelineResult:
        """
        Execute the full three-stage pipeline.

        Args:
            topic: The AI trend topic to research (e.g. "multimodal AI agents").
            audience: Target audience for the final memo.

        Returns:
            PipelineResult with all outputs and stage metadata.
        """
        pipeline_start = time.time()
        result = PipelineResult(topic=topic, success=False)

        self._log(f"Starting pipeline for topic: {topic}")
        self._log(f"Target audience: {audience.value}")
        self._log("")

        # --- Stage 1: Trend Research ---
        self._log("=" * 50)
        self._log("STAGE 1: Trend Research")
        self._log("=" * 50)

        stage1_start = time.time()
        try:
            research_agent = ResearchAgent(self.research_config)
            research_report = await research_agent.research_and_build_report(topic)
            result.research_report = research_report

            self._log(f"Research complete: {research_report.title}")
            self._log(f"  Trends found: {len(research_report.trends)}")
            self._log(f"  Analyses: {len(research_report.analyses)}")

            result.stages.append(StageResult(
                name="Trend Research",
                success=True,
                duration_seconds=time.time() - stage1_start,
            ))
        except Exception as e:
            result.stages.append(StageResult(
                name="Trend Research",
                success=False,
                duration_seconds=time.time() - stage1_start,
                error=f"{type(e).__name__}: {e}",
            ))
            self._log(f"Stage 1 FAILED: {e}")
            result.total_duration_seconds = time.time() - pipeline_start
            return result

        # --- Stage 2: Synthesis ---
        self._log("")
        self._log("=" * 50)
        self._log("STAGE 2: Synthesis")
        self._log("=" * 50)

        stage2_start = time.time()
        try:
            synthesis_agent = SynthesisAgent(self.synthesis_config)
            synthesis_report = await synthesis_agent.synthesize_to_report(research_report)
            result.synthesis_report = synthesis_report

            self._log(f"Synthesis complete: {synthesis_report.title}")
            self._log(f"  Trend syntheses: {len(synthesis_report.trend_syntheses)}")
            self._log(f"  Strategic themes: {len(synthesis_report.strategic_themes)}")
            self._log(f"  Top opportunities: {len(synthesis_report.top_opportunities)}")

            result.stages.append(StageResult(
                name="Synthesis",
                success=True,
                duration_seconds=time.time() - stage2_start,
            ))
        except Exception as e:
            result.stages.append(StageResult(
                name="Synthesis",
                success=False,
                duration_seconds=time.time() - stage2_start,
                error=f"{type(e).__name__}: {e}",
            ))
            self._log(f"Stage 2 FAILED: {e}")
            result.total_duration_seconds = time.time() - pipeline_start
            return result

        # --- Stage 3: Memo Generation ---
        self._log("")
        self._log("=" * 50)
        self._log("STAGE 3: Memo Generation")
        self._log("=" * 50)

        stage3_start = time.time()
        try:
            memo_agent = MemoGenerationAgent(self.memo_config)
            artifact_bundle = await memo_agent.run_full_pipeline(
                research_report=research_report,
                synthesis_report=synthesis_report,
                audience=audience,
            )
            result.artifact_bundle = artifact_bundle

            self._log(f"Memo generation complete!")
            self._log(f"  Bundle ID: {artifact_bundle.bundle_id}")
            self._log(f"  Memo (MD): {artifact_bundle.memo_path}")
            self._log(f"  Memo (HTML): {artifact_bundle.memo_html_path}")

            result.stages.append(StageResult(
                name="Memo Generation",
                success=True,
                duration_seconds=time.time() - stage3_start,
            ))
        except Exception as e:
            result.stages.append(StageResult(
                name="Memo Generation",
                success=False,
                duration_seconds=time.time() - stage3_start,
                error=f"{type(e).__name__}: {e}",
            ))
            self._log(f"Stage 3 FAILED: {e}")
            result.total_duration_seconds = time.time() - pipeline_start
            return result

        # --- Pipeline Complete ---
        result.success = True
        result.total_duration_seconds = time.time() - pipeline_start

        self._log("")
        self._log("=" * 50)
        self._log("PIPELINE COMPLETE")
        self._log("=" * 50)
        self._log(f"Total duration: {result.total_duration_seconds:.1f}s")

        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


async def run_pipeline(
    topic: str,
    output_dir: str | None = None,
    audience: MemoAudience = MemoAudience.PRODUCT,
    verbose: bool = True,
) -> PipelineResult:
    """
    Run the full three-agent pipeline for a given AI trend topic.

    Args:
        topic: The AI trend topic to research.
        output_dir: Directory for output files (default: ./pipeline_output).
        audience: Target audience for the memo.
        verbose: Print progress to stderr.

    Returns:
        PipelineResult with all outputs and metadata.
    """
    orchestrator = Orchestrator(output_dir=output_dir or "./pipeline_output", verbose=verbose)
    return await orchestrator.run(topic, audience=audience)
