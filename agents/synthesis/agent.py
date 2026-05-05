"""
Synthesis Agent — Takes research findings from the Trend Research Agent
and maps them to actionable product applications for OphthoFlow and Xena.

This agent analyzes each trend for clinical/workflow relevance, identifies
concrete feature opportunities, and produces a prioritized synthesis report.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import anthropic

from ..trend_research.models import ResearchReport
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


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYNTHESIS_AGENT_SYSTEM = """You are a strategic technology analyst specializing in healthcare AI applications.

Your task: Given research findings about emerging technology trends, analyze each trend and identify concrete applications for two healthcare platforms:

## Platform 1: OphthoFlow (Ophthalmology Workflow)
OphthoFlow automates prior authorization, clinical documentation, and practice management for eye care providers.
Key capabilities:
- Prior authorization submission and tracking (CPT/ICD-10 coding)
- Payer portal integration (Aetna, UnitedHealthcare, Cigna, BCBS, Medicare)
- Clinical note parsing and structured data extraction
- Procedure coding for 12+ ophthalmology procedures (anti-VEGF injections, cataract surgery, vitrectomy, laser treatments)
- Step therapy documentation and appeals workflow

## Platform 2: Xena (Clinical Platform)
Xena provides care coordination, patient engagement, and clinical decision support across healthcare specialties.
Key capabilities:
- Clinical workflow orchestration and task routing
- Patient engagement and communication
- Clinical decision support with AI-assisted recommendations
- Cross-specialty data integration and interoperability
- Care team collaboration and handoff management

## Analysis Framework

For each trend, evaluate:
1. **Relevance**: How directly does this trend apply to each platform's domain?
2. **Maturity**: Is the technology ready for production, or still experimental?
3. **Competitive Landscape**: Who else is applying this? What's the differentiation opportunity?
4. **Applications**: Concrete feature ideas with fit level, impact, and effort estimates.
5. **Synergies**: Can applying this trend to both platforms create compound value?

## Output Requirements

Provide your complete analysis as a JSON object with this structure:

```json
{
  "executive_summary": "2-3 sentence overview of findings",
  "trend_syntheses": [
    {
      "trend_name": "Name of the trend",
      "relevance_summary": "Why this trend matters for our platforms",
      "maturity_assessment": "How production-ready is this",
      "competitive_landscape": "Market context and differentiation",
      "applications": [
        {
          "title": "Feature/capability name",
          "description": "What it does",
          "platform": "ophthoflow|xena",
          "fit_level": "high|medium|low|none",
          "impact": "transformative|high|moderate|incremental",
          "effort": "low|medium|high",
          "use_case": "Specific clinical/workflow scenario",
          "user_benefit": "How end users benefit",
          "technical_approach": "High-level implementation approach",
          "dependencies": ["Dep 1"],
          "risks": ["Risk 1"]
        }
      ],
      "cross_platform_synergies": ["Synergy 1"],
      "watch_items": ["Item to monitor"],
      "overall_priority": "high|medium|low|none"
    }
  ],
  "strategic_themes": [
    {
      "name": "Theme name",
      "description": "What this theme is about",
      "contributing_trends": ["Trend 1", "Trend 2"],
      "strategic_implications": ["Implication 1"]
    }
  ],
  "top_opportunities": [...],
  "quick_wins": [...],
  "moonshots": [...],
  "key_risks": ["Risk 1"],
  "recommended_next_steps": ["Step 1"]
}
```

Be specific and actionable. Every application idea should be something a product team could evaluate and potentially build. Prioritize clinical value and workflow efficiency over technical novelty."""


# ---------------------------------------------------------------------------
# Agent Class
# ---------------------------------------------------------------------------

class SynthesisAgent:
    """
    Agentic synthesis loop that uses Claude to analyze research findings
    and map them to product applications for OphthoFlow and Xena.
    """

    def __init__(self, config: SynthesisConfig | None = None) -> None:
        self.config = config or SynthesisConfig.from_env()
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        return self._client

    async def synthesize(self, research_report: ResearchReport) -> dict[str, Any]:
        """
        Analyze research findings and produce structured synthesis.

        Args:
            research_report: The ResearchReport from Agent 1 (Trend Research).

        Returns:
            Dictionary with structured synthesis findings.
        """
        print(f"\n[synthesis] Analyzing report: {research_report.title}", file=sys.stderr)
        print(f"[synthesis] Trends to analyze: {len(research_report.trends)}", file=sys.stderr)

        # Serialize the research report for the prompt
        report_json = json.dumps(
            research_report.model_dump(mode="json"),
            indent=2,
            default=str,
        )

        # Build the user message with the full research context
        user_message = (
            f"## Research Report to Analyze\n\n"
            f"The following research report was produced by our Trend Research Agent. "
            f"Analyze each trend and identify concrete applications for OphthoFlow "
            f"and Xena.\n\n"
            f"```json\n{report_json}\n```\n\n"
            f"## Instructions\n\n"
            f"1. Analyze each trend in the report for relevance to OphthoFlow and Xena.\n"
            f"2. For each relevant trend, propose specific product features or capabilities.\n"
            f"3. Score each application idea by fit, impact, and effort.\n"
            f"4. Identify cross-cutting strategic themes.\n"
            f"5. Prioritize: separate quick wins from moonshots.\n"
            f"6. Provide your complete analysis in the JSON format from your instructions."
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]

        # Agent loop (synthesis is typically single-turn but we support multi-turn)
        turn = 0
        max_turns = self.config.max_agent_turns

        while turn < max_turns:
            turn += 1
            print(f"[synthesis] Turn {turn}/{max_turns}", file=sys.stderr)

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=SYNTHESIS_AGENT_SYSTEM,
                messages=messages,
            )

            # Extract text response
            text_blocks = [b.text for b in response.content if b.type == "text"]
            final_text = "\n".join(text_blocks)

            if response.stop_reason == "end_turn":
                return self._parse_synthesis(final_text, research_report.title)

            # If the model wants to continue (unlikely without tools), let it
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": "Please continue and complete your analysis in the JSON format specified.",
            })

        print(f"[synthesis] Max turns ({max_turns}) reached", file=sys.stderr)
        return {"error": "Max agent turns reached", "research_source": research_report.title}

    def _parse_synthesis(self, text: str, report_title: str) -> dict[str, Any]:
        """Parse the agent's final text response into structured synthesis."""
        json_match = None

        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            json_match = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            candidate = text[start:end].strip()
            if candidate.startswith("{"):
                json_match = candidate
        elif text.strip().startswith("{"):
            json_match = text.strip()

        if json_match:
            try:
                findings = json.loads(json_match)
                findings.setdefault("research_source", report_title)
                return findings
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text
        return {
            "research_source": report_title,
            "executive_summary": text[:500],
            "raw_analysis": text,
            "trend_syntheses": [],
        }

    async def synthesize_to_report(self, research_report: ResearchReport) -> SynthesisReport:
        """
        Full pipeline: analyze research and produce a validated SynthesisReport.

        Args:
            research_report: The ResearchReport from Agent 1.

        Returns:
            A validated SynthesisReport with applications, themes, and priorities.
        """
        findings = await self.synthesize(research_report)

        # Build TrendSynthesis objects
        trend_syntheses = []
        for ts_data in findings.get("trend_syntheses", []):
            applications = [
                ApplicationIdea(
                    title=app.get("title", ""),
                    description=app.get("description", ""),
                    platform=_parse_platform(app.get("platform", "xena")),
                    fit_level=_parse_enum(FitLevel, app.get("fit_level", "medium")),
                    impact=_parse_enum(ImpactLevel, app.get("impact", "moderate")),
                    effort=_parse_enum(EffortLevel, app.get("effort", "medium")),
                    use_case=app.get("use_case", ""),
                    user_benefit=app.get("user_benefit", ""),
                    technical_approach=app.get("technical_approach", ""),
                    dependencies=app.get("dependencies", []),
                    risks=app.get("risks", []),
                )
                for app in ts_data.get("applications", [])
            ]
            trend_syntheses.append(
                TrendSynthesis(
                    trend_name=ts_data.get("trend_name", ""),
                    relevance_summary=ts_data.get("relevance_summary", ""),
                    maturity_assessment=ts_data.get("maturity_assessment", ""),
                    competitive_landscape=ts_data.get("competitive_landscape", ""),
                    applications=applications,
                    cross_platform_synergies=ts_data.get("cross_platform_synergies", []),
                    watch_items=ts_data.get("watch_items", []),
                    overall_priority=_parse_enum(FitLevel, ts_data.get("overall_priority", "medium")),
                )
            )

        # Build StrategicTheme objects
        strategic_themes = [
            StrategicTheme(
                name=t.get("name", ""),
                description=t.get("description", ""),
                contributing_trends=t.get("contributing_trends", []),
                strategic_implications=t.get("strategic_implications", []),
            )
            for t in findings.get("strategic_themes", [])
        ]

        # Build prioritized application lists
        top_opportunities = _parse_application_list(findings.get("top_opportunities", []))
        quick_wins = _parse_application_list(findings.get("quick_wins", []))
        moonshots = _parse_application_list(findings.get("moonshots", []))

        report = SynthesisReport(
            title=f"Synthesis: {research_report.title}",
            research_source=research_report.title,
            executive_summary=findings.get("executive_summary", ""),
            trend_syntheses=trend_syntheses,
            strategic_themes=strategic_themes,
            top_opportunities=top_opportunities,
            quick_wins=quick_wins,
            moonshots=moonshots,
            key_risks=findings.get("key_risks", []),
            recommended_next_steps=findings.get("recommended_next_steps", []),
            metadata={
                "input_trends": len(research_report.trends),
                "input_analyses": len(research_report.analyses),
                "applications_generated": sum(
                    len(ts.applications) for ts in trend_syntheses
                ),
            },
        )

        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_enum(enum_cls: type, value: str) -> Any:
    """Safely parse a string to an enum value with fallback."""
    try:
        return enum_cls(value.lower())
    except (ValueError, AttributeError):
        return list(enum_cls)[1]  # Return second value as default (usually "medium"/"moderate")


def _parse_platform(value: str) -> Platform:
    """Parse platform string to Platform enum."""
    try:
        return Platform(value.lower())
    except (ValueError, AttributeError):
        return Platform.XENA


def _parse_application_list(items: list[dict[str, Any]]) -> list[ApplicationIdea]:
    """Parse a list of application idea dictionaries."""
    return [
        ApplicationIdea(
            title=app.get("title", ""),
            description=app.get("description", ""),
            platform=_parse_platform(app.get("platform", "xena")),
            fit_level=_parse_enum(FitLevel, app.get("fit_level", "medium")),
            impact=_parse_enum(ImpactLevel, app.get("impact", "moderate")),
            effort=_parse_enum(EffortLevel, app.get("effort", "medium")),
            use_case=app.get("use_case", ""),
            user_benefit=app.get("user_benefit", ""),
            technical_approach=app.get("technical_approach", ""),
            dependencies=app.get("dependencies", []),
            risks=app.get("risks", []),
        )
        for app in items
    ]


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

async def run_synthesis(
    research_report: ResearchReport,
    config: SynthesisConfig | None = None,
) -> dict[str, Any]:
    """
    Convenience function to synthesize a research report.

    Args:
        research_report: The ResearchReport from Agent 1.
        config: Optional config override.

    Returns:
        Structured synthesis findings dictionary.
    """
    agent = SynthesisAgent(config)
    return await agent.synthesize(research_report)


async def run_full_synthesis(
    research_report: ResearchReport,
    config: SynthesisConfig | None = None,
) -> SynthesisReport:
    """
    Convenience function to produce a full SynthesisReport.

    Args:
        research_report: The ResearchReport from Agent 1.
        config: Optional config override.

    Returns:
        Complete SynthesisReport object.
    """
    agent = SynthesisAgent(config)
    return await agent.synthesize_to_report(research_report)
