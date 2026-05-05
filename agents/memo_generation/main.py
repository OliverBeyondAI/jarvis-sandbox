"""
CLI entry point for the Memo Generation Agent.

Usage:
    python -m agents.memo_generation --synthesis report.json
    python -m agents.memo_generation --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..synthesis.models import (
    ApplicationIdea,
    EffortLevel,
    FitLevel,
    ImpactLevel,
    Platform,
    StrategicTheme,
    SynthesisReport,
    TrendSynthesis,
)
from ..trend_research.models import (
    ResearchReport,
    Source,
    Trend,
    TrendAnalysis,
    TrendCategory,
)
from .agent import MemoGenerationAgent, run_memo_pipeline
from .config import MemoConfig
from .models import MemoAudience


def _build_demo_research_report() -> ResearchReport:
    """Build a demo ResearchReport for end-to-end testing."""
    return ResearchReport(
        title="Emerging Trends in AI and Healthcare — Q2 2026",
        domain="healthcare AI",
        executive_summary=(
            "Research identified five major trends reshaping healthcare AI: "
            "multimodal foundation models for clinical imaging, agentic AI "
            "for administrative workflows, federated learning for privacy-preserving "
            "model training, real-time clinical decision support, and AI-powered "
            "patient engagement platforms."
        ),
        trends=[
            Trend(
                name="Multimodal Foundation Models for Clinical Imaging",
                category=TrendCategory.AI_ML,
                summary="Large vision-language models achieving specialist-level ophthalmology diagnostics.",
                significance="Transforms diagnostic workflows with AI-assisted reads at scale.",
                timeline="12-18 months for production deployment",
                key_players=["Google Health", "Microsoft Nuance", "Retina-AI", "Eyenuk"],
                sources=[
                    Source(title="Foundation Models in Medical Imaging 2026", url="https://example.com/fm", snippet="95%+ accuracy on retinal disease classification", relevance_score=0.95),
                ],
                confidence=0.88,
                tags=["imaging", "diagnostics", "ophthalmology"],
            ),
            Trend(
                name="Agentic AI for Healthcare Administration",
                category=TrendCategory.AI_ML,
                summary="AI agents autonomously handling prior authorizations and claims processing.",
                significance="Administrative overhead consumes 30% of healthcare spending — agents can cut this by 40-60%.",
                timeline="6-12 months — some solutions already in production",
                key_players=["Olive AI", "Waystar", "Infinitus Health", "Cohere Health"],
                sources=[
                    Source(title="AI Agents Transform Prior Auth", url="https://example.com/ai-pa", snippet="PA turnaround from 14 days to under 24 hours", relevance_score=0.92),
                ],
                confidence=0.91,
                tags=["agents", "prior-auth", "RCM"],
            ),
            Trend(
                name="Real-Time Clinical Decision Support with LLMs",
                category=TrendCategory.HEALTHCARE,
                summary="LLM-powered CDS providing context-aware recommendations at point of care.",
                significance="25% reduction in diagnostic errors in early trials.",
                timeline="6-18 months for pilot deployments",
                key_players=["Epic Cognitive Computing", "Nuance DAX", "Abridge"],
                sources=[
                    Source(title="LLM-CDS Study", url="https://example.com/llm-cds", snippet="Significant improvement in diagnostic accuracy", relevance_score=0.91),
                ],
                confidence=0.84,
                tags=["CDS", "LLM", "EHR-integration"],
            ),
        ],
        analyses=[
            TrendAnalysis(
                trend_name="Multimodal Foundation Models for Clinical Imaging",
                current_state="Several models achieving specialist-level accuracy; FDA clearance pathway established.",
                opportunities=["Automated retinal disease screening", "AI-assisted OCT interpretation"],
                risks=["Regulatory uncertainty", "Dataset bias"],
                predictions=["FDA-cleared ophthalmology AI by late 2026"],
            ),
            TrendAnalysis(
                trend_name="Agentic AI for Healthcare Administration",
                current_state="Production deployments handling PA and claims. Agentic architecture is the new standard.",
                opportunities=["End-to-end PA automation", "Intelligent coding assistance", "Appeals automation"],
                risks=["Error propagation", "Payer resistance", "Integration complexity"],
                predictions=["50%+ of PA submissions AI-handled by 2027"],
            ),
            TrendAnalysis(
                trend_name="Real-Time Clinical Decision Support with LLMs",
                current_state="Pilot deployments at major health systems. EHR integration is the challenge.",
                opportunities=["Context-aware treatment recommendations", "Guideline compliance"],
                risks=["Hallucination in clinical context", "Workflow disruption"],
                predictions=["LLM-CDS will become standard EHR feature within 3 years"],
            ),
        ],
        methodology="Multi-agent research pipeline using Tavily web search and Claude analysis",
    )


def _build_demo_synthesis_report() -> SynthesisReport:
    """Build a demo SynthesisReport for end-to-end testing."""
    return SynthesisReport(
        title="Synthesis: Emerging Trends in AI and Healthcare — Q2 2026",
        research_source="Emerging Trends in AI and Healthcare — Q2 2026",
        executive_summary=(
            "Analysis of five healthcare AI trends reveals strong opportunities for both "
            "OphthoFlow and Xena. Agentic AI for administration is the highest-priority "
            "opportunity for OphthoFlow (directly aligned with PA automation), while "
            "multimodal foundation models and LLM-based CDS offer transformative potential "
            "for Xena's clinical decision support capabilities."
        ),
        trend_syntheses=[
            TrendSynthesis(
                trend_name="Multimodal Foundation Models for Clinical Imaging",
                relevance_summary="Directly applicable to OphthoFlow's ophthalmology focus; enables automated diagnostic screening.",
                maturity_assessment="Research-grade today; 12-18 months from production clinical deployment.",
                competitive_landscape="Google Health and Eyenuk lead; significant moat possible via ophthalmology specialization.",
                applications=[
                    ApplicationIdea(
                        title="AI-Assisted OCT Interpretation",
                        description="Integrate multimodal models to provide preliminary OCT scan reads, flagging urgent findings.",
                        platform=Platform.OPHTHOFLOW,
                        fit_level=FitLevel.HIGH,
                        impact=ImpactLevel.TRANSFORMATIVE,
                        effort=EffortLevel.HIGH,
                        use_case="Retina specialists reviewing 50+ OCT scans daily get AI pre-reads highlighting pathology.",
                        user_benefit="Reduces read time by 40%, catches subtle findings, enables triage.",
                        technical_approach="Fine-tune vision-language model on OCT dataset; deploy as async analysis service.",
                        dependencies=["OCT image pipeline", "FDA 510(k) clearance pathway"],
                        risks=["Regulatory timeline uncertainty", "Model bias on rare conditions"],
                    ),
                    ApplicationIdea(
                        title="Clinical Imaging Insights for Care Coordination",
                        description="Surface imaging findings in Xena's care coordination workflow for cross-specialty referrals.",
                        platform=Platform.XENA,
                        fit_level=FitLevel.MEDIUM,
                        impact=ImpactLevel.HIGH,
                        effort=EffortLevel.MEDIUM,
                        use_case="PCPs referring to ophthalmology see AI-generated imaging summaries.",
                        user_benefit="Better-informed referrals, reduced unnecessary specialist visits.",
                        technical_approach="API integration with imaging AI services; display in referral workflow.",
                        dependencies=["Imaging AI vendor partnership", "EHR interop"],
                        risks=["Data sharing agreements", "Imaging format variability"],
                    ),
                ],
                cross_platform_synergies=["Shared imaging AI pipeline serves both platforms"],
                watch_items=["FDA AI/ML SaMD regulatory guidance updates", "Google Med-PaLM 3 launch"],
                overall_priority=FitLevel.HIGH,
            ),
            TrendSynthesis(
                trend_name="Agentic AI for Healthcare Administration",
                relevance_summary="Core alignment with OphthoFlow's PA automation mission. Agentic AI IS OphthoFlow's competitive advantage.",
                maturity_assessment="Production-ready. Competitors already deploying. Urgency is high.",
                competitive_landscape="Crowded space but ophthalmology specialization is a moat.",
                applications=[
                    ApplicationIdea(
                        title="Autonomous PA Submission Agent",
                        description="End-to-end PA agent that gathers clinical data, selects codes, submits to payer portals, and handles responses.",
                        platform=Platform.OPHTHOFLOW,
                        fit_level=FitLevel.HIGH,
                        impact=ImpactLevel.TRANSFORMATIVE,
                        effort=EffortLevel.MEDIUM,
                        use_case="Practice staff clicks 'Submit PA' and the agent handles everything through approval.",
                        user_benefit="PA turnaround from days to hours; staff freed from portal navigation.",
                        technical_approach="Multi-step agent with tool-use for portal APIs, clinical data extraction, and coding logic.",
                        dependencies=["Payer portal API access", "Clinical data extraction pipeline"],
                        risks=["Payer portal changes", "Error in autonomous submissions"],
                    ),
                    ApplicationIdea(
                        title="Intelligent Appeals Agent",
                        description="Agent that automatically drafts and submits PA appeals with supporting clinical evidence.",
                        platform=Platform.OPHTHOFLOW,
                        fit_level=FitLevel.HIGH,
                        impact=ImpactLevel.HIGH,
                        effort=EffortLevel.MEDIUM,
                        use_case="When PA is denied, agent generates appeal letter citing clinical guidelines and patient data.",
                        user_benefit="Higher appeal success rate; eliminates hours of manual appeal drafting.",
                        technical_approach="RAG over clinical guidelines + patient chart; templated appeal generation.",
                        dependencies=["Clinical guidelines database", "Denial reason parsing"],
                        risks=["Medical-legal review requirements", "Guideline currency"],
                    ),
                ],
                cross_platform_synergies=["Agentic framework reusable for Xena's clinical workflows"],
                watch_items=["Olive AI's pivot to agents", "CMS PA interoperability rule enforcement"],
                overall_priority=FitLevel.HIGH,
            ),
            TrendSynthesis(
                trend_name="Real-Time Clinical Decision Support with LLMs",
                relevance_summary="Strong fit for Xena's clinical decision support mission; enhances OphthoFlow's coding assistance.",
                maturity_assessment="Pilot-stage. Promising results but production deployment requires careful validation.",
                competitive_landscape="Epic and Nuance have head start; specialty-focused CDS is differentiation opportunity.",
                applications=[
                    ApplicationIdea(
                        title="LLM-Powered Coding Assistant",
                        description="Real-time CPT/ICD-10 code suggestions based on clinical documentation context.",
                        platform=Platform.OPHTHOFLOW,
                        fit_level=FitLevel.HIGH,
                        impact=ImpactLevel.HIGH,
                        effort=EffortLevel.LOW,
                        use_case="As clinician documents procedure, system suggests optimal codes with confidence scores.",
                        user_benefit="Fewer coding errors, faster documentation, higher reimbursement accuracy.",
                        technical_approach="Fine-tuned LLM on ophthalmology coding data; streaming suggestions UI.",
                        dependencies=["Ophthalmology coding training data"],
                        risks=["Coding accuracy liability", "Model hallucination"],
                    ),
                    ApplicationIdea(
                        title="Context-Aware Clinical Recommendations",
                        description="Point-of-care LLM that synthesizes patient data and suggests evidence-based next steps.",
                        platform=Platform.XENA,
                        fit_level=FitLevel.HIGH,
                        impact=ImpactLevel.TRANSFORMATIVE,
                        effort=EffortLevel.HIGH,
                        use_case="Clinician reviewing patient chart gets AI-generated differential diagnosis and workup suggestions.",
                        user_benefit="Reduced diagnostic errors, faster decision-making, continuous education.",
                        technical_approach="RAG over clinical guidelines + patient longitudinal data; real-time inference.",
                        dependencies=["EHR data integration", "Clinical knowledge base"],
                        risks=["Hallucination risk", "Liability framework", "Clinician trust"],
                    ),
                ],
                cross_platform_synergies=["Shared clinical NLP pipeline", "Unified medical knowledge graph"],
                watch_items=["Epic Cognitive Computing announcements", "FDA guidance on LLM CDS"],
                overall_priority=FitLevel.HIGH,
            ),
        ],
        strategic_themes=[
            StrategicTheme(
                name="Ophthalmology AI Specialization as Competitive Moat",
                description="Deep specialization in ophthalmology workflows + AI creates defensible position vs. horizontal players.",
                contributing_trends=["Multimodal Foundation Models", "Agentic AI", "Clinical Decision Support"],
                strategic_implications=["Invest in ophthalmology-specific training data", "Build specialty AI benchmarks", "Partner with academic ophthalmology centers"],
            ),
            StrategicTheme(
                name="Agentic Architecture as Platform Foundation",
                description="Tool-using agent architecture is the right foundation for both PA automation and clinical workflows.",
                contributing_trends=["Agentic AI for Healthcare Administration", "Real-Time CDS"],
                strategic_implications=["Standardize on agentic framework across products", "Build shared tool library", "Invest in agent observability"],
            ),
        ],
        top_opportunities=[
            ApplicationIdea(
                title="Autonomous PA Submission Agent",
                description="End-to-end PA automation via agentic AI.",
                platform=Platform.OPHTHOFLOW,
                fit_level=FitLevel.HIGH,
                impact=ImpactLevel.TRANSFORMATIVE,
                effort=EffortLevel.MEDIUM,
                use_case="Fully automated PA submission and tracking.",
                user_benefit="PA turnaround from days to hours.",
            ),
            ApplicationIdea(
                title="LLM-Powered Coding Assistant",
                description="Real-time CPT/ICD-10 suggestions.",
                platform=Platform.OPHTHOFLOW,
                fit_level=FitLevel.HIGH,
                impact=ImpactLevel.HIGH,
                effort=EffortLevel.LOW,
                use_case="Contextual code suggestions during documentation.",
                user_benefit="Fewer errors, faster documentation.",
            ),
        ],
        quick_wins=[
            ApplicationIdea(
                title="LLM-Powered Coding Assistant",
                description="Real-time CPT/ICD-10 suggestions from clinical context.",
                platform=Platform.OPHTHOFLOW,
                fit_level=FitLevel.HIGH,
                impact=ImpactLevel.HIGH,
                effort=EffortLevel.LOW,
                use_case="Contextual code suggestions during documentation.",
                user_benefit="Fewer errors, faster documentation.",
            ),
        ],
        moonshots=[
            ApplicationIdea(
                title="AI-Assisted OCT Interpretation",
                description="Multimodal AI for automated retinal scan analysis.",
                platform=Platform.OPHTHOFLOW,
                fit_level=FitLevel.HIGH,
                impact=ImpactLevel.TRANSFORMATIVE,
                effort=EffortLevel.HIGH,
                use_case="AI pre-reads for retina specialists.",
                user_benefit="40% reduction in read time, catches subtle findings.",
            ),
        ],
        key_risks=[
            "FDA regulatory timeline for clinical AI is uncertain",
            "Payer resistance to AI-generated PA submissions",
            "LLM hallucination risk in clinical contexts requires robust guardrails",
            "Competitive pressure from well-funded horizontal AI platforms (Epic, Google)",
        ],
        recommended_next_steps=[
            "Immediately: Prototype autonomous PA submission agent (OphthoFlow)",
            "Q3 2026: Ship LLM-powered coding assistant as beta feature",
            "Q3 2026: Begin ophthalmology imaging AI partnership evaluation",
            "Q4 2026: Pilot LLM-CDS in Xena with selected health system partner",
            "Ongoing: Monitor FDA AI/ML SaMD guidance for regulatory strategy",
        ],
    )


async def run_demo(config: MemoConfig) -> None:
    """Run a demo memo generation cycle with sample data."""
    print("\n--- Memo Generation Agent Demo ---\n", file=sys.stderr)

    research_report = _build_demo_research_report()
    synthesis_report = _build_demo_synthesis_report()

    print(f"Research input: {research_report.title}", file=sys.stderr)
    print(f"Synthesis input: {synthesis_report.title}", file=sys.stderr)
    print(f"Trends: {len(research_report.trends)}", file=sys.stderr)
    print(f"Applications: {sum(len(ts.applications) for ts in synthesis_report.trend_syntheses)}", file=sys.stderr)

    # Run full pipeline
    agent = MemoGenerationAgent(config)
    bundle = await agent.run_full_pipeline(
        research_report=research_report,
        synthesis_report=synthesis_report,
        audience=MemoAudience.PRODUCT,
    )

    # Output bundle summary
    print(f"\n[demo] Pipeline complete!", file=sys.stderr)
    print(f"[demo] Bundle ID: {bundle.bundle_id}", file=sys.stderr)
    print(f"[demo] Research: {bundle.research_report_path}", file=sys.stderr)
    print(f"[demo] Synthesis: {bundle.synthesis_report_path}", file=sys.stderr)
    print(f"[demo] Memo (MD): {bundle.memo_path}", file=sys.stderr)
    print(f"[demo] Memo (HTML): {bundle.memo_html_path}", file=sys.stderr)

    # Output bundle manifest to stdout
    print(json.dumps(bundle.model_dump(mode="json"), indent=2, default=str))


async def run_from_files(
    research_path: str | None,
    synthesis_path: str,
    config: MemoConfig,
) -> None:
    """Load reports from JSON files and generate memo."""
    synth_file = Path(synthesis_path)
    if not synth_file.exists():
        print(f"Error: Synthesis file not found: {synthesis_path}", file=sys.stderr)
        sys.exit(1)

    synth_data = json.loads(synth_file.read_text(encoding="utf-8"))
    synthesis_report = SynthesisReport(**synth_data)
    print(f"Loaded synthesis: {synthesis_report.title}", file=sys.stderr)

    # Load research report if provided
    if research_path:
        res_file = Path(research_path)
        if not res_file.exists():
            print(f"Error: Research file not found: {research_path}", file=sys.stderr)
            sys.exit(1)
        res_data = json.loads(res_file.read_text(encoding="utf-8"))
        research_report = ResearchReport(**res_data)
    else:
        # Create minimal research report from synthesis metadata
        research_report = ResearchReport(
            title=synthesis_report.research_source or "Research Report",
            domain="healthcare AI",
        )

    print(f"Research: {research_report.title}", file=sys.stderr)

    # Run pipeline
    agent = MemoGenerationAgent(config)
    bundle = await agent.run_full_pipeline(
        research_report=research_report,
        synthesis_report=synthesis_report,
    )

    print(f"\nBundle ID: {bundle.bundle_id}", file=sys.stderr)
    print(json.dumps(bundle.model_dump(mode="json"), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memo Generation Agent — Produce formatted memos from synthesis data",
        prog="memo-generation-agent",
    )
    parser.add_argument(
        "--synthesis",
        help="Path to a SynthesisReport JSON file from Agent 2",
    )
    parser.add_argument(
        "--research",
        help="Path to a ResearchReport JSON file from Agent 1 (optional)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo data",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit",
    )
    args = parser.parse_args()

    config = MemoConfig.from_env()

    if args.validate:
        warnings = config.validate()
        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}", file=sys.stderr)
            sys.exit(1)
        print("[validate] Configuration OK", file=sys.stderr)
        sys.exit(0)

    if args.demo:
        asyncio.run(run_demo(config))
    elif args.synthesis:
        asyncio.run(run_from_files(args.research, args.synthesis, config))
    else:
        print("No input specified, running demo...\n", file=sys.stderr)
        asyncio.run(run_demo(config))


if __name__ == "__main__":
    main()
