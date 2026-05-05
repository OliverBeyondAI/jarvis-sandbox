"""
CLI entry point for the Synthesis Agent.

Usage:
    python -m agents.synthesis --input report.json
    python -m agents.synthesis --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..trend_research.models import (
    ResearchReport,
    Source,
    Trend,
    TrendAnalysis,
    TrendCategory,
)
from .agent import SynthesisAgent, run_full_synthesis
from .config import SynthesisConfig


def _build_demo_report() -> ResearchReport:
    """Build a realistic demo ResearchReport for testing the synthesis agent."""
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
                summary=(
                    "Large vision-language models trained on medical imaging datasets "
                    "are achieving specialist-level performance in radiology and "
                    "ophthalmology diagnostics, enabling automated screening and "
                    "triage at scale."
                ),
                significance=(
                    "Could transform diagnostic workflows by providing AI-assisted "
                    "reads that augment clinician expertise, reduce turnaround times, "
                    "and catch subtle findings."
                ),
                timeline="12-18 months for production-grade clinical deployment",
                key_players=["Google Health", "Microsoft Nuance", "Retina-AI", "Eyenuk"],
                sources=[
                    Source(title="Foundation Models in Medical Imaging 2026", url="https://example.com/fm-imaging", snippet="Survey of multimodal models achieving 95%+ accuracy on retinal disease classification", relevance_score=0.95),
                    Source(title="GPT-4V in Ophthalmology", url="https://example.com/gpt4v-ophth", snippet="Evaluation of vision-language models for OCT and fundus image interpretation", relevance_score=0.9),
                ],
                confidence=0.88,
                tags=["imaging", "diagnostics", "ophthalmology", "foundation-models"],
            ),
            Trend(
                name="Agentic AI for Healthcare Administration",
                category=TrendCategory.AI_ML,
                summary=(
                    "AI agents that autonomously handle prior authorizations, "
                    "claims processing, and scheduling are reducing administrative "
                    "burden by 40-60% in early deployments."
                ),
                significance=(
                    "Administrative overhead consumes 30% of healthcare spending. "
                    "Agentic AI that handles PA workflows, coding, and documentation "
                    "end-to-end could save billions annually."
                ),
                timeline="6-12 months — some solutions already in production",
                key_players=["Olive AI", "Waystar", "Infinitus Health", "Cohere Health"],
                sources=[
                    Source(title="AI Agents Transform Prior Auth", url="https://example.com/ai-pa", snippet="Agentic systems cut PA turnaround from 14 days to under 24 hours", relevance_score=0.92),
                    Source(title="RCM Automation Benchmark 2026", url="https://example.com/rcm-bench", snippet="Comparison of AI-driven revenue cycle management solutions", relevance_score=0.85),
                ],
                confidence=0.91,
                tags=["agents", "prior-auth", "administration", "RCM"],
            ),
            Trend(
                name="Federated Learning for Clinical AI",
                category=TrendCategory.AI_ML,
                summary=(
                    "Federated and privacy-preserving training techniques are enabling "
                    "multi-institutional model development without sharing raw patient "
                    "data, accelerating FDA-cleared AI tool pipelines."
                ),
                significance=(
                    "Unlocks training on diverse, multi-site datasets while maintaining "
                    "HIPAA compliance, producing models that generalize better across "
                    "patient populations."
                ),
                timeline="12-24 months for widespread adoption",
                key_players=["NVIDIA FLARE", "Rhino Health", "Owkin", "Apheris"],
                sources=[
                    Source(title="Federated Learning Healthcare Review", url="https://example.com/fl-review", snippet="Meta-analysis of federated approaches achieving near-centralized accuracy", relevance_score=0.87),
                ],
                confidence=0.78,
                tags=["privacy", "federated-learning", "training", "compliance"],
            ),
            Trend(
                name="Real-Time Clinical Decision Support with LLMs",
                category=TrendCategory.HEALTHCARE,
                summary=(
                    "LLM-powered CDS systems are providing context-aware, evidence-based "
                    "recommendations at the point of care, integrating with EHR workflows "
                    "to surface relevant guidelines and flag potential issues."
                ),
                significance=(
                    "Addresses alert fatigue by providing nuanced, contextual support "
                    "rather than rigid rule-based alerts. Early trials show 25% reduction "
                    "in diagnostic errors."
                ),
                timeline="6-18 months for pilot deployments",
                key_players=["Epic Cognitive Computing", "Nuance DAX", "Abridge", "Suki AI"],
                sources=[
                    Source(title="LLM-CDS Point of Care Study", url="https://example.com/llm-cds", snippet="Randomized trial showing significant improvement in diagnostic accuracy with LLM-CDS", relevance_score=0.91),
                ],
                confidence=0.84,
                tags=["CDS", "LLM", "point-of-care", "EHR-integration"],
            ),
            Trend(
                name="AI-Powered Patient Engagement Platforms",
                category=TrendCategory.HEALTHCARE,
                summary=(
                    "Conversational AI platforms are handling patient intake, medication "
                    "adherence, post-operative follow-up, and chronic disease management "
                    "through natural language interactions across text and voice channels."
                ),
                significance=(
                    "Improves patient outcomes through proactive engagement while "
                    "reducing no-show rates and call center volume. Particularly "
                    "impactful for chronic disease management in ophthalmology."
                ),
                timeline="Already in production; advanced features 6-12 months out",
                key_players=["Hyro", "Syllable", "Hippocratic AI", "K Health"],
                sources=[
                    Source(title="Patient Engagement AI Market 2026", url="https://example.com/pe-market", snippet="Market projected to reach $4.2B by 2027 driven by conversational AI adoption", relevance_score=0.83),
                ],
                confidence=0.86,
                tags=["patient-engagement", "conversational-AI", "adherence", "chronic-care"],
            ),
        ],
        analyses=[
            TrendAnalysis(
                trend_name="Multimodal Foundation Models for Clinical Imaging",
                current_state="Several models achieving specialist-level accuracy in research settings; FDA clearance pathway established but few commercial products yet.",
                opportunities=["Automated retinal disease screening", "AI-assisted OCT interpretation", "Pre-visit diagnostic triage"],
                risks=["Regulatory uncertainty", "Liability for AI-assisted diagnoses", "Dataset bias across populations"],
                predictions=["FDA-cleared multimodal ophthalmology AI by late 2026", "Integration into standard ophtho workflows within 2 years"],
            ),
            TrendAnalysis(
                trend_name="Agentic AI for Healthcare Administration",
                current_state="Several production deployments handling PA and claims. Agentic architecture (tool-use, multi-step reasoning) is the new standard.",
                opportunities=["End-to-end PA automation", "Intelligent coding assistance", "Appeals automation"],
                risks=["Error propagation in autonomous workflows", "Payer resistance to AI submissions", "Integration complexity with legacy systems"],
                predictions=["50%+ of PA submissions AI-handled by 2027", "Agentic approach will displace rigid RPA solutions"],
            ),
            TrendAnalysis(
                trend_name="Federated Learning for Clinical AI",
                current_state="Proven in research; production infrastructure emerging via NVIDIA FLARE and Rhino Health.",
                opportunities=["Multi-practice model training without data sharing", "Improved model fairness across demographics"],
                risks=["Infrastructure complexity", "Communication overhead", "Heterogeneous data quality"],
                predictions=["Standard approach for multi-site clinical AI training by 2027"],
            ),
            TrendAnalysis(
                trend_name="Real-Time Clinical Decision Support with LLMs",
                current_state="Pilot deployments at major health systems. Integration with existing EHR alerting infrastructure is the primary challenge.",
                opportunities=["Context-aware treatment recommendations", "Guideline compliance monitoring", "Differential diagnosis support"],
                risks=["Hallucination in clinical context", "Workflow disruption", "Evidence lag"],
                predictions=["LLM-CDS will become standard EHR feature within 3 years"],
            ),
            TrendAnalysis(
                trend_name="AI-Powered Patient Engagement Platforms",
                current_state="Conversational AI for scheduling and FAQ well-established. Clinical-grade engagement (adherence, follow-up) is the growth frontier.",
                opportunities=["Automated injection adherence reminders for anti-VEGF patients", "Post-op follow-up automation", "Chronic disease management programs"],
                risks=["Patient trust and adoption barriers", "Liability for medical advice via AI", "Health literacy gaps"],
                predictions=["AI patient engagement will be table-stakes for specialty practices by 2027"],
            ),
        ],
        methodology="Multi-agent research pipeline using Tavily web search and Claude analysis",
    )


async def run_demo(config: SynthesisConfig) -> None:
    """Run a demo synthesis cycle with sample research data."""
    print("\n--- Synthesis Agent Demo ---\n", file=sys.stderr)

    # Build demo research report
    research_report = _build_demo_report()
    print(f"Input report: {research_report.title}", file=sys.stderr)
    print(f"Trends: {len(research_report.trends)}", file=sys.stderr)

    # Run synthesis
    synthesis_report = await run_full_synthesis(research_report, config)

    # Store locally
    storage_dir = Path(config.local_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    output_path = storage_dir / "demo_synthesis_report.json"
    output_path.write_text(
        json.dumps(synthesis_report.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {output_path}", file=sys.stderr)

    # Output to stdout
    print(json.dumps(synthesis_report.model_dump(mode="json"), indent=2, default=str))


async def run_from_file(input_path: str, config: SynthesisConfig) -> None:
    """Load a ResearchReport from JSON file and run synthesis."""
    path = Path(input_path)
    if not path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    research_report = ResearchReport(**data)
    print(f"Loaded report: {research_report.title}", file=sys.stderr)

    # Run synthesis
    synthesis_report = await run_full_synthesis(research_report, config)

    # Store output
    storage_dir = Path(config.local_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    output_path = storage_dir / "synthesis_report.json"
    output_path.write_text(
        json.dumps(synthesis_report.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nReport saved to: {output_path}", file=sys.stderr)

    # Output to stdout
    print(json.dumps(synthesis_report.model_dump(mode="json"), indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesis Agent — Map research trends to OphthoFlow & Xena applications",
        prog="synthesis-agent",
    )
    parser.add_argument(
        "--input",
        help="Path to a ResearchReport JSON file from Agent 1",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo research data",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit",
    )
    args = parser.parse_args()

    config = SynthesisConfig.from_env()

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
    elif args.input:
        asyncio.run(run_from_file(args.input, config))
    else:
        # Default: run demo
        print("No input specified, running demo...\n", file=sys.stderr)
        asyncio.run(run_demo(config))


if __name__ == "__main__":
    main()
