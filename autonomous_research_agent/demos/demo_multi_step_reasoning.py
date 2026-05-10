#!/usr/bin/env python3
"""
Demo: Multi-Step Reasoning & Tool Orchestration

Showcases the agent's multi-step reasoning capabilities:
  - Query decomposition into prioritized sub-queries
  - Iterative search → analyze → fill-gaps → synthesize loop
  - Contradiction detection and resolution
  - Confidence-weighted finding aggregation
  - Evidence chain tracking from source to insight

Runs entirely with mock data to illustrate the reasoning pipeline.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Any

from autonomous_research_agent.models import (
    ActionableInsight,
    ConfidenceLevel,
    Evidence,
    Finding,
    FindingType,
    InsightPriority,
    QueryPlan,
    ReportSection,
    ResearchReport,
    Source,
    SubQuery,
    Synthesis,
)


def print_header(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step: int, title: str, detail: str = "") -> None:
    print(f"\n  Step {step}: {title}")
    if detail:
        for line in detail.split("\n"):
            print(f"    {line}")


async def run_demo() -> None:
    """Demonstrate multi-step reasoning through the full research pipeline."""

    print_header("MULTI-STEP REASONING DEMO")
    print("  Showing how the agent reasons through a complex research query")
    print()

    query = "Should enterprises adopt AI agents for customer support in 2026?"
    print(f"  Query: {query}")

    # -----------------------------------------------------------------------
    # Step 1: Query Decomposition (structured reasoning)
    # -----------------------------------------------------------------------
    print_step(1, "Query Decomposition — Breaking down the question")

    plan = QueryPlan(
        original_query=query,
        sub_queries=[
            SubQuery(
                question="What is the current ROI of AI agents in customer support?",
                rationale="ROI is the primary decision factor for enterprise adoption.",
                search_terms=["AI agent customer support ROI 2026", "chatbot vs agent support metrics"],
                priority=1,
            ),
            SubQuery(
                question="What are the risks and failure modes of support agents?",
                rationale="Risk assessment is critical for enterprise decision-making.",
                search_terms=["AI agent customer support failures", "chatbot escalation rates"],
                priority=2,
            ),
            SubQuery(
                question="How do AI agents compare to traditional support solutions?",
                rationale="Need a baseline for comparison — what's the alternative?",
                search_terms=["AI agent vs human support cost comparison", "hybrid support models"],
                priority=3,
            ),
            SubQuery(
                question="What do early adopters report after 12+ months?",
                rationale="Long-term data is more credible than vendor projections.",
                search_terms=["enterprise AI agent case study results", "support agent long-term outcomes"],
                priority=2,
            ),
        ],
        research_angles=["ROI & economics", "Risk & safety", "Competitive landscape", "Real-world outcomes"],
        estimated_searches=6,
    )

    print(f"    Identified {len(plan.sub_queries)} sub-questions:")
    for sq in plan.sub_queries:
        print(f"      P{sq.priority} — {sq.question}")
        print(f"           Rationale: {sq.rationale}")
    print(f"    Research angles: {', '.join(plan.research_angles)}")
    print(f"    Estimated searches: {plan.estimated_searches}")

    # -----------------------------------------------------------------------
    # Step 2: Broad Search & Finding Extraction
    # -----------------------------------------------------------------------
    print_step(2, "Broad Search — Gathering evidence from multiple angles")

    sources = [
        Source(title="Gartner: AI Agent ROI Report 2026", url="https://example.com/gartner-roi", relevance_score=0.96),
        Source(title="Forrester: Customer Support Automation", url="https://example.com/forrester", relevance_score=0.92),
        Source(title="McKinsey: AI in Enterprise Operations", url="https://example.com/mckinsey", relevance_score=0.89),
        Source(title="TechCrunch: When AI Agents Fail", url="https://example.com/tc-failures", relevance_score=0.85),
        Source(title="HBR: The Human-AI Support Balance", url="https://example.com/hbr-balance", relevance_score=0.83),
    ]

    print(f"    Found {len(sources)} relevant sources:")
    for s in sources:
        print(f"      [{s.relevance_score:.2f}] {s.title}")

    # -----------------------------------------------------------------------
    # Step 3: Finding Extraction with Contradiction Detection
    # -----------------------------------------------------------------------
    print_step(3, "Finding Extraction — Identifying facts, trends, and contradictions")

    findings = [
        Finding(
            headline="AI support agents deliver 3.2x ROI within 12 months",
            detail="Based on analysis of 150+ enterprise deployments.",
            finding_type=FindingType.DATA_POINT,
            confidence=ConfidenceLevel.HIGH,
            evidence=[Evidence(claim="3.2x ROI", source=sources[0], confidence=ConfidenceLevel.HIGH)],
            sources=[sources[0], sources[1]],
        ),
        Finding(
            headline="15-20% of support queries still require human escalation",
            detail="Complex, emotional, or novel issues are beyond current agent capabilities.",
            finding_type=FindingType.FACT,
            confidence=ConfidenceLevel.HIGH,
            evidence=[Evidence(claim="15-20% escalation rate", source=sources[1], confidence=ConfidenceLevel.HIGH)],
            sources=[sources[1]],
        ),
        Finding(
            headline="Customer satisfaction scores improve by 12% with AI agents",
            detail="Driven by faster response times and 24/7 availability.",
            finding_type=FindingType.TREND,
            confidence=ConfidenceLevel.MEDIUM,
            sources=[sources[0], sources[2]],
        ),
        Finding(
            headline="CONTRADICTION: Two sources disagree on cost savings magnitude",
            detail="Gartner says 35% cost reduction; McKinsey says 22%. Likely due to "
                   "different sample compositions (Gartner includes only mature deployments).",
            finding_type=FindingType.CONTRADICTION,
            confidence=ConfidenceLevel.MEDIUM,
            sources=[sources[0], sources[2]],
        ),
        Finding(
            headline="Poorly implemented agents can damage brand trust",
            detail="42% of consumers report frustration with AI agents that can't handle "
                   "their specific issue and don't escalate appropriately.",
            finding_type=FindingType.RISK,
            confidence=ConfidenceLevel.HIGH,
            sources=[sources[3]],
        ),
    ]

    for f in findings:
        icon = {
            FindingType.DATA_POINT: "DATA",
            FindingType.FACT: "FACT",
            FindingType.TREND: "TREND",
            FindingType.CONTRADICTION: "CONTRADICTION",
            FindingType.RISK: "RISK",
        }.get(f.finding_type, "?")
        conf = f.confidence.value.upper()
        marker = " *** " if f.finding_type == FindingType.CONTRADICTION else "     "
        print(f"   {marker}[{icon}] ({conf}) {f.headline}")

    # -----------------------------------------------------------------------
    # Step 4: Gap Analysis — What's missing?
    # -----------------------------------------------------------------------
    print_step(4, "Gap Analysis — Identifying missing information")

    gaps = [
        "No data on implementation timeline and ramp-up costs",
        "Missing perspective from support agents (employee impact)",
        "Need industry-specific data (healthcare vs retail vs SaaS)",
    ]
    print("    Gaps identified:")
    for g in gaps:
        print(f"      ⟐ {g}")
    print("    → Running 2 additional targeted searches to fill gaps")

    # Simulate gap-filling search
    gap_findings = [
        Finding(
            headline="Average implementation takes 3-6 months with $200K-500K investment",
            detail="Varies significantly by complexity and integration requirements.",
            finding_type=FindingType.DATA_POINT,
            confidence=ConfidenceLevel.MEDIUM,
            sources=[sources[2]],
        ),
        Finding(
            headline="Healthcare sees 2.1x ROI; retail sees 4.5x ROI for support agents",
            detail="Retail benefits more due to higher volume, lower complexity queries.",
            finding_type=FindingType.DATA_POINT,
            confidence=ConfidenceLevel.MEDIUM,
            sources=[sources[2], sources[4]],
        ),
    ]
    print("\n    Gap-fill findings:")
    for f in gap_findings:
        print(f"      [DATA] ({f.confidence.value.upper()}) {f.headline}")
    findings.extend(gap_findings)

    # -----------------------------------------------------------------------
    # Step 5: Cross-Reference Synthesis
    # -----------------------------------------------------------------------
    print_step(5, "Synthesis — Cross-referencing and resolving contradictions")

    syntheses = [
        Synthesis(
            theme="Strong but nuanced ROI",
            summary="ROI is consistently positive (2.1x-4.5x) but varies significantly by "
                    "industry and implementation maturity. The Gartner/McKinsey cost savings "
                    "contradiction resolves when controlling for deployment maturity.",
            supporting_findings=[f.headline for f in findings[:3]],
            contradictions=["Cost savings estimates vary 22-35% depending on sample maturity"],
            confidence=ConfidenceLevel.HIGH,
            implications=[
                "ROI projections should be industry-specific",
                "Mature deployments see significantly better results",
                "Budget for 6+ month ramp-up period before expecting full ROI",
            ],
        ),
        Synthesis(
            theme="The hybrid model wins",
            summary="Pure AI agent deployments underperform hybrid human+AI models. "
                    "The 15-20% escalation rate is a feature, not a bug — it protects "
                    "brand trust for complex cases.",
            supporting_findings=[findings[1].headline, findings[4].headline],
            confidence=ConfidenceLevel.HIGH,
            implications=[
                "Plan for seamless human handoff from day one",
                "Staff human agents for escalation, not full volume",
            ],
        ),
    ]

    for syn in syntheses:
        print(f"\n    Theme: {syn.theme}")
        print(f"    Confidence: {syn.confidence.value}")
        print(f"    Summary: {syn.summary[:100]}...")
        if syn.contradictions:
            print(f"    Contradiction resolved: {syn.contradictions[0]}")
        print(f"    Implications:")
        for imp in syn.implications:
            print(f"      → {imp}")

    # -----------------------------------------------------------------------
    # Step 6: Actionable Insights
    # -----------------------------------------------------------------------
    print_step(6, "Actionable Insights — Deriving recommendations from evidence")

    insights = [
        ActionableInsight(
            title="Adopt hybrid AI+human support model",
            description="Deploy AI agents for 80% of queries with seamless human escalation.",
            priority=InsightPriority.HIGH,
            rationale="3.2x ROI proven, but pure-AI approach risks brand damage (42% frustration rate).",
            next_steps=[
                "Select agent SDK (Claude or OpenAI based on integration needs)",
                "Start with top-10 most common support queries",
                "Build escalation triggers for sentiment and complexity signals",
                "Run 90-day pilot, measure CSAT and cost per resolution",
            ],
            risks_if_ignored="Competitors already deploying — 34% of Fortune 500 have agents in production.",
            timeline="3-6 months for initial deployment, 12 months to full ROI",
        ),
        ActionableInsight(
            title="Invest in agent monitoring before scaling",
            description="Build real-time monitoring for agent quality, escalation rates, and cost.",
            priority=InsightPriority.CRITICAL,
            rationale="42% consumer frustration rate with poorly implemented agents. "
                      "Without monitoring, you won't know until brand damage occurs.",
            next_steps=[
                "Implement CSAT tracking per agent interaction",
                "Set up cost alerting and budget caps",
                "Create weekly agent performance review process",
            ],
            risks_if_ignored="Undetected agent failures erode customer trust.",
            timeline="Implement before or concurrent with agent deployment",
        ),
    ]

    for insight in insights:
        badge = insight.priority.value.upper()
        print(f"\n    [{badge}] {insight.title}")
        print(f"    {insight.description}")
        print(f"    Rationale: {insight.rationale[:80]}...")
        print(f"    Next steps: {len(insight.next_steps)} concrete actions")
        print(f"    Timeline: {insight.timeline}")

    # -----------------------------------------------------------------------
    # Step 7: Build structured report
    # -----------------------------------------------------------------------
    print_step(7, "Report Assembly — Building structured output")

    report = ResearchReport(
        title="Should Enterprises Adopt AI Agents for Customer Support in 2026?",
        query=query,
        executive_summary=(
            "Yes, with caveats. AI agents deliver strong ROI (2.1x-4.5x depending on industry) "
            "and 34% of Fortune 500 companies already have them in production. However, success "
            "requires a hybrid model with human escalation, proper monitoring, and a realistic "
            "3-6 month ramp-up timeline. Pure-AI deployments risk customer frustration."
        ),
        key_takeaways=[
            "AI support agents deliver 3.2x average ROI within 12 months",
            "Hybrid AI+human model outperforms pure-AI deployment",
            "15-20% of queries still need human escalation — plan for it",
            "Implementation costs $200K-500K with 3-6 month timeline",
            "Retail sees highest ROI (4.5x); healthcare lowest (2.1x)",
            "Agent monitoring is critical — 42% frustration rate without it",
        ],
        sections=[
            ReportSection(title="ROI Analysis", content="...", findings=findings[:3]),
            ReportSection(title="Risk Assessment", content="...", findings=[findings[4]]),
            ReportSection(title="Industry Breakdown", content="...", findings=gap_findings),
        ],
        actionable_insights=insights,
        syntheses=syntheses,
        all_sources=sources,
        metadata={
            "search_count": 6,
            "fetch_count": 3,
            "duration_seconds": 67.3,
            "findings_count": len(findings),
        },
    )

    md = report.to_markdown()
    print(f"\n    Report: \"{report.title}\"")
    print(f"    Sections: {len(report.sections)}")
    print(f"    Findings: {len(findings)}")
    print(f"    Insights: {len(insights)}")
    print(f"    Sources: {len(sources)}")
    print(f"    Markdown length: {len(md):,} chars")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_header("MULTI-STEP REASONING COMPLETE")
    print()
    print("  Reasoning chain demonstrated:")
    print("    1. Decomposed complex yes/no question into 4 research angles")
    print("    2. Prioritized sub-queries by decision relevance")
    print("    3. Extracted 7 typed findings (facts, trends, risks, data points)")
    print("    4. Detected and resolved 1 contradiction between sources")
    print("    5. Identified 3 research gaps and filled them with targeted searches")
    print("    6. Synthesized 2 cross-cutting themes with implications")
    print("    7. Derived 2 actionable insights with concrete next steps")
    print("    8. Assembled everything into a structured, evidence-backed report")
    print()
    print("  Key reasoning capabilities shown:")
    print("    • Query decomposition with prioritization")
    print("    • Contradiction detection and resolution")
    print("    • Gap analysis and iterative research")
    print("    • Confidence-weighted evidence aggregation")
    print("    • From evidence → synthesis → actionable insight pipeline")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_demo())
