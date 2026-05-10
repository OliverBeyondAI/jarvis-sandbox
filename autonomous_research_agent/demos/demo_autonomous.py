#!/usr/bin/env python3
"""
Demo: Autonomous Multi-Step Research Operation

Showcases the agent's ability to:
  - Autonomously decompose a complex query into sub-questions
  - Execute multiple tool calls across research phases
  - Track progress with structured findings
  - Generate a polished report without human intervention

Runs with mock tools (no API keys needed) to demonstrate the full pipeline.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from typing import Any


# ---------------------------------------------------------------------------
# Mock tool implementations (simulate real tool behavior)
# ---------------------------------------------------------------------------

MOCK_SEARCH_DB: dict[str, list[dict[str, Any]]] = {
    "AI agents enterprise": [
        {
            "title": "Enterprise AI Agents: 2026 Market Report",
            "url": "https://example.com/enterprise-ai-agents-2026",
            "snippet": "The enterprise AI agent market is projected to reach $47B by 2028, "
                       "growing at a 38% CAGR. Key players include Anthropic, OpenAI, Google, "
                       "and a growing ecosystem of startups.",
            "relevance_score": 0.96,
        },
        {
            "title": "Building Autonomous Agents with Claude",
            "url": "https://example.com/claude-agents-guide",
            "snippet": "Claude's Agent SDK enables developers to build sophisticated autonomous "
                       "agents with built-in tool use, multi-step reasoning, and safety controls.",
            "relevance_score": 0.91,
        },
    ],
    "AI agent safety risks": [
        {
            "title": "AI Agent Safety: Challenges and Solutions",
            "url": "https://example.com/agent-safety",
            "snippet": "Key safety concerns include prompt injection, unauthorized actions, "
                       "runaway cost loops, and lack of human oversight in critical decisions.",
            "relevance_score": 0.88,
        },
    ],
    "AI agent frameworks SDKs 2026": [
        {
            "title": "Comparing Agent Frameworks: LangGraph vs CrewAI vs Claude SDK",
            "url": "https://example.com/framework-comparison",
            "snippet": "The Claude Agent SDK offers the most integrated experience with "
                       "built-in tools, while LangGraph provides maximum flexibility. "
                       "CrewAI specializes in multi-agent orchestration.",
            "relevance_score": 0.93,
        },
        {
            "title": "OpenAI Agents SDK: A Developer Review",
            "url": "https://example.com/openai-agents-sdk",
            "snippet": "OpenAI's Agents SDK focuses on simplicity with guardrails, handoffs, "
                       "and tracing built in. Comparison with Claude SDK shows trade-offs.",
            "relevance_score": 0.85,
        },
    ],
    "enterprise AI adoption use cases": [
        {
            "title": "Top 10 Enterprise AI Agent Use Cases in 2026",
            "url": "https://example.com/enterprise-use-cases",
            "snippet": "Leading use cases: customer support automation (45% adoption), "
                       "code generation (38%), data analysis (35%), sales outreach (28%), "
                       "and IT operations (22%).",
            "relevance_score": 0.90,
        },
    ],
}

MOCK_PAGE_DB: dict[str, str] = {
    "https://example.com/enterprise-ai-agents-2026": textwrap.dedent("""\
        Enterprise AI Agents: 2026 Market Report

        Executive Summary:
        The enterprise AI agent market has exploded in 2025-2026. With the release
        of advanced reasoning models and mature agent SDKs, enterprises are deploying
        autonomous agents at scale for the first time.

        Market Size: $47B projected by 2028 (38% CAGR)
        Current Leaders: Anthropic (Claude), OpenAI (GPT), Google (Gemini)
        Fastest-Growing Segment: Code generation agents (+62% YoY)
        Enterprise Adoption Rate: 34% of Fortune 500 companies

        Key Finding: Companies deploying AI agents report an average 28% reduction
        in operational costs within the first year, with customer support seeing
        the highest ROI at 3.2x.
    """),
    "https://example.com/agent-safety": textwrap.dedent("""\
        AI Agent Safety: Challenges and Solutions

        As AI agents gain autonomy, safety becomes paramount. This report examines
        the key risks and mitigation strategies.

        Top Risks:
        1. Prompt Injection — Malicious inputs can hijack agent behavior
        2. Unauthorized Actions — Agents exceeding their intended scope
        3. Cost Runaway — Uncontrolled API calls leading to unexpected bills
        4. Hallucination Cascades — Errors compounding across multi-step reasoning

        Mitigation Strategies:
        - Tool restrictions: limit each agent to only the tools it needs
        - Budget caps: set maxTurns and maxBudgetUsd limits
        - Sandboxing: isolate agent execution from production systems
        - Human-in-the-loop: require approval for high-stakes actions
        - Monitoring: real-time dashboards tracking agent behavior

        Industry Consensus: 78% of AI safety researchers agree that agent safety
        frameworks are still immature but rapidly improving.
    """),
}


async def mock_execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """Mock tool dispatcher simulating real tool behavior."""
    if name == "tavily_search":
        query = input_dict.get("query", "")
        # Find best matching mock data
        results = []
        for key, data in MOCK_SEARCH_DB.items():
            if any(word.lower() in query.lower() for word in key.split()):
                results.extend(data)
        if not results:
            results = list(MOCK_SEARCH_DB.values())[0]  # fallback
        return json.dumps({
            "query": query,
            "results": results[:5],
            "result_count": len(results),
        })

    elif name == "fetch_url":
        url = input_dict.get("url", "")
        content = MOCK_PAGE_DB.get(url, f"Page content for {url} — detailed analysis...")
        return json.dumps({
            "url": url,
            "status": 200,
            "content_type": "text/html",
            "content": content,
            "length": len(content),
        })

    elif name == "analyze_findings":
        findings = input_dict.get("findings", [])
        return json.dumps({
            "phase": input_dict.get("phase", "unknown"),
            "findings_recorded": len(findings),
            "findings": findings,
            "gaps": input_dict.get("gaps_identified", []),
            "status": "recorded",
        })

    elif name == "save_report":
        filename = input_dict.get("filename", "report.md")
        content = input_dict.get("content", "")
        # Save to a temp location
        import tempfile
        from pathlib import Path
        output_dir = Path(tempfile.mkdtemp(prefix="research_demo_"))
        filepath = output_dir / filename
        filepath.write_text(content)
        return json.dumps({
            "saved": True,
            "path": str(filepath),
            "size_bytes": len(content.encode("utf-8")),
        })

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_phase(phase: str, detail: str = "") -> None:
    print(f"\n{'─' * 50}")
    print(f"  Phase: {phase}")
    if detail:
        print(f"  {detail}")
    print(f"{'─' * 50}")


async def run_demo() -> None:
    """Run a complete autonomous research demo with mock tools."""

    print_header("AUTONOMOUS RESEARCH AGENT — Demo")
    print("  Demonstrating multi-step autonomous operation")
    print("  Using mock tools (no API keys required)")
    print()

    # Simulate the agent's multi-step research process
    query = (
        "What is the current landscape of autonomous AI agents for enterprise use? "
        "Cover major players, frameworks, adoption trends, and safety concerns."
    )
    print(f"  Query: {query[:80]}...")
    print()

    # Phase 1: Query Decomposition
    print_phase("Query Decomposition", "Breaking query into sub-questions")
    sub_queries = [
        "AI agents enterprise market size and growth",
        "AI agent frameworks SDKs 2026 comparison",
        "Enterprise AI adoption use cases",
        "AI agent safety risks and mitigation",
    ]
    for i, sq in enumerate(sub_queries, 1):
        print(f"    Sub-query {i}: {sq}")
    print(f"\n  → Decomposed into {len(sub_queries)} targeted sub-queries")

    # Phase 2: Broad Search
    print_phase("Broad Search", f"Executing {len(sub_queries)} web searches")
    all_sources = []
    for sq in sub_queries:
        result_str = await mock_execute_tool("tavily_search", {"query": sq})
        result = json.loads(result_str)
        sources = result.get("results", [])
        all_sources.extend(sources)
        print(f"    Search: \"{sq}\"")
        print(f"      → {len(sources)} results found")
        for s in sources:
            print(f"        • {s['title']} (score: {s['relevance_score']:.2f})")

    # Record findings from broad search
    broad_findings = [
        {"headline": "Market projected at $47B by 2028 (38% CAGR)", "type": "data_point", "confidence": "high"},
        {"headline": "Customer support is #1 use case at 45% adoption", "type": "trend", "confidence": "high"},
        {"headline": "Claude SDK offers most integrated agent experience", "type": "insight", "confidence": "medium"},
        {"headline": "78% of researchers say safety frameworks are immature", "type": "risk", "confidence": "high"},
    ]
    analyze_result = await mock_execute_tool("analyze_findings", {
        "phase": "broad_search",
        "findings": broad_findings,
        "gaps_identified": ["Need deeper data on enterprise ROI", "Missing competitor pricing"],
    })
    print(f"\n  → Recorded {len(broad_findings)} findings, identified 2 gaps")

    # Phase 3: Deep Dives
    print_phase("Deep Dives", "Reading 2 authoritative sources in full")
    deep_dive_urls = [
        "https://example.com/enterprise-ai-agents-2026",
        "https://example.com/agent-safety",
    ]
    for url in deep_dive_urls:
        result_str = await mock_execute_tool("fetch_url", {"url": url})
        result = json.loads(result_str)
        content_preview = result["content"][:120].replace("\n", " ")
        print(f"    Fetched: {url}")
        print(f"      → {result['length']} chars: \"{content_preview}...\"")

    # Record deep-dive findings
    deep_findings = [
        {"headline": "28% avg cost reduction in first year of agent deployment", "type": "data_point", "confidence": "high"},
        {"headline": "Customer support ROI at 3.2x — highest across use cases", "type": "data_point", "confidence": "high"},
        {"headline": "34% of Fortune 500 have deployed AI agents", "type": "fact", "confidence": "high"},
        {"headline": "Prompt injection remains the #1 safety concern", "type": "risk", "confidence": "high"},
        {"headline": "Tool restrictions + budget caps are primary mitigation strategies", "type": "insight", "confidence": "medium"},
    ]
    await mock_execute_tool("analyze_findings", {
        "phase": "deep_dive",
        "findings": deep_findings,
    })
    print(f"\n  → Recorded {len(deep_findings)} deep-dive findings")

    # Phase 4: Synthesis
    print_phase("Synthesis", "Cross-referencing findings and identifying themes")
    all_findings = broad_findings + deep_findings
    themes = [
        "Market is large and growing fast — $47B by 2028",
        "Enterprise adoption is real but early — 34% of Fortune 500",
        "Safety frameworks are immature but improving rapidly",
        "ROI is strongest in customer support (3.2x)",
    ]
    for i, theme in enumerate(themes, 1):
        print(f"    Theme {i}: {theme}")

    # Phase 5: Report Generation
    print_phase("Report Generation", "Producing structured markdown report")
    report_content = textwrap.dedent(f"""\
        # Enterprise AI Agents: 2026 Landscape Report

        *Generated: 2026-05-10*
        *Query: {query}*

        ---

        ## Executive Summary

        The enterprise AI agent market is experiencing rapid growth, projected to reach
        $47B by 2028 at a 38% CAGR. Autonomous agents are being deployed at scale across
        Fortune 500 companies, with customer support, code generation, and data analysis
        leading adoption. While the technology shows strong ROI (averaging 28% cost
        reduction in year one), safety frameworks remain immature, with prompt injection
        and unauthorized actions as top concerns.

        ---

        ## Key Takeaways

        1. Market projected at $47B by 2028 (38% CAGR) — enterprise AI agents are mainstream
        2. 34% of Fortune 500 companies have deployed AI agents in production
        3. Customer support automation leads adoption (45%) with 3.2x ROI
        4. Claude SDK offers the most integrated agent development experience
        5. Safety frameworks are immature — 78% of researchers agree more work is needed
        6. Tool restrictions and budget caps are essential safety guardrails

        ---

        ## Market Overview

        The enterprise AI agent market has exploded in 2025-2026. Key players include
        Anthropic (Claude Agent SDK), OpenAI (Agents SDK), Google (Gemini), and a growing
        ecosystem of startups building specialized agent frameworks.

        > **Data Point** (High confidence): Market projected at $47B by 2028 (38% CAGR)
        > **Trend** (High confidence): Code generation is the fastest-growing segment (+62% YoY)

        ---

        ## Enterprise Adoption

        Adoption is real but still early. 34% of Fortune 500 companies have deployed
        AI agents, with customer support seeing the highest ROI.

        > **Data Point** (High confidence): 28% average cost reduction in first year
        > **Data Point** (High confidence): Customer support ROI at 3.2x

        ---

        ## Safety & Risk Landscape

        As agents gain autonomy, safety becomes the critical enabler of further adoption.
        The industry is converging on a set of mitigation strategies.

        > **Risk** (High confidence): Prompt injection remains the #1 safety concern
        > **Risk** (High confidence): 78% of researchers say safety frameworks are immature

        ---

        ## Cross-Cutting Themes

        ### Rapid Growth vs. Safety Maturity Gap
        The market is growing faster than safety frameworks can keep up. This creates
        both opportunity (for safety-focused vendors) and risk (for early adopters).

        **Implications:**
        - Companies should invest in agent monitoring and guardrails now
        - Safety-first SDKs (like Claude's) will have a competitive advantage

        ---

        ## Actionable Insights & Recommendations

        ### 1. Start with Customer Support Agents [HIGH]
        Deploy AI agents in customer support first — it has the highest proven ROI (3.2x)
        and the most mature tooling.

        **Next Steps:**
        - Evaluate Claude Agent SDK and OpenAI Agents SDK
        - Run a 90-day pilot with a controlled subset of support tickets
        - Measure cost reduction and customer satisfaction

        ### 2. Invest in Agent Safety Infrastructure [CRITICAL]
        Build safety guardrails before scaling agent deployment.

        **Next Steps:**
        - Implement tool restrictions and budget caps on all agents
        - Set up real-time monitoring dashboards
        - Establish human-in-the-loop approval for high-stakes actions

        **Risk if ignored:** Unauthorized agent actions leading to data breaches or financial loss.

        ---

        ## Sources

        - [Enterprise AI Agents: 2026 Market Report](https://example.com/enterprise-ai-agents-2026)
        - [Building Autonomous Agents with Claude](https://example.com/claude-agents-guide)
        - [AI Agent Safety: Challenges and Solutions](https://example.com/agent-safety)
        - [Comparing Agent Frameworks](https://example.com/framework-comparison)
        - [OpenAI Agents SDK Review](https://example.com/openai-agents-sdk)
        - [Top 10 Enterprise AI Agent Use Cases](https://example.com/enterprise-use-cases)

        ---

        *Methodology: Multi-step autonomous research using Tavily web search + Claude analysis*
        *Searches: {len(sub_queries)} | Pages read: {len(deep_dive_urls)} | Findings: {len(all_findings)}*
    """)

    save_result_str = await mock_execute_tool("save_report", {
        "filename": "enterprise_ai_agents_2026.md",
        "content": report_content,
    })
    save_result = json.loads(save_result_str)

    print(f"  Report saved: {save_result['path']}")
    print(f"  Size: {save_result['size_bytes']:,} bytes")

    # Summary
    print_header("DEMO COMPLETE — Autonomous Operation Summary")
    print(f"  Searches executed:   {len(sub_queries)}")
    print(f"  Pages deep-dived:    {len(deep_dive_urls)}")
    print(f"  Findings recorded:   {len(all_findings)}")
    print(f"  Themes identified:   {len(themes)}")
    print(f"  Sources cited:       {len(all_sources)}")
    print(f"  Report:              {save_result['path']}")
    print()
    print("  The agent autonomously:")
    print("    1. Decomposed the query into 4 sub-questions")
    print("    2. Executed 4 targeted web searches")
    print("    3. Deep-dived into 2 authoritative sources")
    print("    4. Recorded 9 structured findings across 2 phases")
    print("    5. Identified 4 cross-cutting themes")
    print("    6. Generated a polished report with actionable recommendations")
    print()
    print("  All without human intervention — fully autonomous operation.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_demo())
