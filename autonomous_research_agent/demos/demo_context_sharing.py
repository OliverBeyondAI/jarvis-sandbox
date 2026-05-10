#!/usr/bin/env python3
"""
Demo: Context Sharing Between Research Phases

Showcases how the agent maintains and shares context across phases:
  - Findings from Phase 1 inform Phase 2 search strategy
  - Sources are deduplicated and cross-referenced across phases
  - The AgentResult accumulates state across the entire run
  - The conversation history acts as shared memory between turns
  - Structured models ensure consistent data flow between phases

Runs entirely with mock data to illustrate the context-sharing pipeline.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from dataclasses import dataclass, field
from typing import Any

from autonomous_research_agent.agent import AgentResult
from autonomous_research_agent.models import Source


def print_header(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_context(label: str, data: dict[str, Any]) -> None:
    """Print a context snapshot showing accumulated state."""
    print(f"\n  [{label}]")
    for key, value in data.items():
        if isinstance(value, list):
            print(f"    {key}: {len(value)} items")
            for item in value[:3]:
                if isinstance(item, dict):
                    print(f"      • {item.get('headline', item.get('title', str(item)[:60]))}")
                elif isinstance(item, Source):
                    print(f"      • {item.title} ({item.url})")
                else:
                    print(f"      • {str(item)[:60]}")
            if len(value) > 3:
                print(f"      ... and {len(value) - 3} more")
        else:
            print(f"    {key}: {value}")


async def run_demo() -> None:
    """Demonstrate context sharing across research phases."""

    print_header("CONTEXT SHARING DEMO")
    print("  Showing how context flows between research phases")
    print()

    # Initialize the shared result object — this accumulates across phases
    result = AgentResult()
    conversation_context: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Phase 1: Initial Search — establishes baseline context
    # -----------------------------------------------------------------------
    print("─" * 70)
    print("  PHASE 1: Initial Search")
    print("─" * 70)

    phase1_query = "AI agent market size 2026"
    phase1_sources = [
        Source(title="Market Analysis: AI Agents 2026", url="https://example.com/market", relevance_score=0.95),
        Source(title="Gartner Magic Quadrant: AI Agents", url="https://example.com/gartner", relevance_score=0.92),
        Source(title="Startup Landscape: Agent Companies", url="https://example.com/startups", relevance_score=0.87),
    ]
    phase1_findings = [
        {"headline": "Market projected at $47B by 2028", "type": "data_point", "confidence": "high",
         "source_urls": ["https://example.com/market"]},
        {"headline": "Anthropic and OpenAI lead the market", "type": "fact", "confidence": "high",
         "source_urls": ["https://example.com/gartner"]},
    ]

    # Accumulate into shared result
    result.sources.extend(phase1_sources)
    result.search_count += 1
    result.findings.extend(phase1_findings)
    conversation_context.append({
        "phase": "initial_search",
        "query": phase1_query,
        "findings_count": len(phase1_findings),
        "sources_found": len(phase1_sources),
    })

    print(f"\n  Search: \"{phase1_query}\"")
    print(f"  Found: {len(phase1_sources)} sources, {len(phase1_findings)} findings")
    print_context("Accumulated Result State", {
        "sources": result.sources,
        "findings": result.findings,
        "search_count": result.search_count,
        "fetch_count": result.fetch_count,
    })

    # -----------------------------------------------------------------------
    # Phase 2: Adaptive Search — uses Phase 1 findings to guide next queries
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  PHASE 2: Adaptive Search (informed by Phase 1)")
    print("─" * 70)

    # The agent reviews Phase 1 findings and adapts its search strategy
    print("\n  Agent reasoning:")
    print("    \"Phase 1 found Anthropic and OpenAI lead the market.")
    print("     I should search specifically for their agent SDKs and")
    print("     also look for challenger frameworks I might be missing.\"")

    # Context from Phase 1 guides Phase 2 queries
    phase2_queries = [
        "Claude Agent SDK capabilities features",     # <- informed by Phase 1 finding
        "OpenAI Agents SDK comparison alternatives",   # <- informed by Phase 1 finding
        "LangGraph CrewAI challenger agent frameworks", # <- filling a gap
    ]

    phase2_sources = [
        Source(title="Claude Agent SDK Deep Dive", url="https://example.com/claude-sdk", relevance_score=0.94),
        Source(title="OpenAI Agents SDK Review", url="https://example.com/openai-sdk", relevance_score=0.90),
        # Duplicate source detected — agent should deduplicate
        Source(title="Gartner Magic Quadrant: AI Agents", url="https://example.com/gartner", relevance_score=0.92),
    ]
    phase2_findings = [
        {"headline": "Claude SDK offers built-in Bash, Read, Edit tools", "type": "fact", "confidence": "high",
         "source_urls": ["https://example.com/claude-sdk"]},
        {"headline": "OpenAI SDK focuses on guardrails and handoffs", "type": "fact", "confidence": "high",
         "source_urls": ["https://example.com/openai-sdk"]},
        {"headline": "LangGraph provides maximum flexibility but steeper learning curve", "type": "insight",
         "confidence": "medium", "source_urls": []},
    ]

    # Accumulate — but deduplicate sources by URL
    existing_urls = {s.url for s in result.sources}
    new_sources = [s for s in phase2_sources if s.url not in existing_urls]
    deduped_count = len(phase2_sources) - len(new_sources)

    result.sources.extend(new_sources)
    result.search_count += len(phase2_queries)
    result.findings.extend(phase2_findings)
    conversation_context.append({
        "phase": "adaptive_search",
        "queries": phase2_queries,
        "informed_by": "Phase 1 finding about Anthropic and OpenAI leadership",
        "sources_deduped": deduped_count,
    })

    for q in phase2_queries:
        print(f"  Search: \"{q}\"")
    print(f"\n  Found: {len(phase2_sources)} sources ({deduped_count} duplicate removed)")
    print(f"  New findings: {len(phase2_findings)}")

    print_context("Accumulated Result State", {
        "sources": result.sources,
        "findings": result.findings,
        "search_count": result.search_count,
        "fetch_count": result.fetch_count,
    })

    # -----------------------------------------------------------------------
    # Phase 3: Deep Dive — selects URLs from earlier phases
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  PHASE 3: Deep Dive (selects from Phase 1+2 sources)")
    print("─" * 70)

    # Agent picks the highest-relevance sources from ALL previous phases
    all_sources_sorted = sorted(result.sources, key=lambda s: s.relevance_score, reverse=True)
    top_sources = all_sources_sorted[:2]

    print("\n  Agent reasoning:")
    print("    \"I have 5 unique sources from Phases 1 and 2.")
    print("     Ranking by relevance score to pick the 2 best for deep dives:\"")
    for s in all_sources_sorted:
        marker = " ← selected" if s in top_sources else ""
        print(f"      [{s.relevance_score:.2f}] {s.title}{marker}")

    # Simulate deep dive
    deep_dive_findings = [
        {"headline": "Fortune 500 adoption at 34%, up from 12% in 2024", "type": "data_point",
         "confidence": "high", "source_urls": [top_sources[0].url]},
        {"headline": "Agent SDK market consolidating around 3 main players", "type": "trend",
         "confidence": "medium", "source_urls": [top_sources[1].url]},
    ]
    result.fetch_count += 2
    result.findings.extend(deep_dive_findings)
    conversation_context.append({
        "phase": "deep_dive",
        "sources_read": [s.title for s in top_sources],
        "selected_from": f"{len(result.sources)} total sources across Phases 1-2",
    })

    print(f"\n  Deep-dived into {len(top_sources)} sources")
    print(f"  New findings: {len(deep_dive_findings)}")

    print_context("Accumulated Result State", {
        "sources": result.sources,
        "findings": result.findings,
        "search_count": result.search_count,
        "fetch_count": result.fetch_count,
    })

    # -----------------------------------------------------------------------
    # Phase 4: Synthesis — uses ALL context from ALL phases
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  PHASE 4: Synthesis (cross-references ALL phases)")
    print("─" * 70)

    all_findings = result.findings
    print(f"\n  Agent has {len(all_findings)} findings from {len(conversation_context)} phases")
    print("  Cross-referencing findings by type:")

    by_type: dict[str, list[dict]] = {}
    for f in all_findings:
        t = f.get("type", "unknown")
        by_type.setdefault(t, []).append(f)

    for finding_type, items in by_type.items():
        print(f"    {finding_type}: {len(items)} findings")
        for item in items:
            print(f"      • {item['headline']}")

    # -----------------------------------------------------------------------
    # Show conversation context chain
    # -----------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  CONVERSATION CONTEXT CHAIN")
    print("─" * 70)
    print("\n  Each phase has access to all prior context via the message history:")

    for i, ctx in enumerate(conversation_context):
        phase = ctx["phase"]
        print(f"\n  Turn {i + 1}: {phase}")
        for k, v in ctx.items():
            if k != "phase":
                print(f"    {k}: {v}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_header("CONTEXT SHARING COMPLETE")
    print()
    print("  Context sharing mechanisms demonstrated:")
    print()
    print("  1. AgentResult Accumulation")
    print(f"     Sources: {len(result.sources)} (deduplicated by URL)")
    print(f"     Findings: {len(result.findings)} (across all phases)")
    print(f"     Searches: {result.search_count} | Fetches: {result.fetch_count}")
    print()
    print("  2. Adaptive Search Strategy")
    print("     Phase 2 queries were directly informed by Phase 1 findings")
    print("     (Anthropic/OpenAI leadership → SDK-specific searches)")
    print()
    print("  3. Cross-Phase Source Selection")
    print("     Deep dive selected sources from ALL prior phases by relevance")
    print("     (not just the most recent phase)")
    print()
    print("  4. Source Deduplication")
    print(f"     {deduped_count} duplicate source(s) detected and removed")
    print()
    print("  5. Conversation History as Memory")
    print(f"     {len(conversation_context)} context snapshots in message chain")
    print("     Each turn can reference all prior tool results and reasoning")
    print()
    print("  6. Structured Models for Data Flow")
    print("     Source, Finding, AgentResult models ensure type-safe context")
    print("     passing between phases with no data loss or format drift")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_demo())
