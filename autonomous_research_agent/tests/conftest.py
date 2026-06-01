"""Shared fixtures for the autonomous research agent test suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from autonomous_research_agent.config import Config
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


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_config(tmp_path):
    """Config pointing to a temp output directory."""
    return Config(
        model="claude-opus-4-7-20250501",
        max_tokens=1024,
        max_agent_turns=5,
        tavily_api_key="test-tavily-key",
        anthropic_api_key="test-anthropic-key",
        output_dir=str(tmp_path / "reports"),
        min_searches=2,
        min_deep_dives=1,
    )


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_source():
    return Source(
        title="AI Agents Overview",
        url="https://example.com/ai-agents",
        snippet="AI agents are transforming enterprise software.",
        relevance_score=0.92,
    )


@pytest.fixture
def sample_finding(sample_source):
    return Finding(
        headline="AI agents market growing rapidly",
        detail="The market is expected to reach $50B by 2028.",
        finding_type=FindingType.TREND,
        confidence=ConfidenceLevel.HIGH,
        sources=[sample_source],
        tags=["market", "growth"],
    )


@pytest.fixture
def sample_report(sample_source, sample_finding):
    return ResearchReport(
        title="AI Agents Landscape Report",
        query="What is the current state of AI agents?",
        executive_summary="AI agents are evolving rapidly...",
        key_takeaways=[
            "Market growing at 40% CAGR",
            "Enterprise adoption accelerating",
            "Safety remains a key challenge",
        ],
        sections=[
            ReportSection(
                title="Market Overview",
                content="The AI agents market is expanding...",
                findings=[sample_finding],
            ),
        ],
        actionable_insights=[
            ActionableInsight(
                title="Invest in agent infrastructure",
                description="Build internal agent platform capabilities.",
                priority=InsightPriority.HIGH,
                rationale="Early movers will have a significant advantage.",
                next_steps=["Evaluate SDKs", "Run pilot project"],
                risks_if_ignored="Falling behind competitors.",
            ),
        ],
        syntheses=[
            Synthesis(
                theme="Convergence of agent frameworks",
                summary="Multiple frameworks are converging on similar patterns.",
                supporting_findings=["AI agents market growing rapidly"],
                implications=["Standardization is likely within 2 years"],
            ),
        ],
        all_sources=[sample_source],
        metadata={"search_count": 6, "fetch_count": 3, "duration_seconds": 45.2},
    )


# ---------------------------------------------------------------------------
# Mock API response helpers
# ---------------------------------------------------------------------------


def make_text_block(text: str):
    """Create a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_id: str, name: str, input_dict: dict[str, Any]):
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    block.text = ""
    return block


def make_response(content_blocks: list, stop_reason: str = "tool_use"):
    """Create a mock Claude API response."""
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    return resp


@pytest.fixture
def mock_search_result():
    """A realistic tavily_search result."""
    return json.dumps({
        "query": "AI agents enterprise 2026",
        "results": [
            {
                "title": "Enterprise AI Agents Report",
                "url": "https://example.com/report",
                "snippet": "AI agents are transforming how enterprises operate.",
                "relevance_score": 0.95,
            },
            {
                "title": "Building with Claude Agent SDK",
                "url": "https://example.com/claude-sdk",
                "snippet": "The Claude Agent SDK enables building autonomous agents.",
                "relevance_score": 0.88,
            },
        ],
        "result_count": 2,
    })


@pytest.fixture
def mock_fetch_result():
    """A realistic fetch_url result."""
    return json.dumps({
        "url": "https://example.com/report",
        "status": 200,
        "content_type": "text/html",
        "content": "AI agents are autonomous systems that can plan, reason, and act...",
        "length": 68,
    })


@pytest.fixture
def mock_analyze_result():
    """A realistic analyze_findings result."""
    return json.dumps({
        "phase": "broad_search",
        "findings_recorded": 2,
        "findings": [
            {"headline": "Market growing at 40% CAGR", "type": "trend", "confidence": "high"},
            {"headline": "Safety concerns remain", "type": "risk", "confidence": "medium"},
        ],
        "gaps": ["Need more data on enterprise adoption rates"],
        "status": "recorded",
    })


@pytest.fixture
def mock_save_result(tmp_path):
    """A realistic save_report result."""
    return json.dumps({
        "saved": True,
        "path": str(tmp_path / "reports" / "test_report.md"),
        "size_bytes": 5000,
    })
