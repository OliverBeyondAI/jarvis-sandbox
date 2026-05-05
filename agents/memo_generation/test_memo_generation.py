"""
Tests for the Memo Generation Agent.

Validates models, storage, HTML conversion, and the memo generation pipeline
without requiring live API calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
from ..trend_research.models import ResearchReport, Source, Trend, TrendAnalysis, TrendCategory
from .agent import MemoGenerationAgent
from .config import MemoConfig
from .models import ArtifactBundle, InternalMemo, MemoAudience, MemoSection
from .storage import MemoStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memo_config(tmp_path: Path) -> MemoConfig:
    """Create a test config with local storage."""
    return MemoConfig(
        anthropic_api_key="test-key",
        storage_local=True,
        local_storage_dir=str(tmp_path / "memo_output"),
        model="claude-opus-4-7-20250501",
        max_tokens=4096,
        max_agent_turns=3,
    )


@pytest.fixture
def sample_research_report() -> ResearchReport:
    """Minimal research report for testing."""
    return ResearchReport(
        title="Test Research — Q2 2026",
        domain="healthcare AI",
        executive_summary="Two major trends identified.",
        trends=[
            Trend(
                name="Agentic AI",
                category=TrendCategory.AI_ML,
                summary="AI agents for healthcare admin.",
                confidence=0.9,
            ),
        ],
        analyses=[
            TrendAnalysis(
                trend_name="Agentic AI",
                current_state="Production-ready.",
                opportunities=["PA automation"],
                risks=["Error propagation"],
            ),
        ],
    )


@pytest.fixture
def sample_synthesis_report() -> SynthesisReport:
    """Minimal synthesis report for testing."""
    return SynthesisReport(
        title="Synthesis: Test Research — Q2 2026",
        research_source="Test Research — Q2 2026",
        executive_summary="Agentic AI is the top priority for OphthoFlow.",
        trend_syntheses=[
            TrendSynthesis(
                trend_name="Agentic AI",
                relevance_summary="Core to PA automation.",
                maturity_assessment="Production-ready.",
                competitive_landscape="Competitive but specialization is a moat.",
                applications=[
                    ApplicationIdea(
                        title="Autonomous PA Agent",
                        description="End-to-end PA submission.",
                        platform=Platform.OPHTHOFLOW,
                        fit_level=FitLevel.HIGH,
                        impact=ImpactLevel.TRANSFORMATIVE,
                        effort=EffortLevel.MEDIUM,
                        use_case="Automated PA workflow.",
                        user_benefit="PA in hours not days.",
                    ),
                ],
                overall_priority=FitLevel.HIGH,
            ),
        ],
        strategic_themes=[
            StrategicTheme(
                name="Agentic Architecture",
                description="Foundation for all products.",
                contributing_trends=["Agentic AI"],
                strategic_implications=["Invest in agent framework"],
            ),
        ],
        top_opportunities=[
            ApplicationIdea(
                title="Autonomous PA Agent",
                description="End-to-end PA.",
                platform=Platform.OPHTHOFLOW,
                fit_level=FitLevel.HIGH,
                impact=ImpactLevel.TRANSFORMATIVE,
                effort=EffortLevel.MEDIUM,
            ),
        ],
        key_risks=["Error propagation in autonomous workflows"],
        recommended_next_steps=["Prototype PA agent immediately"],
    )


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestModels:
    """Test data model creation and serialization."""

    def test_internal_memo_creation(self):
        memo = InternalMemo(
            title="Test Memo",
            subtitle="Q2 Findings",
            audience=MemoAudience.PRODUCT,
            tldr="Key findings summary.",
        )
        assert memo.title == "Test Memo"
        assert memo.audience == MemoAudience.PRODUCT
        assert memo.distribution == "Internal — Do Not Distribute"

    def test_memo_section_nesting(self):
        section = MemoSection(
            heading="Top Level",
            content="Main content.",
            subsections=[
                MemoSection(heading="Sub A", content="Detail A."),
                MemoSection(heading="Sub B", content="Detail B."),
            ],
        )
        assert len(section.subsections) == 2

    def test_artifact_bundle_creation(self):
        bundle = ArtifactBundle(
            bundle_id="run-20260505",
            research_report_path="/path/to/research.json",
            synthesis_report_path="/path/to/synthesis.json",
            memo_path="/path/to/memo.md",
            trends_analyzed=5,
            applications_identified=12,
        )
        assert bundle.bundle_id == "run-20260505"
        assert bundle.trends_analyzed == 5

    def test_artifact_bundle_storage_key(self):
        bundle = ArtifactBundle(bundle_id="test")
        key = bundle.to_storage_key()
        assert key.startswith("bundles/")
        assert key.endswith("/manifest.json")

    def test_internal_memo_storage_key(self):
        memo = InternalMemo(title="Test")
        key = memo.to_storage_key("markdown")
        assert key.endswith(".md")
        key_html = memo.to_storage_key("html")
        assert key_html.endswith(".html")

    def test_memo_serialization(self):
        memo = InternalMemo(
            title="Serialization Test",
            tldr="Testing JSON round-trip.",
            sections=[MemoSection(heading="Intro", content="Hello.")],
        )
        data = memo.model_dump(mode="json")
        restored = InternalMemo(**data)
        assert restored.title == "Serialization Test"
        assert len(restored.sections) == 1


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        config = MemoConfig(anthropic_api_key="test")
        assert config.model == "claude-opus-4-7-20250501"
        assert config.memo_format == "markdown"
        assert config.storage_local is True

    def test_validation_missing_key(self):
        config = MemoConfig(anthropic_api_key="")
        warnings = config.validate()
        assert any("ANTHROPIC_API_KEY" in w for w in warnings)

    def test_validation_passes(self):
        config = MemoConfig(anthropic_api_key="sk-test")
        warnings = config.validate()
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Storage Tests
# ---------------------------------------------------------------------------


class TestStorage:
    """Test local storage operations."""

    def test_store_all_outputs(self, memo_config: MemoConfig):
        storage = MemoStorage(memo_config)
        bundle = asyncio.run(storage.store_all_outputs(
            research_data={"title": "Research", "trends": [{"name": "Trend A"}]},
            synthesis_data={
                "title": "Synthesis",
                "executive_summary": "Summary",
                "trend_syntheses": [{"applications": [1, 2]}],
                "strategic_themes": [],
            },
            memo_markdown="# Memo\n\nContent here.",
            memo_html="<h1>Memo</h1><p>Content here.</p>",
            memo_title="Test Memo",
        ))
        assert bundle.bundle_id.startswith("run-")
        assert bundle.research_title == "Research"
        assert bundle.memo_title == "Test Memo"
        assert bundle.trends_analyzed == 1
        assert bundle.applications_identified == 2

        # Verify files exist on disk
        output_dir = Path(memo_config.local_storage_dir)
        assert output_dir.exists()
        json_files = list(output_dir.rglob("*.json"))
        assert len(json_files) >= 3  # research, synthesis, manifest
        md_files = list(output_dir.rglob("*.md"))
        assert len(md_files) >= 1
        html_files = list(output_dir.rglob("*.html"))
        assert len(html_files) >= 1


# ---------------------------------------------------------------------------
# Agent Tests (mocked Claude)
# ---------------------------------------------------------------------------


MOCK_MEMO_RESPONSE = """---
TO: Product & Engineering Leadership
FROM: AI Research Pipeline
DATE: 2026-05-05
RE: Healthcare AI Trends — Strategic Opportunities
CLASSIFICATION: Internal — Do Not Distribute
---

## TL;DR

Agentic AI for healthcare administration represents the highest-priority opportunity for OphthoFlow, with potential to reduce PA turnaround from days to hours. Immediate action recommended on PA agent prototype.

## Context & Methodology

Research was conducted using a multi-agent pipeline analyzing emerging healthcare AI trends. Five trends were evaluated for relevance to OphthoFlow and Xena platforms.

## Key Findings

- **Agentic AI** is production-ready and directly aligned with OphthoFlow's core mission
- **Multimodal foundation models** offer transformative potential for ophthalmology diagnostics (12-18 month horizon)
- **LLM-based CDS** can enhance both coding assistance (OphthoFlow) and clinical recommendations (Xena)

## Strategic Opportunities

### 1. Autonomous PA Submission Agent (OphthoFlow)
- **What**: End-to-end agent handling PA from data gathering through approval
- **Impact**: Transformative — PA turnaround from 14 days to <24 hours
- **Effort**: Medium (weeks to a month)
- **Next step**: Prototype with top-3 payer portals by end of Q2

## Quick Wins

- **LLM-Powered Coding Assistant**: Ship beta in Q3 2026 (low effort, high impact)

## Moonshots

- **AI-Assisted OCT Interpretation**: Requires FDA pathway but transformative if achieved

## Risk Assessment

- Payer resistance to AI-generated submissions
- LLM hallucination risk requires robust guardrails

## Recommended Next Steps

1. Immediately: Prototype autonomous PA submission agent
2. Q3 2026: Ship LLM-powered coding assistant beta
3. Q4 2026: Begin ophthalmology imaging AI evaluation

## Appendix: Trend Details

### Agentic AI for Healthcare Administration
Production-ready technology with 40-60% reduction in administrative burden. Directly aligned with OphthoFlow's PA automation mission. Competition is intensifying but ophthalmology specialization provides defensible moat.
"""


class TestMemoAgent:
    """Test the MemoGenerationAgent with mocked Claude responses."""

    def test_generate_memo(self, memo_config: MemoConfig, sample_synthesis_report: SynthesisReport):
        agent = MemoGenerationAgent(memo_config)

        # Mock the Anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text=MOCK_MEMO_RESPONSE)]
        mock_response.stop_reason = "end_turn"

        with patch.object(agent, "_client") as mock_client:
            mock_client.messages.create.return_value = mock_response
            memo = asyncio.run(agent.generate_memo(sample_synthesis_report))

        assert "TL;DR" in memo or "tldr" in memo.lower()
        assert "OphthoFlow" in memo
        assert "PA" in memo

    def test_markdown_to_html(self, memo_config: MemoConfig):
        agent = MemoGenerationAgent(memo_config)
        html = agent.markdown_to_html(MOCK_MEMO_RESPONSE, title="Test Memo")

        assert "<!DOCTYPE html>" in html
        assert "Test Memo" in html
        assert "Product & Engineering Leadership" in html
        assert "</html>" in html

    def test_full_pipeline(
        self,
        memo_config: MemoConfig,
        sample_research_report: ResearchReport,
        sample_synthesis_report: SynthesisReport,
    ):
        agent = MemoGenerationAgent(memo_config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text=MOCK_MEMO_RESPONSE)]
        mock_response.stop_reason = "end_turn"

        with patch.object(agent, "_client") as mock_client:
            mock_client.messages.create.return_value = mock_response
            bundle = asyncio.run(agent.run_full_pipeline(
                research_report=sample_research_report,
                synthesis_report=sample_synthesis_report,
            ))

        assert bundle.bundle_id.startswith("run-")
        assert bundle.research_title == "Test Research — Q2 2026"
        assert bundle.synthesis_title == "Synthesis: Test Research — Q2 2026"

        # Verify files were stored
        output_dir = Path(memo_config.local_storage_dir)
        assert output_dir.exists()
        all_files = list(output_dir.rglob("*"))
        file_names = [f.name for f in all_files if f.is_file()]
        assert "research_report.json" in file_names
        assert "synthesis_report.json" in file_names
        assert "memo.md" in file_names
        assert "memo.html" in file_names
        assert "manifest.json" in file_names


# ---------------------------------------------------------------------------
# HTML Conversion Tests
# ---------------------------------------------------------------------------


class TestHTMLConversion:
    """Test the Markdown-to-HTML conversion logic."""

    def test_headings(self):
        agent = MemoGenerationAgent(MemoConfig(anthropic_api_key="test"))
        html = agent._convert_md_to_html("## Section\n\n### Subsection\n\nText.")
        assert "<h2>Section</h2>" in html
        assert "<h3>Subsection</h3>" in html
        assert "<p>Text.</p>" in html

    def test_lists(self):
        agent = MemoGenerationAgent(MemoConfig(anthropic_api_key="test"))
        html = agent._convert_md_to_html("- Item 1\n- Item 2\n\n1. First\n2. Second")
        assert "<ul>" in html
        assert "<li>Item 1</li>" in html
        assert "<ol>" in html
        assert "<li>First</li>" in html

    def test_inline_formatting(self):
        agent = MemoGenerationAgent(MemoConfig(anthropic_api_key="test"))
        result = agent._inline_format("**bold** and *italic* and `code`")
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result
        assert "<code>code</code>" in result

    def test_tldr_styling(self):
        agent = MemoGenerationAgent(MemoConfig(anthropic_api_key="test"))
        md = "## TL;DR\n\nThis is the key takeaway.\n\n## Next Section\n\nMore content."
        html = agent._convert_md_to_html(md)
        assert 'class="tldr"' in html
        assert "key takeaway" in html
