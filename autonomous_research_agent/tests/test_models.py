"""Tests for Pydantic data models — serialization, validation, markdown rendering."""

from __future__ import annotations

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


class TestSource:
    """Source model creation and serialization."""

    def test_create_with_defaults(self):
        s = Source(title="Test", url="https://example.com")
        assert s.title == "Test"
        assert s.snippet == ""
        assert s.relevance_score == 0.0

    def test_relevance_clamped(self):
        s = Source(title="T", url="https://x.com", relevance_score=0.95)
        assert 0.0 <= s.relevance_score <= 1.0

    def test_json_roundtrip(self):
        s = Source(title="Test", url="https://example.com", snippet="hello")
        data = s.model_dump(mode="json")
        s2 = Source.model_validate(data)
        assert s2.title == s.title
        assert s2.url == s.url


class TestFinding:
    """Finding model with type and confidence enums."""

    def test_default_type_and_confidence(self):
        f = Finding(headline="Test finding")
        assert f.finding_type == FindingType.FACT
        assert f.confidence == ConfidenceLevel.MEDIUM

    def test_with_sources(self, sample_source):
        f = Finding(
            headline="Important finding",
            sources=[sample_source],
            tags=["ai", "market"],
        )
        assert len(f.sources) == 1
        assert "ai" in f.tags

    def test_all_finding_types(self):
        for ft in FindingType:
            f = Finding(headline=f"Test {ft.value}", finding_type=ft)
            assert f.finding_type == ft


class TestQueryPlan:
    """Query decomposition model."""

    def test_empty_plan(self):
        plan = QueryPlan(original_query="Test query")
        assert plan.sub_queries == []
        assert plan.research_angles == []

    def test_with_sub_queries(self):
        plan = QueryPlan(
            original_query="AI agents landscape",
            sub_queries=[
                SubQuery(question="Who are the major players?", priority=1),
                SubQuery(question="What frameworks exist?", priority=2),
            ],
            estimated_searches=6,
        )
        assert len(plan.sub_queries) == 2
        assert plan.estimated_searches == 6


class TestSynthesis:
    """Cross-cutting synthesis model."""

    def test_with_contradictions(self):
        syn = Synthesis(
            theme="Market sizing",
            summary="Estimates vary widely.",
            contradictions=["Source A says $30B, Source B says $50B"],
            confidence=ConfidenceLevel.LOW,
        )
        assert len(syn.contradictions) == 1
        assert syn.confidence == ConfidenceLevel.LOW


class TestActionableInsight:
    """Actionable insight with priority levels."""

    def test_critical_priority(self):
        insight = ActionableInsight(
            title="Act now",
            description="Immediate action required.",
            priority=InsightPriority.CRITICAL,
            next_steps=["Step 1", "Step 2"],
            risks_if_ignored="Significant revenue loss.",
        )
        assert insight.priority == InsightPriority.CRITICAL
        assert len(insight.next_steps) == 2


class TestResearchReportMarkdown:
    """Report markdown rendering — the primary output format."""

    def test_markdown_contains_title(self, sample_report):
        md = sample_report.to_markdown()
        assert "# AI Agents Landscape Report" in md

    def test_markdown_contains_executive_summary(self, sample_report):
        md = sample_report.to_markdown()
        assert "## Executive Summary" in md
        assert "AI agents are evolving rapidly" in md

    def test_markdown_contains_key_takeaways(self, sample_report):
        md = sample_report.to_markdown()
        assert "## Key Takeaways" in md
        assert "Market growing at 40% CAGR" in md

    def test_markdown_contains_sections(self, sample_report):
        md = sample_report.to_markdown()
        assert "## Market Overview" in md

    def test_markdown_contains_findings_in_sections(self, sample_report):
        md = sample_report.to_markdown()
        assert "Trend" in md
        assert "High" in md

    def test_markdown_contains_cross_cutting_themes(self, sample_report):
        md = sample_report.to_markdown()
        assert "## Cross-Cutting Themes" in md
        assert "Convergence of agent frameworks" in md

    def test_markdown_contains_actionable_insights(self, sample_report):
        md = sample_report.to_markdown()
        assert "## Actionable Insights" in md
        assert "Invest in agent infrastructure" in md
        assert "[HIGH]" in md

    def test_markdown_contains_sources(self, sample_report):
        md = sample_report.to_markdown()
        assert "## Sources" in md
        assert "https://example.com/ai-agents" in md

    def test_markdown_contains_metadata(self, sample_report):
        md = sample_report.to_markdown()
        assert "Searches: 6" in md
        assert "Duration: 45.2s" in md

    def test_markdown_deduplicates_sources(self):
        src = Source(title="Dup", url="https://example.com/dup")
        report = ResearchReport(
            title="Test",
            query="test",
            all_sources=[src, src, src],
        )
        md = report.to_markdown()
        assert md.count("https://example.com/dup") == 1


class TestEvidence:
    """Evidence model linking claims to sources."""

    def test_evidence_with_quote(self, sample_source):
        ev = Evidence(
            claim="Agents reduce operational costs by 30%",
            source=sample_source,
            verbatim_quote="Our analysis shows a 30% reduction...",
            confidence=ConfidenceLevel.HIGH,
        )
        assert ev.verbatim_quote.startswith("Our analysis")
