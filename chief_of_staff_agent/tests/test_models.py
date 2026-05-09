"""Tests for Pydantic data models."""

from datetime import datetime, timezone

from chief_of_staff_agent.models import (
    ExecutiveReport,
    Finding,
    FindingType,
    ReportSection,
    ResearchBrief,
    ResearchPriority,
    ResearchQuestion,
    Source,
)


class TestSource:
    def test_create_with_defaults(self):
        s = Source(title="Test", url="https://example.com")
        assert s.title == "Test"
        assert s.url == "https://example.com"
        assert s.snippet == ""
        assert s.relevance_score == 0.0
        assert isinstance(s.fetched_at, datetime)

    def test_relevance_score_bounds(self):
        s = Source(title="T", url="https://x.com", relevance_score=0.95)
        assert s.relevance_score == 0.95


class TestFinding:
    def test_defaults(self):
        f = Finding(headline="AI adoption growing")
        assert f.finding_type == FindingType.FACT
        assert f.confidence == 0.7
        assert f.sources == []
        assert f.tags == []

    def test_with_sources(self):
        src = Source(title="Report", url="https://example.com")
        f = Finding(
            headline="Key trend",
            detail="Details here",
            finding_type=FindingType.TREND,
            sources=[src],
            tags=["ai", "market"],
        )
        assert len(f.sources) == 1
        assert f.tags == ["ai", "market"]


class TestResearchBrief:
    def test_minimal(self):
        b = ResearchBrief(topic="AI regulation")
        assert b.topic == "AI regulation"
        assert b.priority == ResearchPriority.STANDARD
        assert b.questions == []

    def test_with_questions(self):
        b = ResearchBrief(
            topic="Semiconductors",
            priority=ResearchPriority.HIGH,
            questions=[
                ResearchQuestion(question="What is the supply status?"),
                ResearchQuestion(question="Tariff impact?", context="US-China"),
            ],
            background="For Q3 board meeting",
        )
        assert len(b.questions) == 2
        assert b.background == "For Q3 board meeting"


class TestExecutiveReport:
    def test_to_markdown_minimal(self):
        report = ExecutiveReport(title="Test Report", topic="Testing")
        md = report.to_markdown()
        assert "# Test Report" in md
        assert "Generated:" in md

    def test_to_markdown_full(self):
        report = ExecutiveReport(
            title="AI Landscape",
            topic="AI",
            executive_summary="AI is transforming industries.",
            key_takeaways=["Takeaway 1", "Takeaway 2"],
            sections=[
                ReportSection(
                    title="Market Overview",
                    content="The market is growing.",
                    findings=[
                        Finding(
                            headline="Growth accelerating",
                            finding_type=FindingType.TREND,
                        )
                    ],
                )
            ],
            all_sources=[
                Source(title="Source 1", url="https://example.com/1"),
                Source(title="Source 2", url="https://example.com/2"),
            ],
        )
        md = report.to_markdown()
        assert "## Executive Summary" in md
        assert "AI is transforming industries." in md
        assert "1. Takeaway 1" in md
        assert "## Market Overview" in md
        assert "**Trend:**" in md
        assert "## Sources" in md
        assert "[Source 1](https://example.com/1)" in md

    def test_to_markdown_deduplicates_sources(self):
        report = ExecutiveReport(
            title="Test",
            topic="Test",
            all_sources=[
                Source(title="Same", url="https://example.com"),
                Source(title="Same", url="https://example.com"),
            ],
        )
        md = report.to_markdown()
        assert md.count("https://example.com") == 1
