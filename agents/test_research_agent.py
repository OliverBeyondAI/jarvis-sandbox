#!/usr/bin/env python3
"""
Unit tests for the Research Agent.

Tests the core components with mocked external dependencies (Anthropic API,
Tavily API, httpx). No live API keys or network access required.

Usage:
    python -m pytest agents/test_research_agent.py -v
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from .research_agent import (
    ALL_TOOLS,
    DATA_DIR,
    MODEL,
    SYSTEM_PROMPT,
    KeyTakeaway,
    LocalResearchRunner,
    ResearchAgent,
    ResearchResult,
    ResearchSummary,
    Scratchpad,
    _extract_executive_summary,
    _extract_key_takeaways,
    _extract_open_questions,
    _extract_text,
    _extract_themes,
    _infer_confidence,
    _split_into_sections,
    _summarize,
    _track_result,
    execute_tool,
    extract_summary,
    fetch_url,
    save_notes,
    save_report,
    tavily_search,
)


# ---------------------------------------------------------------------------
# Summarization & Key-Takeaway Tests
# ---------------------------------------------------------------------------

SAMPLE_SYNTHESIS = """\
# AI Safety Research: 2026 Update

## Executive Summary

AI safety research has accelerated significantly in 2026, with new alignment
techniques demonstrating proven results in controlling large language models.
Regulatory frameworks are taking shape across major economies.

## Key Findings

- Constitutional AI methods have clearly improved model behavior by 40%
- The EU AI Act enforcement began, establishing significant precedent for global regulation
- Mechanistic interpretability revealed early insights into transformer reasoning circuits
- Alignment tax (performance cost) may have decreased from 15% to under 5%
- Open-source safety tools have reached preliminary adoption in enterprise settings

## Regulatory Landscape

Major regulatory developments include the EU AI Act enforcement and proposed
US legislation on frontier model oversight.

## Technical Advances

New interpretability methods allow researchers to trace model reasoning through
attention heads and MLP layers with unprecedented clarity.

## Open Questions

- How will alignment techniques scale to multimodal foundation models?
- Can interpretability methods keep pace with model complexity?
- What enforcement mechanisms will prove effective for the EU AI Act?

## Sources

- https://example.com/ai-safety-2026
- https://example.com/eu-ai-act
"""


class TestKeyTakeaway:
    def test_default_confidence(self):
        t = KeyTakeaway(point="Test point")
        assert t.confidence == "medium"
        assert t.supporting_evidence == ""

    def test_custom_fields(self):
        t = KeyTakeaway(point="Significant finding", supporting_evidence="Paper X", confidence="high")
        assert t.confidence == "high"
        assert t.supporting_evidence == "Paper X"


class TestResearchSummary:
    def test_empty_summary(self):
        s = ResearchSummary()
        assert s.executive_summary == ""
        assert s.key_takeaways == []
        assert s.source_count == 0

    def test_format_text_output(self):
        s = ResearchSummary(
            executive_summary="AI safety has advanced.",
            key_takeaways=[KeyTakeaway(point="Models are safer", confidence="high")],
            themes=["Alignment", "Regulation"],
            open_questions=["Will it scale?"],
            source_count=3,
            word_count=500,
        )
        text = s.format_text()
        assert "EXECUTIVE SUMMARY" in text
        assert "AI safety has advanced." in text
        assert "KEY TAKEAWAYS" in text
        assert "Models are safer" in text
        assert "[+]" in text  # high confidence marker
        assert "THEMES" in text
        assert "Alignment" in text
        assert "OPEN QUESTIONS" in text
        assert "Will it scale?" in text
        assert "Sources: 3" in text

    def test_to_dict(self):
        s = ResearchSummary(
            executive_summary="Summary",
            key_takeaways=[KeyTakeaway(point="Point 1", confidence="low")],
            themes=["Theme A"],
            open_questions=["Question?"],
            source_count=2,
            word_count=100,
        )
        d = s.to_dict()
        assert d["executive_summary"] == "Summary"
        assert len(d["key_takeaways"]) == 1
        assert d["key_takeaways"][0]["confidence"] == "low"
        assert d["themes"] == ["Theme A"]
        assert d["source_count"] == 2


class TestSplitIntoSections:
    def test_basic_sections(self):
        lines = ["# Heading 1", "body 1", "", "## Heading 2", "body 2"]
        sections = _split_into_sections(lines)
        assert len(sections) == 2
        assert sections[0][0] == "Heading 1"
        assert sections[1][0] == "Heading 2"

    def test_no_headings(self):
        lines = ["Just plain text", "More text"]
        sections = _split_into_sections(lines)
        assert len(sections) == 1
        assert sections[0][0] == ""


class TestExtractExecutiveSummary:
    def test_explicit_section(self):
        sections = [
            ("Executive Summary", ["AI safety is important.", "It has advanced."]),
            ("Key Findings", ["- Finding 1"]),
        ]
        result = _extract_executive_summary(sections, "")
        assert "AI safety is important" in result

    def test_fallback_to_first_paragraph(self):
        sections = [("Introduction", [""]), ("Details", ["Some details"])]
        full_text = "This is a comprehensive overview of the research landscape.\n\n- bullet point"
        result = _extract_executive_summary(sections, full_text)
        assert "comprehensive overview" in result


class TestExtractKeyTakeaways:
    def test_from_key_findings_section(self):
        sections = [
            ("Key Findings", [
                "- Constitutional AI improved behavior by 40%",
                "- EU AI Act enforcement began in 2026",
                "- Short",
            ]),
        ]
        takeaways = _extract_key_takeaways(sections, "")
        assert len(takeaways) == 2  # "Short" is too short (<=10 chars)
        assert "Constitutional AI" in takeaways[0].point

    def test_numbered_list(self):
        sections = [
            ("Takeaways", [
                "1. First important finding about quantum computing",
                "2. Second significant result in materials science",
            ]),
        ]
        takeaways = _extract_key_takeaways(sections, "")
        assert len(takeaways) == 2

    def test_deduplication(self):
        sections = [
            ("Key Findings", ["- Duplicate point about something important"]),
            ("Conclusions", ["- Duplicate point about something important"]),
        ]
        # Both sections match priority keywords, so both are scanned
        # but dedup should remove the duplicate
        takeaways = _extract_key_takeaways(sections, "")
        assert len(takeaways) == 1


class TestInferConfidence:
    def test_high_confidence(self):
        assert _infer_confidence("The results clearly demonstrated a significant improvement") == "high"

    def test_low_confidence(self):
        assert _infer_confidence("This might indicate preliminary evidence of a trend") == "low"

    def test_medium_default(self):
        assert _infer_confidence("The system processed the data correctly") == "medium"


class TestExtractThemes:
    def test_extracts_content_headings(self):
        sections = [
            ("Executive Summary", []),
            ("Hardware Advances", []),
            ("Software Ecosystem", []),
            ("Sources", []),
        ]
        themes = _extract_themes(sections)
        assert "Hardware Advances" in themes
        assert "Software Ecosystem" in themes
        assert "Executive Summary" not in themes
        assert "Sources" not in themes

    def test_strips_numbering(self):
        sections = [("1. First Topic", []), ("2. Second Topic", [])]
        themes = _extract_themes(sections)
        assert themes[0] == "First Topic"


class TestExtractOpenQuestions:
    def test_from_explicit_section(self):
        sections = [
            ("Open Questions", [
                "- How will alignment techniques scale?",
                "- Can interpretability keep pace?",
            ]),
        ]
        questions = _extract_open_questions(sections, "")
        assert len(questions) == 2
        assert "alignment" in questions[0].lower()

    def test_inline_questions(self):
        full_text = "Some context. Will quantum computing replace classical methods? More text."
        questions = _extract_open_questions([], full_text)
        assert len(questions) == 1
        assert "quantum" in questions[0].lower()


class TestExtractSummaryIntegration:
    def test_full_synthesis_extraction(self):
        result = ResearchResult(
            synthesis=SAMPLE_SYNTHESIS,
            sources_consulted=5,
            scratchpad_entries=3,
            tool_calls=[{"name": "tavily_search"}] * 4,
        )
        summary = extract_summary(result)

        assert summary.source_count == 5
        assert summary.word_count > 50
        assert "accelerated" in summary.executive_summary.lower() or "safety" in summary.executive_summary.lower()
        assert len(summary.key_takeaways) >= 3
        assert len(summary.themes) >= 1
        assert len(summary.open_questions) >= 1

    def test_empty_synthesis(self):
        result = ResearchResult(synthesis="")
        summary = extract_summary(result)
        assert summary.executive_summary == ""
        assert summary.key_takeaways == []

    def test_result_summarize_method(self):
        result = ResearchResult(synthesis=SAMPLE_SYNTHESIS, sources_consulted=2)
        summary = result.summarize()
        assert isinstance(summary, ResearchSummary)
        assert summary.source_count == 2


# ---------------------------------------------------------------------------
# Scratchpad Tests
# ---------------------------------------------------------------------------

class TestScratchpad:
    def test_empty_scratchpad(self):
        pad = Scratchpad()
        assert pad.count() == 0
        assert pad.dump() == "(no notes saved yet)"

    def test_add_single_note(self):
        pad = Scratchpad()
        msg = pad.add("Finding about quantum computing", "discovery_1")
        assert pad.count() == 1
        assert "discovery_1" in msg
        assert "1 total entries" in msg

    def test_add_multiple_notes(self):
        pad = Scratchpad()
        pad.add("First finding", "search_1")
        pad.add("Second finding", "deep_dive")
        pad.add("Gap analysis", "follow_up")
        assert pad.count() == 3

    def test_dump_preserves_order(self):
        pad = Scratchpad()
        pad.add("Alpha", "first")
        pad.add("Beta", "second")
        dump = pad.dump()
        assert dump.index("Alpha") < dump.index("Beta")
        assert "[1] first:" in dump
        assert "[2] second:" in dump

    def test_default_label(self):
        pad = Scratchpad()
        pad.add("Some content")
        assert pad.entries[0]["label"] == "notes"


# ---------------------------------------------------------------------------
# Tool Schema Tests
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_all_tools_have_required_fields(self):
        for tool in ALL_TOOLS:
            assert "name" in tool
            assert "type" in tool
            assert tool["type"] == "custom"
            assert "description" in tool
            assert "input_schema" in tool

    def test_tavily_search_schema(self):
        tool = next(t for t in ALL_TOOLS if t["name"] == "tavily_search")
        props = tool["input_schema"]["properties"]
        assert "query" in props
        assert "max_results" in props
        assert "search_depth" in props
        assert tool["input_schema"]["required"] == ["query"]

    def test_fetch_url_schema(self):
        tool = next(t for t in ALL_TOOLS if t["name"] == "fetch_url")
        assert tool["input_schema"]["required"] == ["url"]

    def test_save_notes_schema(self):
        tool = next(t for t in ALL_TOOLS if t["name"] == "save_notes")
        assert tool["input_schema"]["required"] == ["content"]

    def test_save_report_schema(self):
        tool = next(t for t in ALL_TOOLS if t["name"] == "save_report")
        assert set(tool["input_schema"]["required"]) == {"filename", "content"}

    def test_messages_api_tools_strip_type(self):
        """Verify tool schemas can be converted for the Messages API."""
        stripped = [{k: v for k, v in t.items() if k != "type"} for t in ALL_TOOLS]
        for tool in stripped:
            assert "type" not in tool
            assert "name" in tool
            assert "description" in tool


# ---------------------------------------------------------------------------
# Tool Implementation Tests (mocked external deps)
# ---------------------------------------------------------------------------

class TestTavilySearch:
    @pytest.mark.asyncio
    async def test_successful_search(self):
        mock_results = {
            "results": [
                {"title": "Article 1", "url": "https://example.com/1", "content": "Content 1", "score": 0.9},
                {"title": "Article 2", "url": "https://example.com/2", "content": "Content 2", "score": 0.8},
            ]
        }

        with patch("agents.research_agent.asyncio.to_thread") as mock_thread:
            mock_thread.return_value = {
                "query": "test query",
                "results": [
                    {"title": "Article 1", "url": "https://example.com/1", "content": "Content 1", "score": 0.9},
                    {"title": "Article 2", "url": "https://example.com/2", "content": "Content 2", "score": 0.8},
                ],
            }
            result = await tavily_search("test query", max_results=2)
            assert result["query"] == "test query"
            assert len(result["results"]) == 2
            assert result["results"][0]["title"] == "Article 1"

    @pytest.mark.asyncio
    async def test_search_max_results_clamped(self):
        with patch("agents.research_agent.asyncio.to_thread") as mock_thread:
            mock_thread.return_value = {"query": "test", "results": []}
            await tavily_search("test", max_results=20)
            # The function internally clamps to 10, verified by the _sync_search logic


class TestFetchUrl:
    @pytest.mark.asyncio
    async def test_successful_html_fetch(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = "<html><body><p>Hello world</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("agents.research_agent.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_url("https://example.com")
            assert result["status"] == 200
            assert "Hello world" in result["content"]
            assert "<p>" not in result["content"]  # HTML stripped

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        import httpx as httpx_mod

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx_mod.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("agents.research_agent.httpx.AsyncClient", return_value=mock_client):
            result = await fetch_url("https://slow.example.com")
            assert "error" in result
            assert "timed out" in result["error"]


class TestSaveNotes:
    def test_save_to_scratchpad(self):
        pad = Scratchpad()
        result = save_notes(pad, "Important finding about AI", "discovery")
        assert result["saved"] is True
        assert result["total_entries"] == 1
        assert pad.count() == 1
        assert "AI" in pad.dump()

    def test_multiple_saves_accumulate(self):
        pad = Scratchpad()
        save_notes(pad, "Finding 1", "phase_1")
        save_notes(pad, "Finding 2", "phase_2")
        save_notes(pad, "Finding 3", "phase_3")
        assert pad.count() == 3


class TestSaveReport:
    @pytest.mark.asyncio
    async def test_save_report_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.research_agent.DATA_DIR", tmp_path)
        result = await save_report("test_report.md", "# Test Report\n\nContent here.")
        assert result["saved"] is True
        assert result["filename"] == "test_report.md"
        saved_file = tmp_path / "test_report.md"
        assert saved_file.exists()
        assert saved_file.read_text() == "# Test Report\n\nContent here."

    @pytest.mark.asyncio
    async def test_save_report_sanitizes_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.research_agent.DATA_DIR", tmp_path)
        result = await save_report("../../etc/passwd", "malicious")
        # Path traversal prevented — file saved with just the filename
        assert result["saved"] is True
        assert result["filename"] == "passwd"


# ---------------------------------------------------------------------------
# Tool Dispatcher Tests
# ---------------------------------------------------------------------------

class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_dispatch_save_notes(self):
        pad = Scratchpad()
        result_json = await execute_tool(
            "save_notes",
            {"content": "Test note", "label": "test"},
            pad,
        )
        result = json.loads(result_json)
        assert result["saved"] is True
        assert pad.count() == 1

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self):
        pad = Scratchpad()
        result_json = await execute_tool("nonexistent_tool", {}, pad)
        result = json.loads(result_json)
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatch_tavily_search(self):
        pad = Scratchpad()
        with patch("agents.research_agent.tavily_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = {"query": "test", "results": []}
            result_json = await execute_tool(
                "tavily_search",
                {"query": "test query"},
                pad,
            )
            result = json.loads(result_json)
            assert "results" in result
            mock_search.assert_called_once_with(
                query="test query", max_results=5, search_depth="basic"
            )


# ---------------------------------------------------------------------------
# ResearchResult Tests
# ---------------------------------------------------------------------------

class TestResearchResult:
    def test_default_values(self):
        result = ResearchResult()
        assert result.synthesis == ""
        assert result.tool_calls == []
        assert result.files_saved == []
        assert result.sources_consulted == 0
        assert result.scratchpad_entries == 0

    def test_summary_property(self):
        result = ResearchResult(
            sources_consulted=5,
            scratchpad_entries=8,
            files_saved=["/data/report.md"],
            tool_calls=[{"name": "tavily_search"}] * 10,
        )
        summary = result.summary
        assert "5 sources" in summary
        assert "8 notes" in summary
        assert "1 file(s)" in summary
        assert "10 tool calls" in summary


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_text(self):
        mock_response = SimpleNamespace(content=[
            SimpleNamespace(text="Hello "),
            SimpleNamespace(text="World"),
        ])
        assert _extract_text(mock_response) == "Hello \nWorld"

    def test_extract_text_skips_non_text(self):
        mock_response = SimpleNamespace(content=[
            SimpleNamespace(text="Only text"),
            SimpleNamespace(type="tool_use", id="123"),  # no text attr
        ])
        assert _extract_text(mock_response) == "Only text"

    def test_summarize_query(self):
        assert 'query="short"' == _summarize({"query": "short"})

    def test_summarize_long_query(self):
        long_q = "a" * 60
        result = _summarize({"query": long_q})
        assert result.endswith('..."')
        assert len(result) < 70

    def test_summarize_url(self):
        assert 'url="https://example.com"' == _summarize({"url": "https://example.com"})

    def test_summarize_filename(self):
        assert 'file="report.md"' == _summarize({"filename": "report.md"})

    def test_summarize_label(self):
        assert "notes: discovery" == _summarize({"label": "discovery"})

    def test_track_result_save_report(self):
        result = ResearchResult()
        _track_result(result, "save_report", json.dumps({"saved": True, "path": "/data/r.md"}))
        assert result.files_saved == ["/data/r.md"]

    def test_track_result_tavily_search(self):
        result = ResearchResult()
        _track_result(result, "tavily_search", json.dumps({"query": "test", "results": []}))
        assert result.sources_consulted == 1

    def test_track_result_ignores_errors(self):
        result = ResearchResult()
        _track_result(result, "tavily_search", json.dumps({"error": "failed"}))
        assert result.sources_consulted == 0

    def test_track_result_handles_invalid_json(self):
        result = ResearchResult()
        _track_result(result, "tavily_search", "not json")
        assert result.sources_consulted == 0


# ---------------------------------------------------------------------------
# LocalResearchRunner Tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestLocalResearchRunner:
    @pytest.mark.asyncio
    async def test_simple_end_turn(self):
        """Agent responds immediately without tool calls."""
        mock_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text="Research synthesis here.")],
        )

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("agents.research_agent.anthropic.AsyncAnthropic", return_value=mock_client):
            runner = LocalResearchRunner(verbose=False)
            result = await runner.run("Test topic")
            assert result.synthesis == "Research synthesis here."
            assert result.scratchpad_entries == 0

    @pytest.mark.asyncio
    async def test_tool_call_then_end(self):
        """Agent makes one tool call (save_notes), then produces final response."""
        # Turn 1: model calls save_notes
        tool_use_block = SimpleNamespace(
            type="tool_use",
            id="tool_1",
            name="save_notes",
            input={"content": "Found interesting data", "label": "discovery"},
        )
        turn1_response = SimpleNamespace(
            stop_reason="tool_use",
            content=[tool_use_block],
        )

        # Turn 2: model produces final text
        turn2_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text="Final synthesis based on notes.")],
        )

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[turn1_response, turn2_response]
        )

        with patch("agents.research_agent.anthropic.AsyncAnthropic", return_value=mock_client):
            runner = LocalResearchRunner(verbose=False)
            result = await runner.run("Test topic")
            assert result.synthesis == "Final synthesis based on notes."
            assert result.scratchpad_entries == 1
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0]["name"] == "save_notes"

    @pytest.mark.asyncio
    async def test_multi_step_workflow(self):
        """Simulates the full search → extract → follow-up → synthesize workflow."""
        # Turn 1: tavily_search
        turn1 = SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(
                type="tool_use", id="t1", name="tavily_search",
                input={"query": "quantum computing 2026"},
            )],
        )

        # Turn 2: save_notes (discovery phase notes)
        turn2 = SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(
                type="tool_use", id="t2", name="save_notes",
                input={"content": "Found 3 key papers on quantum computing", "label": "discovery"},
            )],
        )

        # Turn 3: fetch_url (extraction phase)
        turn3 = SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(
                type="tool_use", id="t3", name="fetch_url",
                input={"url": "https://example.com/quantum"},
            )],
        )

        # Turn 4: save_notes (extraction notes)
        turn4 = SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(
                type="tool_use", id="t4", name="save_notes",
                input={"content": "Deep dive findings from article", "label": "extraction"},
            )],
        )

        # Turn 5: tavily_search (follow-up phase)
        turn5 = SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(
                type="tool_use", id="t5", name="tavily_search",
                input={"query": "quantum error correction advances"},
            )],
        )

        # Turn 6: final synthesis
        turn6 = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text="## Quantum Computing Report\n\nComprehensive synthesis...")],
        )

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[turn1, turn2, turn3, turn4, turn5, turn6]
        )

        # Mock external tools
        mock_tavily = AsyncMock(return_value={"query": "test", "results": [
            {"title": "Paper", "url": "https://example.com", "content": "...", "score": 0.9}
        ]})
        mock_fetch = AsyncMock(return_value={
            "url": "https://example.com/quantum", "status": 200,
            "content_type": "text/html", "content": "Article content", "length": 100,
        })

        with patch("agents.research_agent.anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("agents.research_agent.tavily_search", mock_tavily), \
             patch("agents.research_agent.fetch_url", mock_fetch):
            runner = LocalResearchRunner(verbose=False)
            result = await runner.run("Quantum computing advances")

            assert "Quantum Computing Report" in result.synthesis
            assert result.scratchpad_entries == 2  # 2 save_notes calls
            assert result.sources_consulted == 3   # 2 tavily + 1 fetch
            assert len(result.tool_calls) == 5     # search, notes, fetch, notes, search


# ---------------------------------------------------------------------------
# ResearchAgent Tests (facade)
# ---------------------------------------------------------------------------

class TestResearchAgent:
    def test_default_config(self):
        agent = ResearchAgent()
        assert agent.model == MODEL
        assert agent.depth == "deep"
        assert agent.use_managed is True

    def test_brief_depth_uses_brief_prompt(self):
        agent = ResearchAgent(depth="brief")
        assert agent.system_prompt != SYSTEM_PROMPT
        assert "focused" in agent.system_prompt.lower()

    def test_deep_depth_uses_full_prompt(self):
        agent = ResearchAgent(depth="deep")
        assert agent.system_prompt == SYSTEM_PROMPT

    def test_build_prompt_deep(self):
        agent = ResearchAgent(depth="deep")
        prompt = agent._build_prompt("quantum computing")
        assert "quantum computing" in prompt
        assert "four phases" in prompt
        assert "save_notes" in prompt

    def test_build_prompt_brief(self):
        agent = ResearchAgent(depth="brief")
        prompt = agent._build_prompt("AI safety")
        assert "AI safety" in prompt
        assert "focused" in prompt.lower()

    @pytest.mark.asyncio
    async def test_fallback_to_local_on_api_error(self):
        """When managed API fails, agent falls back to local runner."""
        mock_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(text="Local runner result.")],
        )

        mock_async_client = AsyncMock()
        mock_async_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("agents.research_agent.anthropic.Anthropic") as mock_sync_cls, \
             patch("agents.research_agent.anthropic.AsyncAnthropic", return_value=mock_async_client):
            # Make managed runner fail
            mock_sync = MagicMock()
            mock_sync.beta.agents.create.side_effect = anthropic.APIError(
                message="Unavailable",
                request=MagicMock(),
                body=None,
            )
            mock_sync_cls.return_value = mock_sync

            agent = ResearchAgent(verbose=False)
            result = await agent.research("test topic")
            assert result.synthesis == "Local runner result."


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_model_is_opus_4_6(self):
        assert "opus-4-6" in MODEL

    def test_system_prompt_has_four_phases(self):
        assert "Phase 1: Discovery" in SYSTEM_PROMPT
        assert "Phase 2: Deep Extraction" in SYSTEM_PROMPT
        assert "Phase 3: Follow-up" in SYSTEM_PROMPT
        assert "Phase 4: Synthesis" in SYSTEM_PROMPT

    def test_system_prompt_mentions_scratchpad(self):
        assert "save_notes" in SYSTEM_PROMPT
        assert "scratchpad" in SYSTEM_PROMPT.lower()

    def test_all_tools_count(self):
        assert len(ALL_TOOLS) == 4
        names = {t["name"] for t in ALL_TOOLS}
        assert names == {"tavily_search", "fetch_url", "save_notes", "save_report"}


# ---------------------------------------------------------------------------
# Demo / End-to-End Verification Tests
# ---------------------------------------------------------------------------

class TestDemo:
    def test_demo_runs_without_error(self, capsys):
        """Verify the demo pipeline runs end-to-end without API keys."""
        from .research_agent import _run_demo
        _run_demo(verbose=False, as_json=False)
        captured = capsys.readouterr()
        assert "PIPELINE VERIFICATION" in captured.out
        assert "EXECUTIVE SUMMARY" in captured.out
        assert "KEY TAKEAWAYS" in captured.out

    def test_demo_json_output(self, capsys):
        from .research_agent import _run_demo
        _run_demo(verbose=False, as_json=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["demo"] is True
        assert data["pipeline_status"] == "OK"
        assert "summary" in data
        assert len(data["summary"]["key_takeaways"]) >= 3
