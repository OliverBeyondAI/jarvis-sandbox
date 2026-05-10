"""Tests for tool schemas, dispatcher, and implementations."""

from __future__ import annotations

import json

import pytest

from autonomous_research_agent.tools import (
    ALL_TOOLS,
    analyze_findings,
    execute_tool,
    fetch_url,
    save_report,
    set_output_dir,
)


class TestToolSchemas:
    """Verify tool schemas are well-formed for the Claude Messages API."""

    def test_all_tools_count(self):
        assert len(ALL_TOOLS) == 4

    def test_all_tools_have_required_fields(self):
        for tool in ALL_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_all_tools_have_type_for_managed_api(self):
        for tool in ALL_TOOLS:
            assert tool.get("type") == "custom"

    def test_tool_names(self):
        names = {t["name"] for t in ALL_TOOLS}
        assert names == {"tavily_search", "fetch_url", "analyze_findings", "save_report"}

    def test_tavily_search_schema(self):
        tool = next(t for t in ALL_TOOLS if t["name"] == "tavily_search")
        props = tool["input_schema"]["properties"]
        assert "query" in props
        assert "max_results" in props
        assert "search_depth" in props
        assert "query" in tool["input_schema"]["required"]

    def test_fetch_url_schema(self):
        tool = next(t for t in ALL_TOOLS if t["name"] == "fetch_url")
        props = tool["input_schema"]["properties"]
        assert "url" in props
        assert "url" in tool["input_schema"]["required"]

    def test_analyze_findings_schema(self):
        tool = next(t for t in ALL_TOOLS if t["name"] == "analyze_findings")
        props = tool["input_schema"]["properties"]
        assert "phase" in props
        assert "findings" in props
        assert set(tool["input_schema"]["required"]) == {"phase", "findings"}

    def test_save_report_schema(self):
        tool = next(t for t in ALL_TOOLS if t["name"] == "save_report")
        props = tool["input_schema"]["properties"]
        assert "filename" in props
        assert "content" in props


class TestAnalyzeFindings:
    """Unit tests for the analyze_findings tool."""

    @pytest.mark.asyncio
    async def test_records_findings(self):
        result = await analyze_findings(
            phase="broad_search",
            findings=[
                {"headline": "Finding 1", "type": "fact"},
                {"headline": "Finding 2", "type": "trend"},
            ],
        )
        assert result["status"] == "recorded"
        assert result["findings_recorded"] == 2
        assert result["phase"] == "broad_search"

    @pytest.mark.asyncio
    async def test_records_gaps(self):
        result = await analyze_findings(
            phase="deep_dive",
            findings=[{"headline": "F1"}],
            gaps_identified=["Need pricing data", "Missing competitor analysis"],
        )
        assert len(result["gaps"]) == 2

    @pytest.mark.asyncio
    async def test_empty_findings(self):
        result = await analyze_findings(phase="synthesis", findings=[])
        assert result["findings_recorded"] == 0
        assert result["gaps"] == []


class TestSaveReport:
    """Unit tests for the save_report tool."""

    @pytest.mark.asyncio
    async def test_saves_markdown(self, tmp_path):
        result = await save_report(
            filename="test_report.md",
            content="# Test Report\n\nHello world.",
            output_dir=str(tmp_path),
        )
        assert result["saved"] is True
        assert result["size_bytes"] > 0

        saved_file = tmp_path / "test_report.md"
        assert saved_file.exists()
        assert saved_file.read_text().startswith("# Test Report")

    @pytest.mark.asyncio
    async def test_adds_md_extension(self, tmp_path):
        result = await save_report(
            filename="no_extension",
            content="content",
            output_dir=str(tmp_path),
        )
        assert result["path"].endswith(".md")

    @pytest.mark.asyncio
    async def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        result = await save_report(
            filename="report.md",
            content="content",
            output_dir=str(nested),
        )
        assert result["saved"] is True
        assert (nested / "report.md").exists()


class TestExecuteToolDispatcher:
    """Test the central tool dispatcher."""

    @pytest.mark.asyncio
    async def test_dispatch_analyze_findings(self):
        result_str = await execute_tool(
            "analyze_findings",
            {"phase": "broad_search", "findings": [{"headline": "test"}]},
        )
        result = json.loads(result_str)
        assert result["status"] == "recorded"

    @pytest.mark.asyncio
    async def test_dispatch_save_report(self, tmp_path):
        set_output_dir(str(tmp_path))
        result_str = await execute_tool(
            "save_report",
            {"filename": "dispatch_test.md", "content": "# Dispatched"},
        )
        result = json.loads(result_str)
        assert result["saved"] is True

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self):
        result_str = await execute_tool("nonexistent_tool", {})
        result = json.loads(result_str)
        assert "error" in result
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatch_handles_exceptions(self):
        # Pass invalid args to trigger an exception
        result_str = await execute_tool("analyze_findings", {"bad_key": "value"})
        result = json.loads(result_str)
        assert "error" in result
