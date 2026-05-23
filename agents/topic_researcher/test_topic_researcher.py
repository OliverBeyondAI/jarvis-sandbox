"""
Unit tests for the Topic Researcher agent.

Covers:
  - Tool schema validation
  - Tool dispatch (execute_tool routing)
  - Scratchpad accumulation
  - fetch_url truncation messaging
  - tavily_search error paths
  - JSON extraction / _parse_result logic (greedy-regex fix)
  - ResearchResult serialisation (to_dict, to_markdown)

Run:
    python -m pytest agents/topic_researcher/test_topic_researcher.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from .agent import TopicResearcher, ResearchResult, _extract_json_object
from .tools import (
    TOOLS,
    Scratchpad,
    execute_tool,
    fetch_url,
    save_notes,
    tavily_search,
    _strip_html,
)


# -----------------------------------------------------------------------
# Tool schema tests
# -----------------------------------------------------------------------

class TestToolSchemas:
    """Validate that every tool schema is well-formed."""

    def test_all_tools_have_required_fields(self):
        for tool in TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing 'description'"
            assert "input_schema" in tool, f"Tool {tool['name']} missing 'input_schema'"

    def test_input_schemas_are_objects(self):
        for tool in TOOLS:
            schema = tool["input_schema"]
            assert schema["type"] == "object", f"Tool {tool['name']} schema type != object"
            assert "properties" in schema, f"Tool {tool['name']} schema missing 'properties'"

    def test_tool_names_are_unique(self):
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_expected_tools_present(self):
        names = {t["name"] for t in TOOLS}
        assert "tavily_search" in names
        assert "fetch_url" in names
        assert "save_notes" in names


# -----------------------------------------------------------------------
# Scratchpad tests
# -----------------------------------------------------------------------

class TestScratchpad:
    def test_empty_dump(self):
        pad = Scratchpad()
        assert pad.dump() == "(no notes saved yet)"

    def test_add_and_dump(self):
        pad = Scratchpad()
        result = pad.add("First note", "search-1")
        assert "search-1" in result
        assert "1 total" in result
        assert "[1] search-1:" in pad.dump()

    def test_multiple_entries(self):
        pad = Scratchpad()
        pad.add("A", "alpha")
        pad.add("B", "beta")
        dump = pad.dump()
        assert "[1] alpha:" in dump
        assert "[2] beta:" in dump
        assert len(pad.entries) == 2


# -----------------------------------------------------------------------
# Tool implementation tests
# -----------------------------------------------------------------------

class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>hello</p>") == "hello"

    def test_removes_script_blocks(self):
        html = "<script>var x=1;</script><p>text</p>"
        assert "var x" not in _strip_html(html)

    def test_decodes_entities(self):
        result = _strip_html("&amp; &lt; &gt; &quot; &#39; &nbsp;")
        assert "&" in result
        assert "<" in result


class TestFetchUrl:
    def test_rejects_invalid_scheme(self):
        result = fetch_url("ftp://example.com")
        assert "Error" in result

    @patch("httpx.Client")
    def test_truncation_message(self, mock_client_cls):
        """fetch_url must inform the model when content is truncated."""
        long_text = "x" * 20_000
        mock_resp = MagicMock()
        mock_resp.text = long_text
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_url("https://example.com/big")
        assert "[Content truncated" in result
        assert "15,000" in result
        assert "20,000" in result

    @patch("httpx.Client")
    def test_no_truncation_for_short_content(self, mock_client_cls):
        short_text = "Hello world"
        mock_resp = MagicMock()
        mock_resp.text = short_text
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = fetch_url("https://example.com/small")
        assert result == "Hello world"
        assert "[Content truncated" not in result


class TestTavilySearch:
    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            results = tavily_search("test query")
            assert len(results) == 1
            assert "error" in results[0]

    def test_max_results_clamped(self):
        """max_results should be clamped to [1, 10]."""
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            with patch("tavily.TavilyClient") as mock_cls:
                mock_client = MagicMock()
                mock_client.search.return_value = {"results": []}
                mock_cls.return_value = mock_client

                tavily_search("test", max_results=99)
                call_kwargs = mock_client.search.call_args[1]
                assert call_kwargs["max_results"] == 10

                tavily_search("test", max_results=-5)
                call_kwargs = mock_client.search.call_args[1]
                assert call_kwargs["max_results"] == 1


class TestSaveNotes:
    def test_save_notes_adds_to_scratchpad(self):
        pad = Scratchpad()
        result = save_notes(pad, "some content", "label-1")
        assert "label-1" in result
        assert len(pad.entries) == 1


# -----------------------------------------------------------------------
# Tool dispatch tests
# -----------------------------------------------------------------------

class TestExecuteTool:
    def test_unknown_tool(self):
        result = execute_tool("nonexistent", {})
        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    def test_save_notes_dispatch(self):
        pad = Scratchpad()
        result = execute_tool(
            "save_notes",
            {"content": "hello", "label": "test"},
            scratchpad=pad,
        )
        assert "test" in result
        assert len(pad.entries) == 1

    def test_fetch_url_dispatch_invalid(self):
        result = execute_tool("fetch_url", {"url": "not-a-url"})
        assert "Error" in result

    @patch("httpx.Client")
    def test_fetch_url_dispatch_valid(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = "page content"
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = execute_tool("fetch_url", {"url": "https://example.com"})
        assert result == "page content"


# -----------------------------------------------------------------------
# JSON extraction tests (_extract_json_object)
# -----------------------------------------------------------------------

class TestExtractJsonObject:
    def test_plain_json(self):
        data = _extract_json_object('{"key": "value"}')
        assert data == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"summary": "hello"} and some more text.'
        data = _extract_json_object(text)
        assert data is not None
        assert data["summary"] == "hello"

    def test_multiple_json_objects_returns_first(self):
        """Must return the first JSON object, not greedily match everything."""
        text = '{"a": 1} {"b": 2}'
        data = _extract_json_object(text)
        assert data == {"a": 1}

    def test_nested_json(self):
        text = '{"outer": {"inner": true}}'
        data = _extract_json_object(text)
        assert data == {"outer": {"inner": True}}

    def test_json_with_braces_in_strings(self):
        text = '{"code": "if (x) { return }"}'
        data = _extract_json_object(text)
        assert data is not None
        assert "if (x)" in data["code"]

    def test_no_json(self):
        assert _extract_json_object("no json here") is None

    def test_empty_object(self):
        assert _extract_json_object("{}") == {}

    def test_json_in_markdown_fences(self):
        text = 'Some text {"summary": "test", "key_findings": ["a", "b"]}'
        data = _extract_json_object(text)
        assert data is not None
        assert data["key_findings"] == ["a", "b"]

    def test_escaped_quotes_in_strings(self):
        text = r'{"msg": "said \"hello\""}'
        data = _extract_json_object(text)
        assert data is not None
        assert "hello" in data["msg"]


# -----------------------------------------------------------------------
# ResearchResult serialisation tests
# -----------------------------------------------------------------------

class TestResearchResult:
    def test_to_dict(self):
        r = ResearchResult(
            topic="AI",
            summary="A summary.",
            key_findings=["F1", "F2"],
            sources=[{"title": "S", "url": "https://s.com"}],
        )
        d = r.to_dict()
        assert d["topic"] == "AI"
        assert len(d["key_findings"]) == 2
        assert "raw_notes" not in d

    def test_to_markdown(self):
        r = ResearchResult(
            topic="AI",
            summary="A summary.",
            key_findings=["F1"],
            sources=[{"title": "S", "url": "https://s.com"}],
        )
        md = r.to_markdown()
        assert "# Research: AI" in md
        assert "A summary." in md
        assert "- F1" in md
        assert "[S](https://s.com)" in md


# -----------------------------------------------------------------------
# _parse_result integration tests (via TopicResearcher)
# -----------------------------------------------------------------------

class TestParseResult:
    """Test _parse_result through the public TopicResearcher interface."""

    def _make_researcher(self) -> TopicResearcher:
        # Avoid hitting the Anthropic API
        with patch("anthropic.Anthropic"):
            return TopicResearcher()

    def test_parses_clean_json(self):
        researcher = self._make_researcher()
        pad = Scratchpad()
        text = json.dumps({
            "summary": "Some summary",
            "key_findings": ["A", "B"],
            "sources": [{"title": "T", "url": "https://t.com"}],
        })
        result = researcher._parse_result("topic", text, pad)
        assert result.summary == "Some summary"
        assert result.key_findings == ["A", "B"]

    def test_parses_json_in_markdown_fences(self):
        researcher = self._make_researcher()
        pad = Scratchpad()
        text = '```json\n{"summary": "fenced"}\n```'
        result = researcher._parse_result("topic", text, pad)
        assert result.summary == "fenced"

    def test_parses_json_with_surrounding_prose(self):
        researcher = self._make_researcher()
        pad = Scratchpad()
        text = 'Here is my report:\n{"summary": "embedded", "key_findings": []}\nDone!'
        result = researcher._parse_result("topic", text, pad)
        assert result.summary == "embedded"

    def test_handles_non_json_gracefully(self):
        researcher = self._make_researcher()
        pad = Scratchpad()
        result = researcher._parse_result("topic", "just plain text", pad)
        assert result.summary == "just plain text"

    def test_does_not_greedily_match_multiple_objects(self):
        """Regression: greedy regex {[\\s\\S]*} would over-match here."""
        researcher = self._make_researcher()
        pad = Scratchpad()
        text = '{"summary": "first"} some text {"other": "second"}'
        result = researcher._parse_result("topic", text, pad)
        assert result.summary == "first"
