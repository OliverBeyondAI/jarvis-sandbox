"""
Tests for the Trend Research shared utilities.

Run offline by default (no API keys required). Use --live for end-to-end tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .agent import ResearchAgent, run_research
from .config import Config
from .models import (
    ResearchReport,
    Source,
    Trend,
    TrendAnalysis,
    TrendCategory,
)
from .s3_storage import S3Storage
from .tavily_client import TavilyResearchClient
from .tools import ALL_TOOLS, execute_tool


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.model == "claude-opus-4-7-20250501"
        assert config.max_tokens == 8192
        assert config.max_agent_turns == 25
        assert config.tavily_max_results == 10
        assert config.storage_local is True

    def test_validate_missing_keys(self):
        config = Config(tavily_api_key="", anthropic_api_key="")
        warnings = config.validate()
        assert len(warnings) >= 2
        assert any("TAVILY" in w for w in warnings)
        assert any("ANTHROPIC" in w for w in warnings)

    def test_validate_all_set(self):
        config = Config(tavily_api_key="test", anthropic_api_key="test")
        warnings = config.validate()
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_source_creation(self):
        s = Source(title="Test", url="https://example.com", snippet="content")
        assert s.title == "Test"
        assert s.relevance_score == 0.0

    def test_trend_defaults(self):
        t = Trend(name="AI Agents")
        assert t.category == TrendCategory.OTHER
        assert t.confidence == 0.5
        assert t.sources == []

    def test_trend_with_sources(self):
        s = Source(title="Source 1", url="https://example.com")
        t = Trend(
            name="AI Agents",
            category=TrendCategory.AI_ML,
            sources=[s],
            confidence=0.9,
        )
        assert len(t.sources) == 1
        assert t.category == TrendCategory.AI_ML

    def test_research_report_storage_key(self):
        r = ResearchReport(title="Test", domain="AI and ML")
        key = r.to_storage_key()
        assert "ai_and_ml" in key
        assert key.endswith("/report.json")

    def test_trend_analysis(self):
        a = TrendAnalysis(
            trend_name="Quantum Computing",
            opportunities=["Faster drug discovery"],
            risks=["Current hardware limitations"],
        )
        assert a.trend_name == "Quantum Computing"
        assert len(a.opportunities) == 1

    def test_report_serialization(self):
        report = ResearchReport(
            title="Test Report",
            domain="biotech",
            trends=[Trend(name="CRISPR 3.0")],
        )
        data = report.model_dump(mode="json")
        assert data["title"] == "Test Report"
        assert len(data["trends"]) == 1

        # Round-trip
        restored = ResearchReport.model_validate(data)
        assert restored.title == report.title


# ---------------------------------------------------------------------------
# S3 Storage tests (local mode)
# ---------------------------------------------------------------------------

class TestS3Storage:
    def _make_storage(self, tmp_dir: str) -> S3Storage:
        config = Config(storage_local=True, local_storage_dir=tmp_dir, s3_prefix="")
        return S3Storage(config)

    def test_store_and_load_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            data = {"hello": "world", "count": 42}

            path = asyncio.run(storage.store_json("test/data.json", data))
            assert Path(path).exists()

            loaded = asyncio.run(storage.load_json("test/data.json"))
            assert loaded == data

    def test_store_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            path = asyncio.run(storage.store_text("notes.txt", "hello world"))
            assert Path(path).exists()
            assert Path(path).read_text() == "hello world"

    def test_list_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            asyncio.run(storage.store_json("a/1.json", {"x": 1}))
            asyncio.run(storage.store_json("a/2.json", {"x": 2}))
            asyncio.run(storage.store_json("b/3.json", {"x": 3}))

            all_keys = asyncio.run(storage.list_keys())
            assert len(all_keys) == 3

            a_keys = asyncio.run(storage.list_keys("a"))
            assert len(a_keys) == 2

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = self._make_storage(tmp)
            assert asyncio.run(storage.exists("nope.json")) is False

            asyncio.run(storage.store_json("yes.json", {}))
            assert asyncio.run(storage.exists("yes.json")) is True

    def test_prefix_resolution(self):
        config = Config(storage_local=True, local_storage_dir="/tmp/test", s3_prefix="reports/")
        storage = S3Storage(config)
        assert storage._resolve_key("my/file.json") == "reports/my/file.json"


# ---------------------------------------------------------------------------
# Tool schema tests
# ---------------------------------------------------------------------------

class TestTools:
    def test_all_tools_have_required_fields(self):
        for tool in ALL_TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "type" in tool, f"Tool missing 'type': {tool.get('name')}"
            assert tool["type"] == "custom"
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_tool_names_unique(self):
        names = [t["name"] for t in ALL_TOOLS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_expected_tools_present(self):
        names = {t["name"] for t in ALL_TOOLS}
        expected = {"search_trends", "deep_dive", "search_news", "store_report", "fetch_url"}
        assert expected == names


# ---------------------------------------------------------------------------
# Category enum test
# ---------------------------------------------------------------------------

class TestTrendCategory:
    def test_all_categories(self):
        assert len(TrendCategory) >= 10
        assert TrendCategory.AI_ML.value == "ai_ml"
        assert TrendCategory.OTHER.value == "other"


# ---------------------------------------------------------------------------
# Research Agent tests
# ---------------------------------------------------------------------------

class TestResearchAgent:
    def test_agent_instantiation(self):
        config = Config(anthropic_api_key="test-key", tavily_api_key="test-key")
        agent = ResearchAgent(config)
        assert agent.config.anthropic_api_key == "test-key"

    def test_parse_findings_json_block(self):
        config = Config(anthropic_api_key="test", tavily_api_key="test")
        agent = ResearchAgent(config)

        text = '''Here are my findings:

```json
{
  "trend_name": "Multimodal AI",
  "category": "ai_ml",
  "summary": "AI systems that process multiple modalities",
  "significance": "Enables richer human-AI interaction",
  "confidence": 0.85,
  "sources": [{"title": "Paper 1", "url": "https://example.com", "snippet": "content", "relevance_score": 0.9}]
}
```'''
        findings = agent._parse_findings(text, "Multimodal AI")
        assert findings["trend_name"] == "Multimodal AI"
        assert findings["category"] == "ai_ml"
        assert findings["confidence"] == 0.85
        assert len(findings["sources"]) == 1

    def test_parse_findings_raw_json(self):
        config = Config(anthropic_api_key="test", tavily_api_key="test")
        agent = ResearchAgent(config)

        text = '{"trend_name": "Edge AI", "summary": "AI at the edge", "sources": []}'
        findings = agent._parse_findings(text, "Edge AI")
        assert findings["trend_name"] == "Edge AI"

    def test_parse_findings_fallback(self):
        config = Config(anthropic_api_key="test", tavily_api_key="test")
        agent = ResearchAgent(config)

        text = "This is just plain text with no JSON."
        findings = agent._parse_findings(text, "Some Trend")
        assert findings["trend_name"] == "Some Trend"
        assert findings["confidence"] == 0.3
        assert "raw_research" in findings

    def test_research_trend_mocked(self):
        """Test the agent loop with mocked Claude API and tools."""
        config = Config(anthropic_api_key="test-key", tavily_api_key="test-key")
        agent = ResearchAgent(config)

        # Mock the anthropic client to return a final text response (no tool use)
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [
            MagicMock(
                type="text",
                text=json.dumps({
                    "trend_name": "AI Agents",
                    "category": "ai_ml",
                    "summary": "Autonomous AI agents are becoming mainstream",
                    "significance": "Transforms how software is built and used",
                    "timeline": "2025-2027",
                    "key_players": ["Anthropic", "OpenAI", "Google"],
                    "current_state": "Early production deployments",
                    "opportunities": ["Automation", "Productivity"],
                    "risks": ["Safety", "Alignment"],
                    "predictions": ["Widespread adoption by 2027"],
                    "confidence": 0.8,
                    "sources": [
                        {"title": "AI Agents Overview", "url": "https://example.com/agents", "snippet": "...", "relevance_score": 0.9}
                    ],
                }),
            )
        ]

        with patch.object(agent, "_client", create=True) as mock_client:
            mock_client.messages.create.return_value = mock_response
            findings = asyncio.run(agent.research_trend("AI Agents"))

        assert findings["trend_name"] == "AI Agents"
        assert findings["category"] == "ai_ml"
        assert findings["confidence"] == 0.8
        assert "Anthropic" in findings["key_players"]

    def test_research_and_build_report_mocked(self):
        """Test that research_and_build_report produces a valid ResearchReport."""
        config = Config(anthropic_api_key="test-key", tavily_api_key="test-key")
        agent = ResearchAgent(config)

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [
            MagicMock(
                type="text",
                text=json.dumps({
                    "trend_name": "Neuromorphic Computing",
                    "category": "computing",
                    "summary": "Brain-inspired chip architectures",
                    "significance": "Orders of magnitude more energy efficient",
                    "timeline": "2026-2030",
                    "key_players": ["Intel", "IBM"],
                    "current_state": "Research phase",
                    "opportunities": ["Edge AI", "Low power"],
                    "risks": ["Software ecosystem immaturity"],
                    "predictions": ["Commercial chips by 2028"],
                    "confidence": 0.7,
                    "sources": [
                        {"title": "Neuromorphic Review", "url": "https://example.com/neuro", "snippet": "Overview", "relevance_score": 0.85}
                    ],
                }),
            )
        ]

        with patch.object(agent, "_client", create=True) as mock_client:
            mock_client.messages.create.return_value = mock_response
            report = asyncio.run(agent.research_and_build_report("Neuromorphic Computing"))

        assert isinstance(report, ResearchReport)
        assert report.title == "Research: Neuromorphic Computing"
        assert len(report.trends) == 1
        assert report.trends[0].category == TrendCategory.COMPUTING
        assert len(report.analyses) == 1
        assert "Edge AI" in report.analyses[0].opportunities
