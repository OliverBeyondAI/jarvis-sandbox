"""Tests for AutonomousResearchAgent — multi-step reasoning with mocked API."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autonomous_research_agent.agent import AutonomousResearchAgent
from autonomous_research_agent.config import Config

from .conftest import make_response, make_text_block, make_tool_use_block


@pytest.fixture
def agent(test_config):
    return AutonomousResearchAgent(config=test_config)


class TestAgentInit:
    """Agent initialization and config."""

    def test_creates_with_default_config(self):
        agent = AutonomousResearchAgent()
        assert agent.config.model == "claude-opus-4-7-20250501"

    def test_creates_with_custom_config(self, test_config):
        agent = AutonomousResearchAgent(config=test_config)
        assert agent.config.max_agent_turns == 5

    def test_lazy_client_creation(self, agent):
        assert agent._client is None
        client = agent.client
        assert client is not None


class TestAgentToolTracking:
    """Verify the agent tracks tool calls, sources, and findings correctly."""

    @pytest.mark.asyncio
    async def test_tracks_search_count(self, agent, mock_search_result):
        """Agent counts tavily_search calls."""
        mock_client = AsyncMock()

        # Turn 1: agent calls tavily_search
        turn1 = make_response([
            make_text_block("Let me search for information."),
            make_tool_use_block("t1", "tavily_search", {"query": "AI agents 2026"}),
        ])
        # Turn 2: agent finishes
        turn2 = make_response(
            [make_text_block("Research complete.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create = AsyncMock(side_effect=[turn1, turn2])
        agent._client = mock_client

        result = await agent.run("AI agents research")
        assert result.search_count == 1

    @pytest.mark.asyncio
    async def test_tracks_fetch_count(self, agent, mock_fetch_result):
        """Agent counts fetch_url calls."""
        mock_client = AsyncMock()

        turn1 = make_response([
            make_tool_use_block("t1", "fetch_url", {"url": "https://example.com"}),
        ])
        turn2 = make_response(
            [make_text_block("Done.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create = AsyncMock(side_effect=[turn1, turn2])
        agent._client = mock_client

        result = await agent.run("test query")
        assert result.fetch_count == 1

    @pytest.mark.asyncio
    async def test_collects_sources_from_search(self, agent, mock_search_result):
        """Agent extracts Source objects from search results."""
        mock_client = AsyncMock()

        turn1 = make_response([
            make_tool_use_block("t1", "tavily_search", {"query": "test"}),
        ])
        turn2 = make_response(
            [make_text_block("Done.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create = AsyncMock(side_effect=[turn1, turn2])
        agent._client = mock_client

        with patch(
            "autonomous_research_agent.agent.execute_tool",
            new_callable=AsyncMock,
            return_value=mock_search_result,
        ):
            result = await agent.run("test")
        assert len(result.sources) == 2
        assert result.sources[0].title == "Enterprise AI Agents Report"

    @pytest.mark.asyncio
    async def test_collects_findings_from_analysis(self, agent, mock_analyze_result):
        """Agent extracts findings from analyze_findings results."""
        mock_client = AsyncMock()

        turn1 = make_response([
            make_tool_use_block(
                "t1",
                "analyze_findings",
                {"phase": "broad_search", "findings": [{"headline": "test"}]},
            ),
        ])
        turn2 = make_response(
            [make_text_block("Done.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create = AsyncMock(side_effect=[turn1, turn2])
        agent._client = mock_client

        with patch(
            "autonomous_research_agent.agent.execute_tool",
            new_callable=AsyncMock,
            return_value=mock_analyze_result,
        ):
            result = await agent.run("test")
        assert result.analyze_count == 1
        assert len(result.findings) == 2

    @pytest.mark.asyncio
    async def test_tracks_report_path(self, agent, mock_save_result):
        """Agent records the report file path from save_report."""
        mock_client = AsyncMock()

        turn1 = make_response([
            make_tool_use_block(
                "t1",
                "save_report",
                {"filename": "test.md", "content": "# Report"},
            ),
        ])
        turn2 = make_response(
            [make_text_block("Done.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create = AsyncMock(side_effect=[turn1, turn2])
        agent._client = mock_client

        with patch(
            "autonomous_research_agent.agent.execute_tool",
            new_callable=AsyncMock,
            return_value=mock_save_result,
        ):
            result = await agent.run("test")
        assert "test_report.md" in result.report_path


class TestAgentMultiStepReasoning:
    """Verify multi-step agentic loop behavior."""

    @pytest.mark.asyncio
    async def test_multi_turn_research_loop(self, agent):
        """Agent performs multiple turns: search → analyze → save → end."""
        mock_client = AsyncMock()

        # Turn 1: search
        turn1 = make_response([
            make_text_block("Phase 1: Searching..."),
            make_tool_use_block("t1", "tavily_search", {"query": "AI agents"}),
        ])
        # Turn 2: analyze
        turn2 = make_response([
            make_text_block("Phase 2: Analyzing..."),
            make_tool_use_block(
                "t2",
                "analyze_findings",
                {"phase": "broad_search", "findings": [{"headline": "f1"}]},
            ),
        ])
        # Turn 3: save report
        turn3 = make_response([
            make_text_block("Phase 3: Generating report..."),
            make_tool_use_block(
                "t3",
                "save_report",
                {"filename": "report.md", "content": "# Report"},
            ),
        ])
        # Turn 4: end
        turn4 = make_response(
            [make_text_block("Research complete.")],
            stop_reason="end_turn",
        )

        mock_client.messages.create = AsyncMock(
            side_effect=[turn1, turn2, turn3, turn4]
        )
        agent._client = mock_client

        result = await agent.run("AI agents landscape")

        assert result.search_count == 1
        assert result.analyze_count == 1
        assert len(result.tool_calls) == 3
        assert result.text == "Research complete."

    @pytest.mark.asyncio
    async def test_respects_max_turns(self, agent):
        """Agent stops after max_agent_turns even if not done."""
        mock_client = AsyncMock()

        # Every turn returns a tool call — never ends naturally
        infinite_turn = make_response([
            make_tool_use_block("t1", "tavily_search", {"query": "endless"}),
        ])
        mock_client.messages.create = AsyncMock(return_value=infinite_turn)
        agent._client = mock_client

        result = await agent.run("infinite query")

        # Should stop at max_agent_turns (5 in test config)
        assert mock_client.messages.create.call_count == 5

    @pytest.mark.asyncio
    async def test_handles_no_tool_calls(self, agent):
        """Agent handles responses with text only (no tool calls)."""
        mock_client = AsyncMock()

        turn1 = make_response(
            [make_text_block("I already know the answer.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create = AsyncMock(return_value=turn1)
        agent._client = mock_client

        result = await agent.run("simple question")
        assert result.text == "I already know the answer."
        assert len(result.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_duration_tracking(self, agent):
        """Agent records elapsed time."""
        mock_client = AsyncMock()
        turn1 = make_response(
            [make_text_block("Done.")],
            stop_reason="end_turn",
        )
        mock_client.messages.create = AsyncMock(return_value=turn1)
        agent._client = mock_client

        result = await agent.run("timing test")
        assert result.duration_seconds >= 0
