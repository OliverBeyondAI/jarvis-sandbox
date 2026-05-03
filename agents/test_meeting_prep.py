#!/usr/bin/env python3
"""
Tests — Meeting Prep Agent

Covers calendar event generation, tool dispatch, store_briefing file output,
research helpers, synthesis, and the full pipeline flow.

All tests are offline by default (no API keys required). Uses mocks for
Claude API calls and Tavily searches.

Usage:
    python -m agents.test_meeting_prep              # offline tests
    python -m agents.test_meeting_prep --live       # live end-to-end (requires keys)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from .meeting_prep_agent import (
    ALL_TOOLS,
    BRIEFINGS_DIR,
    BriefingResult,
    MeetingPrepAgent,
    _generate_calendar_events,
    execute_tool,
    get_calendar_events,
    research_attendee,
    research_company,
    research_meeting,
    research_topic,
    run_pipeline,
    store_briefing,
    synthesize_briefing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_checks(suite: str, checks: list[tuple[str, bool]]) -> bool:
    """Run a list of (name, passed) checks and print results."""
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    all_ok = passed == total

    status = "\033[38;5;42m PASS \033[0m" if all_ok else "\033[38;5;196m FAIL \033[0m"
    print(f"  [{status}] {suite} ({passed}/{total})")

    for name, ok in checks:
        icon = "\033[38;5;42m✓\033[0m" if ok else "\033[38;5;196m✗\033[0m"
        print(f"    {icon} {name}")

    return all_ok


# ---------------------------------------------------------------------------
# Calendar Event Tests
# ---------------------------------------------------------------------------

def test_calendar_event_generation() -> bool:
    """Validate dummy calendar events have correct structure and relative dates."""
    events = _generate_calendar_events()
    checks = [
        ("returns 3 events", len(events) == 3),
        ("events have IDs", all("id" in e for e in events)),
        ("events have summaries", all("summary" in e for e in events)),
        ("events have start/end", all("start" in e and "end" in e for e in events)),
        ("events have attendees", all(len(e.get("attendees", [])) > 0 for e in events)),
        ("events have descriptions", all("description" in e for e in events)),
    ]

    # Check dates are relative (tomorrow and day after)
    today = datetime.now()
    for evt in events:
        start = datetime.fromisoformat(evt["start"])
        delta = (start - today).days
        checks.append((f"{evt['id']} date is future", delta >= 0))

    # Check attendee structure
    first_event = events[0]
    first_attendee = first_event["attendees"][0]
    checks.append(("attendee has name", "name" in first_attendee))
    checks.append(("attendee has email", "email" in first_attendee))
    checks.append(("attendee has role", "role" in first_attendee))

    return _run_checks("Calendar Event Generation", checks)


def test_get_calendar_events_filtering() -> bool:
    """Test that get_calendar_events filters by days_ahead correctly."""
    # With days_ahead=1, should only get tomorrow's events (2 of 3)
    result_1 = asyncio.run(get_calendar_events(days_ahead=1))
    # With days_ahead=3, should get all events
    result_3 = asyncio.run(get_calendar_events(days_ahead=3))

    checks = [
        ("1-day has events key", "events" in result_1),
        ("1-day has count key", "count" in result_1),
        ("1-day count matches list", result_1["count"] == len(result_1["events"])),
        ("3-day has all 3 events", result_3["count"] == 3),
        ("1-day has fewer events", result_1["count"] <= result_3["count"]),
        ("window_days preserved (1)", result_1["window_days"] == 1),
        ("window_days preserved (3)", result_3["window_days"] == 3),
    ]

    return _run_checks("Calendar Event Filtering", checks)


# ---------------------------------------------------------------------------
# Tool Schema Tests
# ---------------------------------------------------------------------------

def test_tool_schemas() -> bool:
    """Validate all meeting prep tool schemas."""
    checks = []
    expected_tools = {"get_calendar_events", "tavily_search", "fetch_url", "store_briefing"}
    tool_names = {t["name"] for t in ALL_TOOLS}

    checks.append(("has all 4 tools", tool_names == expected_tools))

    for tool in ALL_TOOLS:
        name = tool.get("name", "unknown")
        checks.append((f"{name} has type=custom", tool.get("type") == "custom"))
        checks.append((f"{name} has description", bool(tool.get("description"))))
        checks.append((f"{name} has input_schema", bool(tool.get("input_schema"))))
        schema = tool.get("input_schema", {})
        checks.append((f"{name} schema has properties", "properties" in schema))

    return _run_checks("Meeting Prep Tool Schemas", checks)


# ---------------------------------------------------------------------------
# Tool Dispatch Tests
# ---------------------------------------------------------------------------

def test_tool_dispatch_calendar() -> bool:
    """Test execute_tool routes get_calendar_events correctly."""
    result_str = asyncio.run(execute_tool("get_calendar_events", {"days_ahead": 2}))
    result = json.loads(result_str)

    checks = [
        ("returns events", "events" in result),
        ("returns count", "count" in result),
        ("no error", "error" not in result),
        ("events is list", isinstance(result["events"], list)),
        ("has events", len(result["events"]) > 0),
    ]

    return _run_checks("Tool Dispatch — Calendar", checks)


def test_tool_dispatch_unknown() -> bool:
    """Test execute_tool handles unknown tools gracefully."""
    result_str = asyncio.run(execute_tool("nonexistent_tool", {"foo": "bar"}))
    result = json.loads(result_str)

    checks = [
        ("returns error", "error" in result),
        ("mentions tool name", "nonexistent_tool" in result.get("error", "")),
    ]

    return _run_checks("Tool Dispatch — Unknown Tool", checks)


def test_tool_dispatch_delegates_to_base() -> bool:
    """Test that tavily_search and fetch_url delegate to base tools module."""
    # tavily_search without API key should return an error gracefully
    result_str = asyncio.run(execute_tool("tavily_search", {"query": "test"}))
    result = json.loads(result_str)

    checks = [
        ("tavily returns valid json", isinstance(result, dict)),
        ("tavily has query or error", "query" in result or "error" in result),
    ]

    # fetch_url with invalid URL should return error
    result_str = asyncio.run(execute_tool("fetch_url", {"url": "http://invalid.test.localhost.nowhere"}))
    result = json.loads(result_str)
    checks.append(("fetch_url bad URL returns error", "error" in result))

    return _run_checks("Tool Dispatch — Base Tool Delegation", checks)


# ---------------------------------------------------------------------------
# store_briefing Tests
# ---------------------------------------------------------------------------

def test_store_briefing_writes_file() -> bool:
    """Test that store_briefing creates a markdown file with correct content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        content = "# Test Briefing\n\nThis is a test."

        result = asyncio.run(store_briefing(
            event_id="evt-test-001",
            title="Test Meeting — Alpha Corp",
            content=content,
            briefings_dir=tmp_path,
        ))

        checks = [
            ("status is stored", result["status"] == "stored"),
            ("has s3_bucket", result["s3_bucket"] == "meeting-briefings"),
            ("has s3_key", result["s3_key"].startswith("briefings/")),
            ("has local_path", bool(result["local_path"])),
            ("has size_bytes", result["size_bytes"] == len(content.encode("utf-8"))),
            ("has timestamp", bool(result["timestamp"])),
        ]

        # Verify file was actually written
        written_files = list(tmp_path.glob("*.md"))
        checks.append(("file exists on disk", len(written_files) == 1))

        if written_files:
            file_content = written_files[0].read_text(encoding="utf-8")
            checks.append(("file content matches", file_content == content))
            checks.append(("filename has event_id", "evt-test-001" in written_files[0].name))
            checks.append(("filename has date", datetime.now().strftime("%Y-%m-%d") in written_files[0].name))

    return _run_checks("store_briefing — File Output", checks)


def test_store_briefing_sanitizes_title() -> bool:
    """Test that store_briefing handles special characters in titles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = asyncio.run(store_briefing(
            event_id="evt-special",
            title="Meeting: Q3/Q4 Review (Confidential!) @HQ",
            content="test",
            briefings_dir=Path(tmpdir),
        ))

        checks = [
            ("status is stored", result["status"] == "stored"),
        ]

        written_files = list(Path(tmpdir).glob("*.md"))
        checks.append(("file created", len(written_files) == 1))

        if written_files:
            fname = written_files[0].name
            checks.append(("no colons in filename", ":" not in fname))
            checks.append(("no parens in filename", "(" not in fname and ")" not in fname))
            checks.append(("no slashes in filename", "/" not in fname))

    return _run_checks("store_briefing — Title Sanitization", checks)


# ---------------------------------------------------------------------------
# Research Helper Tests (mocked Tavily)
# ---------------------------------------------------------------------------

MOCK_TAVILY_RESULT: dict = {
    "query": "test query",
    "results": [
        {"title": "Result 1", "url": "https://example.com/1", "content": "Snippet 1", "score": 0.9},
        {"title": "Result 2", "url": "https://example.com/2", "content": "Snippet 2", "score": 0.8},
    ],
}


def _mock_tavily(*args, **kwargs):
    """Return mock tavily results."""
    return MOCK_TAVILY_RESULT


def _make_tavily_mock():
    return AsyncMock(return_value=MOCK_TAVILY_RESULT)


def test_research_attendee() -> bool:
    """Test research_attendee with mocked Tavily."""
    with patch("agents.meeting_prep_agent.tavily_search", new=_make_tavily_mock()):
        result = asyncio.run(research_attendee(
            name="Jane Doe",
            role="CTO",
            company="TechCorp",
        ))

    checks = [
        ("has name", result["name"] == "Jane Doe"),
        ("has role", result["role"] == "CTO"),
        ("has company", result["company"] == "TechCorp"),
        ("has query_used", "Jane Doe" in result["query_used"]),
        ("query includes role", "CTO" in result["query_used"]),
        ("query includes company", "TechCorp" in result["query_used"]),
        ("has snippets", len(result["snippets"]) == 2),
        ("has urls", len(result["urls"]) == 2),
        ("no error", "error" not in result),
    ]

    return _run_checks("research_attendee", checks)


def test_research_attendee_fallback() -> bool:
    """Test research_attendee falls back gracefully when Tavily errors."""
    error_result = {"query": "test", "error": "API key missing", "results": []}

    with patch("agents.meeting_prep_agent.tavily_search", new=AsyncMock(return_value=error_result)):
        result = asyncio.run(research_attendee(name="John Smith"))

    checks = [
        ("has name", result["name"] == "John Smith"),
        ("has error", "error" in result),
        ("has fallback_note", "fallback_note" in result),
        ("snippets empty", len(result["snippets"]) == 0),
    ]

    return _run_checks("research_attendee — Fallback", checks)


def test_research_company() -> bool:
    """Test research_company with mocked Tavily."""
    with patch("agents.meeting_prep_agent.tavily_search", new=_make_tavily_mock()):
        result = asyncio.run(research_company(
            company="Acme Health",
            context="Series C funding",
        ))

    checks = [
        ("has company", result["company"] == "Acme Health"),
        ("query has company", "Acme Health" in result["query_used"]),
        ("query has context", "Series C funding" in result["query_used"]),
        ("has snippets", len(result["snippets"]) == 2),
        ("has urls", len(result["urls"]) == 2),
    ]

    return _run_checks("research_company", checks)


def test_research_topic() -> bool:
    """Test research_topic with mocked Tavily."""
    with patch("agents.meeting_prep_agent.tavily_search", new=_make_tavily_mock()):
        result = asyncio.run(research_topic(topic="AI product roadmap 2026"))

    checks = [
        ("has topic", result["topic"] == "AI product roadmap 2026"),
        ("has snippets", len(result["snippets"]) == 2),
        ("has urls", len(result["urls"]) == 2),
    ]

    return _run_checks("research_topic", checks)


def test_research_meeting_structure() -> bool:
    """Test research_meeting fans out correctly with mocked Tavily."""
    event = _generate_calendar_events()[0]  # Q3 Partnership Review

    with patch("agents.meeting_prep_agent.tavily_search", new=_make_tavily_mock()):
        result = asyncio.run(research_meeting(event))

    checks = [
        ("has event_id", result["event_id"] == "evt-001"),
        ("has summary", "Partnership" in result["summary"]),
        ("has attendees research", len(result["attendees"]) == 4),
        ("has companies research", len(result["companies"]) > 0),
        ("has topics research", len(result["topics"]) > 0),
        # Each attendee research should have snippets from mock
        ("attendee has snippets", all(len(a["snippets"]) > 0 for a in result["attendees"])),
        # Companies should be deduped
        ("companies deduped", len(result["companies"]) == 1),  # Only "Acme Health Systems"
    ]

    return _run_checks("research_meeting — Structure", checks)


# ---------------------------------------------------------------------------
# BriefingResult Tests
# ---------------------------------------------------------------------------

def test_briefing_result_structure() -> bool:
    """Validate BriefingResult dataclass behavior."""
    r = BriefingResult()
    checks = [
        ("default text empty", r.text == ""),
        ("default briefings_stored empty", r.briefings_stored == []),
        ("default tool_calls empty", r.tool_calls == []),
        ("default has timestamp", bool(r.timestamp)),
    ]

    r2 = BriefingResult(
        text="hello",
        briefings_stored=[{"status": "stored"}],
        tool_calls=[{"name": "test", "input": {}}],
    )
    checks.append(("text set", r2.text == "hello"))
    checks.append(("briefings_stored set", len(r2.briefings_stored) == 1))
    checks.append(("tool_calls set", len(r2.tool_calls) == 1))

    # JSON serialization
    json_out = json.dumps({
        "text": r2.text,
        "briefings_stored": r2.briefings_stored,
        "tool_calls": r2.tool_calls,
        "timestamp": r2.timestamp,
    }, indent=2, default=str)
    parsed = json.loads(json_out)
    checks.append(("json serializable", bool(parsed["text"])))

    return _run_checks("BriefingResult Structure", checks)


# ---------------------------------------------------------------------------
# Synthesis Tests (mocked Claude API)
# ---------------------------------------------------------------------------

def test_synthesize_briefing() -> bool:
    """Test synthesize_briefing with a mocked Claude client."""
    event = _generate_calendar_events()[0]
    research = {
        "event_id": "evt-001",
        "summary": event["summary"],
        "attendees": [{"name": "Test", "snippets": ["Bio snippet"]}],
        "companies": [{"company": "Acme", "snippets": ["Company news"]}],
        "topics": [{"topic": "Partnership", "snippets": ["Topic info"]}],
    }

    # Create a mock response
    mock_block = MagicMock()
    mock_block.text = "## Meeting Briefing: Q3 Review\n\nTest briefing content."
    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    briefing = asyncio.run(synthesize_briefing(
        event=event,
        research=research,
        client=mock_client,
        model="test-model",
        verbose=False,
    ))

    checks = [
        ("returns string", isinstance(briefing, str)),
        ("has content", len(briefing) > 0),
        ("has meeting title", "Q3 Review" in briefing),
        ("client called once", mock_client.messages.create.call_count == 1),
    ]

    # Verify the call was made with correct params
    call_kwargs = mock_client.messages.create.call_args.kwargs
    checks.append(("model passed", call_kwargs["model"] == "test-model"))
    checks.append(("has system prompt", bool(call_kwargs["system"])))
    checks.append(("has messages", len(call_kwargs["messages"]) == 1))

    return _run_checks("synthesize_briefing", checks)


# ---------------------------------------------------------------------------
# Pipeline Flow Tests (fully mocked)
# ---------------------------------------------------------------------------

def test_pipeline_flow() -> bool:
    """Test run_pipeline end-to-end with mocked Tavily and Claude."""
    mock_block = MagicMock()
    mock_block.text = "## Meeting Briefing\n\nMocked briefing content for testing."
    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        original_store = store_briefing

        async def _patched_store(event_id, title, content, briefings_dir=None):
            return await original_store(event_id=event_id, title=title, content=content, briefings_dir=tmp_path)

        with (
            patch("agents.meeting_prep_agent.tavily_search", new=_make_tavily_mock()),
            patch("agents.meeting_prep_agent._create_client", return_value=(mock_client, "test-model")),
            patch("agents.meeting_prep_agent.store_briefing", side_effect=_patched_store),
        ):
            result = asyncio.run(run_pipeline(days_ahead=3, verbose=False))

        checks = [
            ("has text", len(result.text) > 0),
            ("has briefings_stored", len(result.briefings_stored) == 3),
            ("has tool_calls", len(result.tool_calls) == 3),
            ("text mentions pipeline", "Pipeline Complete" in result.text),
        ]

        # Check files were written
        written_files = list(tmp_path.glob("*.md"))
        checks.append(("3 files written", len(written_files) == 3))

        # Verify each stored briefing has expected keys
        for stored in result.briefings_stored:
            checks.append((f"stored {stored.get('s3_key', '?')[:30]} has status", stored["status"] == "stored"))

        # Verify synthesis was parallelized (all 3 calls made)
        checks.append(("synthesis called 3 times", mock_client.messages.create.call_count == 3))

    return _run_checks("Pipeline Flow — End-to-End", checks)


def test_pipeline_no_events() -> bool:
    """Test pipeline handles empty calendar gracefully."""
    with patch("agents.meeting_prep_agent.get_calendar_events", new=AsyncMock(return_value={"events": [], "count": 0, "window_days": 1})):
        result = asyncio.run(run_pipeline(days_ahead=1, verbose=False))

    checks = [
        ("has text", len(result.text) > 0),
        ("no briefings stored", len(result.briefings_stored) == 0),
        ("mentions no meetings", "No upcoming" in result.text),
    ]

    return _run_checks("Pipeline Flow — No Events", checks)


# ---------------------------------------------------------------------------
# Integration: tools reuse from agents.tools
# ---------------------------------------------------------------------------

def test_tools_reuse() -> bool:
    """Verify meeting prep agent reuses tools from agents.tools, not duplicating."""
    from agents.tools import tavily_search as base_tavily, fetch_url as base_fetch
    from agents.meeting_prep_agent import tavily_search as prep_tavily, fetch_url as prep_fetch

    checks = [
        ("tavily_search is same function", prep_tavily is base_tavily),
        ("fetch_url is same function", prep_fetch is base_fetch),
    ]

    return _run_checks("Tools Reuse (no duplication)", checks)


# ---------------------------------------------------------------------------
# Live end-to-end test
# ---------------------------------------------------------------------------

async def test_live_pipeline() -> bool:
    """Run the real pipeline against dummy calendar data (requires API keys)."""
    print("\n  Live Pipeline Test")
    print("  " + "-" * 50)
    print("  Running full pipeline (this may take 30-60 seconds)...\n")

    try:
        result = await run_pipeline(days_ahead=2, verbose=True)
    except Exception as e:
        print(f"  \033[38;5;196m FAIL \033[0m Live Pipeline — {type(e).__name__}: {e}")
        return False

    checks = [
        ("has text output", len(result.text) > 100),
        ("has briefings", len(result.briefings_stored) > 0),
        ("has tool calls", len(result.tool_calls) > 0),
        ("text mentions pipeline", "Pipeline" in result.text),
    ]

    for stored in result.briefings_stored:
        path = stored.get("local_path", "")
        checks.append((f"file exists: {Path(path).name}", Path(path).exists() if path else False))

    return _run_checks("Live Pipeline Output", checks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agents-test-meeting-prep",
        description="Meeting Prep Agent — Tests",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run live end-to-end test (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Meeting Prep Agent — Tests")
    print("=" * 60 + "\n")

    results = [
        test_calendar_event_generation(),
        test_get_calendar_events_filtering(),
        test_tool_schemas(),
        test_tool_dispatch_calendar(),
        test_tool_dispatch_unknown(),
        test_tool_dispatch_delegates_to_base(),
        test_store_briefing_writes_file(),
        test_store_briefing_sanitizes_title(),
        test_research_attendee(),
        test_research_attendee_fallback(),
        test_research_company(),
        test_research_topic(),
        test_research_meeting_structure(),
        test_briefing_result_structure(),
        test_synthesize_briefing(),
        test_pipeline_flow(),
        test_pipeline_no_events(),
        test_tools_reuse(),
    ]

    if args.live:
        live_ok = asyncio.run(test_live_pipeline())
        results.append(live_ok)

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"  \033[38;5;42mAll {total} test suites passed.\033[0m")
    else:
        print(f"  \033[38;5;196m{total - passed}/{total} test suites failed.\033[0m")
    print(f"{'=' * 60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
