#!/usr/bin/env python3
"""
Meeting Prep Agent — Pre-meeting briefing generator using Claude Opus 4.7.

Pulls upcoming calendar events, researches attendees and topics via Tavily,
synthesizes a structured briefing document, and stores it to simulated S3
(local file write).

Architecture:
  1. Fetch upcoming calendar events (dummy JSON data)
  2. For each event, research attendees, companies, and agenda topics
  3. Synthesize a per-meeting briefing via Claude Opus 4.7 (Bedrock)
  4. Write the briefing to simulated S3 (local ./briefings/ directory)

Reuses tool infrastructure from agents.tools (tavily_search, fetch_url,
execute_tool) and agent helpers from agents.agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import anthropic

from .tools import (
    ALL_TOOLS as _BASE_TOOLS,
    execute_tool as _base_execute_tool,
    fetch_url,
    tavily_search,
)
from .agent import _extract_text, _summarize_input


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_BEDROCK = "us.anthropic.claude-opus-4-7-20250501-v1:0"
MODEL_DIRECT = "claude-opus-4-7-20250501"
MAX_TOKENS = 8192
BRIEFINGS_DIR = Path("./briefings")

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Meeting Prep Agent that creates thorough, actionable briefing
    documents for upcoming meetings.

    You have access to the following tools:

    1. **get_calendar_events** — Retrieve upcoming calendar events for a date range.
    2. **tavily_search** — Search the web for background on people, companies, or topics.
    3. **fetch_url** — Fetch and read the content of a web page.
    4. **store_briefing** — Save a completed briefing document to storage (S3).

    For each meeting, produce a briefing with these sections:

    ## Meeting Briefing: {meeting title}
    **Date:** ...  |  **Time:** ...  |  **Duration:** ...

    ### Attendees
    - For each attendee: name, title, company, and a brief bio/background.

    ### Context & Background
    - What is this meeting about? Why is it happening now?
    - Relevant recent news or developments.

    ### Key Topics & Talking Points
    - Bulleted list of likely discussion topics with supporting context.

    ### Recommended Preparation
    - Specific actions or materials to review before the meeting.

    ### Open Questions
    - Questions to ask or clarify during the meeting.

    After generating each briefing, use the store_briefing tool to save it.
""")

SYNTHESIS_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a briefing synthesizer. Given structured research data about an
    upcoming meeting (attendees, companies, topics), produce a polished
    Markdown briefing document.

    Your output MUST follow this exact format:

    ## Meeting Briefing: {meeting title}
    **Date:** ...  |  **Time:** ...  |  **Duration:** ...

    ### Attendees
    - For each attendee: name, title, company, and a brief bio/background
      based on the research provided.

    ### Context & Background
    - What is this meeting about? Why is it happening now?
    - Relevant recent news or developments from the research.

    ### Key Topics & Talking Points
    - Bulleted list of likely discussion topics with supporting context.

    ### Recommended Preparation
    - Specific actions or materials to review before the meeting.

    ### Open Questions
    - Questions to ask or clarify during the meeting.

    Write clearly and concisely. Use the research snippets to add depth
    but do not fabricate information that isn't in the research data or
    meeting description. If research was unavailable, note that and work
    from the meeting metadata alone.
""")


# ---------------------------------------------------------------------------
# Dummy Calendar Data
# ---------------------------------------------------------------------------

def _generate_calendar_events() -> list[dict[str, Any]]:
    """
    Return dummy calendar events simulating a Google Calendar / Outlook API
    response. Uses relative dates so the data always looks current.
    """
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)

    return [
        {
            "id": "evt-001",
            "summary": "Q3 Partnership Review — Acme Health Systems",
            "start": tomorrow.replace(hour=10, minute=0).isoformat(),
            "end": tomorrow.replace(hour=11, minute=0).isoformat(),
            "location": "Zoom — https://zoom.us/j/123456789",
            "organizer": {"name": "Sarah Chen", "email": "s.chen@company.com"},
            "attendees": [
                {"name": "Sarah Chen", "email": "s.chen@company.com", "role": "VP Partnerships"},
                {"name": "James Whitfield", "email": "j.whitfield@acmehealth.com", "role": "Chief Strategy Officer", "company": "Acme Health Systems"},
                {"name": "Dr. Maria Santos", "email": "m.santos@acmehealth.com", "role": "CMO", "company": "Acme Health Systems"},
                {"name": "Oliver", "email": "oliver@company.com", "role": "CEO"},
            ],
            "description": (
                "Quarterly review of Acme Health partnership. Agenda: utilization "
                "metrics, renewal terms, and expansion into their West Coast clinics. "
                "Acme recently announced a $50M Series C."
            ),
        },
        {
            "id": "evt-002",
            "summary": "AI Product Roadmap — Board Prep",
            "start": tomorrow.replace(hour=14, minute=0).isoformat(),
            "end": tomorrow.replace(hour=15, minute=30).isoformat(),
            "location": "Conference Room A",
            "organizer": {"name": "Oliver", "email": "oliver@company.com"},
            "attendees": [
                {"name": "Oliver", "email": "oliver@company.com", "role": "CEO"},
                {"name": "Priya Sharma", "email": "p.sharma@company.com", "role": "VP Engineering"},
                {"name": "Alex Petrov", "email": "a.petrov@company.com", "role": "Head of AI/ML"},
            ],
            "description": (
                "Prep session for the board meeting next week. Need to finalize the "
                "AI product roadmap slides: agentic workflows, Claude integration "
                "strategy, and competitive landscape vs. OpenAI/Google. Priya will "
                "present eng capacity plan."
            ),
        },
        {
            "id": "evt-003",
            "summary": "Investor Catch-up — Meridian Ventures",
            "start": day_after.replace(hour=9, minute=0).isoformat(),
            "end": day_after.replace(hour=9, minute=45).isoformat(),
            "location": "Phone call",
            "organizer": {"name": "David Park", "email": "d.park@meridianvc.com"},
            "attendees": [
                {"name": "David Park", "email": "d.park@meridianvc.com", "role": "Managing Partner", "company": "Meridian Ventures"},
                {"name": "Oliver", "email": "oliver@company.com", "role": "CEO"},
            ],
            "description": (
                "Monthly check-in with lead investor. Topics: burn rate, hiring "
                "plan, and the Acme Health expansion. David mentioned interest in "
                "our AI agent platform strategy."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Meeting-prep-specific Tool Schemas
# ---------------------------------------------------------------------------

GET_CALENDAR_EVENTS_TOOL: dict[str, Any] = {
    "name": "get_calendar_events",
    "type": "custom",
    "description": (
        "Retrieve upcoming calendar events for the specified number of days. "
        "Returns event details including title, time, attendees, and description."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "days_ahead": {
                "type": "integer",
                "description": "Number of days ahead to look for events (1-7).",
                "default": 2,
            },
        },
        "required": [],
    },
}

STORE_BRIEFING_TOOL: dict[str, Any] = {
    "name": "store_briefing",
    "type": "custom",
    "description": (
        "Store a completed meeting briefing document to S3-compatible storage. "
        "Accepts the event ID and the Markdown-formatted briefing content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "The calendar event ID this briefing is for.",
            },
            "title": {
                "type": "string",
                "description": "The meeting title, used for the filename.",
            },
            "content": {
                "type": "string",
                "description": "The full Markdown briefing document content.",
            },
        },
        "required": ["event_id", "title", "content"],
    },
}

# Combine base tools (tavily_search, fetch_url) with meeting-prep-specific tools
ALL_TOOLS: list[dict[str, Any]] = [
    GET_CALENDAR_EVENTS_TOOL,
    *_BASE_TOOLS,
    STORE_BRIEFING_TOOL,
]


# ---------------------------------------------------------------------------
# Meeting-prep-specific Tool Implementations
# ---------------------------------------------------------------------------

async def get_calendar_events(days_ahead: int = 2) -> dict[str, Any]:
    """Return dummy calendar events within the requested window."""
    events = _generate_calendar_events()
    cutoff = datetime.now() + timedelta(days=max(1, min(days_ahead, 7)))
    filtered = [
        evt for evt in events
        if datetime.fromisoformat(evt["start"]) <= cutoff
    ]
    return {
        "events": filtered,
        "count": len(filtered),
        "window_days": days_ahead,
    }


async def store_briefing(
    event_id: str,
    title: str,
    content: str,
    briefings_dir: Path = BRIEFINGS_DIR,
) -> dict[str, Any]:
    """
    Simulated S3 storage — writes the briefing as a Markdown file to
    the local ./briefings/ directory.

    In production, this would use boto3 to upload to an S3 bucket:
        s3.put_object(Bucket="meeting-briefings", Key=key, Body=content)
    """
    briefings_dir.mkdir(parents=True, exist_ok=True)

    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_").lower()
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{safe_title}_{event_id}.md"
    filepath = briefings_dir / filename

    filepath.write_text(content, encoding="utf-8")

    s3_key = f"briefings/{date_str}/{filename}"
    return {
        "status": "stored",
        "s3_bucket": "meeting-briefings",
        "s3_key": s3_key,
        "local_path": str(filepath),
        "size_bytes": len(content.encode("utf-8")),
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Tool Dispatcher (extends base dispatcher with meeting-prep tools)
# ---------------------------------------------------------------------------

_MEETING_PREP_DISPATCH: dict[str, Any] = {
    "get_calendar_events": get_calendar_events,
    "store_briefing": store_briefing,
}


async def execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """Route a tool call to its implementation and return JSON result."""
    handler = _MEETING_PREP_DISPATCH.get(name)
    if handler is not None:
        try:
            result = await handler(**input_dict)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})
    # Delegate to base dispatcher for tavily_search, fetch_url, etc.
    return await _base_execute_tool(name, input_dict)


# ---------------------------------------------------------------------------
# Agent Result
# ---------------------------------------------------------------------------

@dataclass
class BriefingResult:
    """Structured output from a meeting prep agent run."""
    text: str = ""
    briefings_stored: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# High-Level Research Functions
# ---------------------------------------------------------------------------

_FALLBACK_RESEARCH: dict[str, str] = {
    "attendee": "No live search results available. Brief based on provided attendee metadata only.",
    "company": "No live search results available. Brief based on provided company name only.",
    "topic": "No live search results available. Brief based on meeting description only.",
}


async def research_attendee(
    name: str,
    role: str | None = None,
    company: str | None = None,
    max_results: int = 3,
) -> dict[str, Any]:
    """Research a meeting attendee by name, role, and company."""
    query_parts = [name]
    if role:
        query_parts.append(role)
    if company:
        query_parts.append(company)
    query = " ".join(query_parts)

    raw = await tavily_search(query, max_results=max_results)

    result: dict[str, Any] = {
        "name": name,
        "role": role,
        "company": company,
        "query_used": query,
        "snippets": [],
        "urls": [],
    }

    if raw.get("error"):
        result["error"] = raw["error"]
        result["fallback_note"] = _FALLBACK_RESEARCH["attendee"]
        return result

    for r in raw.get("results", []):
        if r.get("content"):
            result["snippets"].append(r["content"])
        if r.get("url"):
            result["urls"].append(r["url"])

    return result


async def research_company(
    company: str,
    context: str | None = None,
    max_results: int = 3,
) -> dict[str, Any]:
    """Research a company — recent news, funding, strategy."""
    query = f"{company} recent news"
    if context:
        query += f" {context}"

    raw = await tavily_search(query, max_results=max_results)

    result: dict[str, Any] = {
        "company": company,
        "query_used": query,
        "snippets": [],
        "urls": [],
    }

    if raw.get("error"):
        result["error"] = raw["error"]
        result["fallback_note"] = _FALLBACK_RESEARCH["company"]
        return result

    for r in raw.get("results", []):
        if r.get("content"):
            result["snippets"].append(r["content"])
        if r.get("url"):
            result["urls"].append(r["url"])

    return result


async def research_topic(
    topic: str,
    max_results: int = 3,
) -> dict[str, Any]:
    """Research a meeting topic — industry trends, recent developments."""
    raw = await tavily_search(topic, max_results=max_results)

    result: dict[str, Any] = {
        "topic": topic,
        "query_used": topic,
        "snippets": [],
        "urls": [],
    }

    if raw.get("error"):
        result["error"] = raw["error"]
        result["fallback_note"] = _FALLBACK_RESEARCH["topic"]
        return result

    for r in raw.get("results", []):
        if r.get("content"):
            result["snippets"].append(r["content"])
        if r.get("url"):
            result["urls"].append(r["url"])

    return result


async def research_meeting(
    event: dict[str, Any],
    max_results_per_query: int = 3,
) -> dict[str, Any]:
    """
    Run a full research pass for one calendar event.

    Concurrently researches all attendees, unique companies, and the
    meeting topic. Returns a consolidated research bundle.
    """
    attendees = event.get("attendees", [])
    description = event.get("description", "")
    summary = event.get("summary", "Untitled Meeting")

    attendee_tasks = [
        research_attendee(
            name=att.get("name", "Unknown"),
            role=att.get("role"),
            company=att.get("company"),
            max_results=max_results_per_query,
        )
        for att in attendees
    ]

    seen_companies: set[str] = set()
    company_tasks = []
    for att in attendees:
        co = att.get("company")
        if co and co not in seen_companies:
            seen_companies.add(co)
            company_tasks.append(
                research_company(co, context=description[:80] if description else None, max_results=max_results_per_query)
            )

    topic_queries: list[str] = []
    if description:
        topic_queries.append(f"{summary} {description[:120]}")
    else:
        topic_queries.append(summary)
    topic_tasks = [research_topic(q, max_results=max_results_per_query) for q in topic_queries]

    all_results = await asyncio.gather(
        asyncio.gather(*attendee_tasks),
        asyncio.gather(*company_tasks),
        asyncio.gather(*topic_tasks),
    )

    return {
        "event_id": event.get("id", ""),
        "summary": summary,
        "attendees": list(all_results[0]),
        "companies": list(all_results[1]),
        "topics": list(all_results[2]),
    }


# ---------------------------------------------------------------------------
# Client Factory
# ---------------------------------------------------------------------------

def _create_client() -> tuple[anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock, str]:
    """Create an async Anthropic client based on environment config. Returns (client, model)."""
    use_bedrock = os.environ.get("MEETING_PREP_USE_BEDROCK", "").lower() in ("1", "true", "yes")
    if use_bedrock:
        return anthropic.AsyncAnthropicBedrock(), MODEL_BEDROCK
    return anthropic.AsyncAnthropic(), MODEL_DIRECT


# ---------------------------------------------------------------------------
# Briefing Synthesis (Claude Opus 4.7)
# ---------------------------------------------------------------------------

async def synthesize_briefing(
    event: dict[str, Any],
    research: dict[str, Any],
    client: anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock,
    model: str,
    verbose: bool = True,
) -> str:
    """
    Synthesize a well-formatted Markdown briefing from research results
    using Claude Opus 4.7.

    Args:
        event: The calendar event dict.
        research: Output from research_meeting().
        client: Pre-created async Anthropic client (avoids per-call overhead).
        model: Model identifier string.
        verbose: Whether to log progress.

    Returns:
        A Markdown-formatted briefing string.
    """
    research_payload = json.dumps(research, indent=2, default=str)
    event_payload = json.dumps(event, indent=2, default=str)

    user_prompt = textwrap.dedent(f"""\
        Synthesize a meeting briefing document from the following data.

        ## Calendar Event
        ```json
        {event_payload}
        ```

        ## Research Results
        ```json
        {research_payload}
        ```

        Produce the briefing in well-formatted Markdown following the template
        in your system prompt. Make it thorough, actionable, and professional.
    """)

    if verbose:
        print(f"  Synthesizing briefing for: {event.get('summary', 'Unknown')}...", file=sys.stderr)

    response = await client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    briefing_text = _extract_text(response)

    if verbose:
        print(f"    Done ({len(briefing_text)} chars)", file=sys.stderr)

    return briefing_text


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

async def run_pipeline(days_ahead: int = 2, verbose: bool = True) -> BriefingResult:
    """
    Run the full meeting prep pipeline end-to-end:

      1. Fetch calendar events
      2. Research each meeting (attendees, companies, topics) — parallel
      3. Synthesize briefings via Claude Opus 4.7 — parallel
      4. Store each briefing to simulated S3 (./briefings/) — parallel

    Returns a BriefingResult with all stored briefings and metadata.
    """
    result = BriefingResult()

    # --- Step 1: Fetch calendar events ---
    if verbose:
        print("Step 1/4: Fetching calendar events...", file=sys.stderr)

    cal_data = await get_calendar_events(days_ahead=days_ahead)
    events = cal_data["events"]

    if verbose:
        print(f"  Found {len(events)} event(s) in the next {days_ahead} day(s).", file=sys.stderr)

    if not events:
        result.text = "No upcoming meetings found in the specified window."
        return result

    # --- Step 2: Research each meeting (fan-out) ---
    if verbose:
        print("Step 2/4: Researching meetings...", file=sys.stderr)

    research_tasks = [research_meeting(evt) for evt in events]
    all_research = await asyncio.gather(*research_tasks)

    if verbose:
        print(f"  Research complete for {len(all_research)} meeting(s).", file=sys.stderr)

    # --- Step 3: Synthesize briefings via Claude Opus 4.7 (parallel) ---
    if verbose:
        print("Step 3/4: Synthesizing briefings via Claude Opus 4.7...", file=sys.stderr)

    client, model = _create_client()

    synthesis_tasks = [
        synthesize_briefing(evt, research_data, client=client, model=model, verbose=verbose)
        for evt, research_data in zip(events, all_research)
    ]
    briefing_texts = await asyncio.gather(*synthesis_tasks)

    # --- Step 4: Store briefings to simulated S3 (parallel) ---
    if verbose:
        print("Step 4/4: Storing briefings to simulated S3...", file=sys.stderr)

    store_tasks = [
        store_briefing(
            event_id=evt.get("id", "unknown"),
            title=evt.get("summary", "Untitled Meeting"),
            content=briefing_text,
        )
        for evt, briefing_text in zip(events, briefing_texts)
    ]
    stored_results = await asyncio.gather(*store_tasks)

    for evt, stored in zip(events, stored_results):
        result.briefings_stored.append(stored)
        result.tool_calls.append({
            "name": "store_briefing",
            "input": {"event_id": evt.get("id"), "title": evt.get("summary")},
        })
        if verbose:
            print(f"    Stored: {stored.get('local_path', 'unknown')}", file=sys.stderr)

    # Build summary text
    summary_lines = [
        "# Meeting Prep — Pipeline Complete\n",
        f"Generated **{len(briefing_texts)}** briefing(s) for the next {days_ahead} day(s).\n",
        "## Briefings Generated\n",
    ]
    for evt, stored in zip(events, result.briefings_stored):
        summary_lines.append(f"- **{evt.get('summary')}** → `{stored.get('local_path')}`")
    summary_lines.append("\n---\n")
    for briefing_text in briefing_texts:
        summary_lines.append(briefing_text)
        summary_lines.append("\n---\n")

    result.text = "\n".join(summary_lines)
    return result


# ---------------------------------------------------------------------------
# Managed-agents runner (Claude Agent SDK)
# ---------------------------------------------------------------------------

class ManagedMeetingPrepRunner:
    """
    Runs the meeting prep pipeline via the Claude Agent SDK managed-agents API.

    Flow:
      1. Create Agent with calendar + research + storage tools.
      2. Create Environment and Session.
      3. Send the prep prompt and stream events.
      4. Handle tool calls (calendar fetch, research, briefing storage).
      5. Clean up resources.
    """

    def __init__(self, verbose: bool = True):
        self.client = anthropic.Anthropic()
        self.verbose = verbose

    async def run(self, prompt: str) -> BriefingResult:
        agent = None
        environment = None
        result = BriefingResult()

        try:
            self._log("Creating managed meeting-prep agent...")
            agent = self.client.beta.agents.create(
                model=MODEL_DIRECT,
                name="meeting-prep-agent",
                description="Pre-meeting briefing generator with calendar, research, and storage tools.",
                system=SYSTEM_PROMPT,
                tools=ALL_TOOLS,
            )
            self._log(f"  Agent created: {agent.id}")

            self._log("Creating environment...")
            environment = self.client.beta.environments.create(name="meeting-prep-env")
            self._log(f"  Environment created: {environment.id}")

            self._log("Creating session...")
            session = self.client.beta.sessions.create(
                agent=agent.id,
                environment_id=environment.id,
            )
            self._log(f"  Session created: {session.id}")

            self._log("Sending meeting prep request...")
            self.client.beta.sessions.events.send(
                session_id=session.id,
                events=[{
                    "type": "user.message",
                    "content": [{"type": "text", "text": prompt}],
                }],
            )

            final_text = ""
            max_rounds = 30
            for round_num in range(1, max_rounds + 1):
                self._log(f"  Streaming events (round {round_num})...")
                pending_tool_results = []

                events = self.client.beta.sessions.events.list(
                    session_id=session.id,
                    order="asc",
                )
                for event in events.data:
                    if event.type == "agent.message":
                        for block in event.content:
                            if hasattr(block, "text"):
                                final_text = block.text
                        self._log(f"    Agent message ({len(final_text)} chars)")

                    elif event.type == "agent.custom_tool_use":
                        tool_name = event.name
                        tool_input = dict(event.input)
                        self._log(f"    -> {tool_name}({_summarize_input(tool_input)})")

                        result.tool_calls.append({"name": tool_name, "input": tool_input})
                        result_str = await execute_tool(tool_name, tool_input)

                        if tool_name == "store_briefing":
                            try:
                                stored = json.loads(result_str)
                                if stored.get("status") == "stored":
                                    result.briefings_stored.append(stored)
                            except json.JSONDecodeError:
                                pass

                        pending_tool_results.append({
                            "type": "user.custom_tool_result",
                            "custom_tool_use_id": event.id,
                            "content": [{"type": "text", "text": result_str}],
                        })

                    elif event.type == "session.status_idle":
                        stop = event.stop_reason
                        if hasattr(stop, "type") and stop.type == "end_turn":
                            self._log("  Agent finished (end_turn).")
                            result.text = final_text
                            return result

                if pending_tool_results:
                    self._log(f"    Sending {len(pending_tool_results)} tool result(s)...")
                    self.client.beta.sessions.events.send(
                        session_id=session.id,
                        events=pending_tool_results,
                    )
                else:
                    if final_text:
                        result.text = final_text
                        return result
                    break

            self._log("  Agent reached maximum rounds.")
            result.text = final_text or "Agent reached maximum rounds without completing."
            return result

        finally:
            if agent:
                try:
                    self.client.beta.agents.archive(agent_id=agent.id)
                except Exception:
                    pass
            if environment:
                try:
                    self.client.beta.environments.delete(environment_id=environment.id)
                except Exception:
                    pass

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Local async runner (fallback / Bedrock-compatible)
# ---------------------------------------------------------------------------

class LocalMeetingPrepRunner:
    """
    Runs the meeting prep agentic loop locally using AsyncAnthropic.

    Supports both direct Anthropic API and AWS Bedrock as the backend,
    controlled by the MEETING_PREP_USE_BEDROCK environment variable.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.client, self.model = _create_client()

        if self.verbose:
            backend = "AWS Bedrock" if self.model == MODEL_BEDROCK else "direct Anthropic API"
            print(f"Using {backend} backend", file=sys.stderr)

    @staticmethod
    def _messages_api_tools() -> list[dict[str, Any]]:
        """Convert tool schemas to messages API format (strip 'type' key)."""
        return [
            {k: v for k, v in tool.items() if k != "type"}
            for tool in ALL_TOOLS
        ]

    async def run(self, prompt: str) -> BriefingResult:
        """Run the multi-step meeting prep loop."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]
        result = BriefingResult()
        max_turns = 30

        self._log("Starting meeting prep agent loop...")

        for turn in range(1, max_turns + 1):
            self._log(f"  Turn {turn}/{max_turns}")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=self._messages_api_tools(),
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                result.text = _extract_text(response)
                self._log("  Agent finished.")
                return result

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type == "tool_use":
                    self._log(f"    -> {block.name}({_summarize_input(block.input)})")
                    result.tool_calls.append({"name": block.name, "input": block.input})
                    result_str = await execute_tool(block.name, block.input)

                    if block.name == "store_briefing":
                        try:
                            stored = json.loads(result_str)
                            if stored.get("status") == "stored":
                                result.briefings_stored.append(stored)
                        except json.JSONDecodeError:
                            pass

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                self._log(f"  Unexpected stop_reason: {response.stop_reason}")
                result.text = _extract_text(response)
                if result.text.strip():
                    return result
                break

        self._log("  Agent reached maximum turns without completing.")
        result.text = "Agent reached maximum turns without completing."
        return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Public Agent Facade
# ---------------------------------------------------------------------------

class MeetingPrepAgent:
    """
    Meeting Prep Agent — generates pre-meeting briefing documents.

    Fetches calendar events, researches attendees and topics via Tavily,
    synthesizes briefings using Claude Opus 4.7, and stores them to
    simulated S3 (local file system).

    Supports managed-agents API with automatic fallback to local runner.
    """

    def __init__(self, use_managed: bool = True, verbose: bool = True):
        self.use_managed = use_managed
        self.verbose = verbose

    async def run(self, days_ahead: int = 2) -> BriefingResult:
        """
        Generate meeting briefings for all events in the specified window.

        Args:
            days_ahead: Number of days to look ahead for calendar events.

        Returns:
            BriefingResult with the agent's output text, stored briefings,
            and tool call history.
        """
        prompt = textwrap.dedent(f"""\
            Prepare meeting briefings for all my upcoming meetings in the
            next {days_ahead} day(s).

            Steps:
            1. First, call get_calendar_events to retrieve my upcoming meetings.
            2. For each meeting, research the attendees and companies using
               tavily_search. Use specific queries like:
               - "{{person name}} {{company}} {{role}}"
               - "{{company}} recent news funding"
               - "{{meeting topic}} industry trends"
            3. Synthesize a comprehensive briefing document for each meeting.
            4. Store each briefing using store_briefing with the event_id.
            5. After storing all briefings, provide a brief summary of what
               was prepared.

            Be thorough in your research — I want to walk into each meeting
            well-informed about the people and topics involved.
        """)

        if self.use_managed:
            try:
                runner = ManagedMeetingPrepRunner(verbose=self.verbose)
                return await runner.run(prompt)
            except (anthropic.APIError, anthropic.APIConnectionError) as exc:
                if self.verbose:
                    print(
                        f"  Managed-agents API unavailable ({type(exc).__name__}), "
                        "falling back to local runner...",
                        file=sys.stderr,
                    )

        runner = LocalMeetingPrepRunner(verbose=self.verbose)
        return await runner.run(prompt)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

async def main_async() -> None:
    """Run the meeting prep agent and print results."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="meeting_prep_agent",
        description="Generate pre-meeting briefing documents using Claude Opus 4.7.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m agents.meeting_prep_agent --pipeline
              python -m agents.meeting_prep_agent --pipeline --days 3 --json
              python -m agents.meeting_prep_agent --local --json
              MEETING_PREP_USE_BEDROCK=1 python -m agents.meeting_prep_agent --pipeline
        """),
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=2,
        help="Number of days ahead to prepare briefings for (default: 2)",
    )
    parser.add_argument(
        "--pipeline", "-p",
        action="store_true",
        help="Use direct pipeline mode (fetch → research → synthesize → store) instead of agentic loop",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local agent loop instead of managed-agents API",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON (includes tool call metadata)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose logging to stderr",
    )

    args = parser.parse_args()

    if args.pipeline:
        print("\nMeeting Prep Agent — Pipeline Mode\n", file=sys.stderr)
        result = await run_pipeline(days_ahead=args.days, verbose=not args.quiet)
    else:
        agent = MeetingPrepAgent(
            use_managed=not args.local,
            verbose=not args.quiet,
        )
        print("\nMeeting Prep Agent starting...\n", file=sys.stderr)
        result = await agent.run(days_ahead=args.days)

    if args.json:
        print(json.dumps({
            "text": result.text,
            "briefings_stored": result.briefings_stored,
            "tool_calls": result.tool_calls,
            "timestamp": result.timestamp,
        }, indent=2, default=str))
    else:
        print(result.text)

    n_briefings = len(result.briefings_stored)
    n_tools = len(result.tool_calls)
    print(f"\n{'─' * 60}", file=sys.stderr)
    print(f"  Briefings stored: {n_briefings}", file=sys.stderr)
    print(f"  Tool calls made:  {n_tools}", file=sys.stderr)
    for b in result.briefings_stored:
        print(f"    → {b.get('local_path', b.get('s3_key', 'unknown'))}", file=sys.stderr)
    print(f"{'─' * 60}\n", file=sys.stderr)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
