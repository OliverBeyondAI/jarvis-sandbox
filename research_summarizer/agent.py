#!/usr/bin/env python3
"""
Research Summarizer Agent

An AI-powered research agent built on the Claude Agent SDK (managed agents
API) that accepts a list of URLs, fetches and analyzes their content, and
produces structured summaries using Claude Opus 4.7 and Tavily web search.

The agent uses the Anthropic managed-agents beta API:
  1. Register an Agent with custom tools (fetch_url, tavily_search).
  2. Create an Environment and Session.
  3. Send a user message and stream events.
  4. Handle ``agent.custom_tool_use`` events by executing tools locally and
     returning results via ``user.custom_tool_result`` events.
  5. Collect the final ``agent.message`` response and parse structured JSON.

A local-only fallback mode is also provided (``AgentRunner``) for
environments where the managed-agents API is unavailable — it drives the
same async tool loop using ``AsyncAnthropic.messages.create``.

Usage:
    from research_summarizer.agent import ResearchSummarizerAgent

    agent = ResearchSummarizerAgent()
    result = await agent.summarize([
        "https://arxiv.org/abs/2301.00001",
        "https://example.com/research-paper",
    ])
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import anthropic

from .tools import ALL_TOOLS, execute_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7-20250501"
MAX_TOKENS = 8192

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Research Summarizer Agent. Your job is to analyze content from
    provided URLs and produce clear, structured research summaries.

    You have access to the following tools:

    1. **fetch_url** - Fetch the text content of a URL. Use this to retrieve
       the full content of each URL the user provides.

    2. **tavily_search** - Search the web for additional context on a topic.
       Use this when you need background information, related work, or
       clarification about concepts found in the source material.

    ## Workflow

    For each URL provided:
    1. Use fetch_url to retrieve its content.
    2. Analyze the content to identify key findings, methodology, and relevance.
    3. If you need more context about a concept or claim, use tavily_search.

    After processing all URLs, produce your final output as a **single valid
    JSON object** (not wrapped in markdown code fences) with this structure:

    {
        "sources": [
            {
                "title": "Article title",
                "url": "https://...",
                "key_findings": ["finding 1", "finding 2", ...],
                "methodology": "Brief description of approach (if applicable)",
                "relevance": "Why this matters in the broader context"
            }
        ],
        "cross_source_synthesis": "Common themes, contradictions, or complementary findings",
        "key_takeaways": ["insight 1", "insight 2", ...],
        "suggested_follow_up": ["question 1", "question 2", ...]
    }

    Be precise, objective, and cite specific claims back to their source URL.
    Provide 3-5 key findings per source and 3-5 overarching takeaways.
""")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SummaryResult:
    """Structured output from the Research Summarizer Agent."""
    sources: list[dict[str, Any]] = field(default_factory=list)
    synthesis: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)
    raw_response: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Managed-agents runner (Claude Agent SDK)
# ---------------------------------------------------------------------------

class ManagedAgentRunner:
    """
    Runs the research summarizer via the Claude Agent SDK managed-agents API.

    Flow:
      1. Create an Agent with custom tools via ``beta.agents.create``.
      2. Create an Environment via ``beta.environments.create``.
      3. Create a Session linking the agent to the environment.
      4. Send a ``user.message`` event with the research prompt.
      5. Stream ``session.event`` objects:
         - On ``agent.custom_tool_use``: execute the tool locally and send
           a ``user.custom_tool_result`` back.
         - On ``agent.message``: collect the final text response.
         - On ``session.status_idle`` with ``end_turn``: parsing is done.
      6. Clean up (archive agent, environment).
    """

    def __init__(
        self,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        verbose: bool = True,
    ):
        self.client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.verbose = verbose

    async def run(self, prompt: str) -> str:
        """Send *prompt* through a managed-agent session and return the final text."""
        agent = None
        environment = None
        session = None

        try:
            # 1. Register the agent with custom tools
            self._log("Creating managed agent...")
            agent = self.client.beta.agents.create(
                model=self.model,
                name="research-summarizer",
                description="Research Summarizer Agent — fetches URLs, searches the web, and produces structured summaries.",
                system=SYSTEM_PROMPT,
                tools=ALL_TOOLS,
            )
            self._log(f"  Agent created: {agent.id}")

            # 2. Create an environment
            self._log("Creating environment...")
            environment = self.client.beta.environments.create(
                name="research-summarizer-env",
            )
            self._log(f"  Environment created: {environment.id}")

            # 3. Create a session
            self._log("Creating session...")
            session = self.client.beta.sessions.create(
                agent=agent.id,
                environment_id=environment.id,
            )
            self._log(f"  Session created: {session.id}")

            # 4. Send the user message
            self._log("Sending user message...")
            self.client.beta.sessions.events.send(
                session_id=session.id,
                events=[{
                    "type": "user.message",
                    "content": [{"type": "text", "text": prompt}],
                }],
            )

            # 5. Stream events and handle tool calls
            final_text = ""
            max_rounds = 25
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
                        self._log(f"    Agent message received ({len(final_text)} chars)")

                    elif event.type == "agent.custom_tool_use":
                        tool_name = event.name
                        tool_input = dict(event.input)
                        self._log(f"    -> Tool call: {tool_name}({_summarize_input(tool_input)})")

                        result_str = await execute_tool(tool_name, tool_input)
                        pending_tool_results.append({
                            "type": "user.custom_tool_result",
                            "custom_tool_use_id": event.id,
                            "content": [{"type": "text", "text": result_str}],
                        })

                    elif event.type == "session.status_idle":
                        stop = event.stop_reason
                        if hasattr(stop, "type") and stop.type == "end_turn":
                            self._log("  Agent finished (end_turn).")
                            return final_text

                # Send any pending tool results
                if pending_tool_results:
                    self._log(f"    Sending {len(pending_tool_results)} tool result(s)...")
                    self.client.beta.sessions.events.send(
                        session_id=session.id,
                        events=pending_tool_results,
                    )
                else:
                    # No tool results and no end_turn — likely done
                    if final_text:
                        return final_text
                    break

            self._log("  Agent reached maximum rounds.")
            return final_text or "Agent reached maximum rounds without completing."

        finally:
            # Cleanup
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
# Local async agent runner (fallback)
# ---------------------------------------------------------------------------

class LocalAgentRunner:
    """
    Runs the agentic tool-use loop locally using ``AsyncAnthropic``.

    This is a fallback for environments where the managed-agents API is
    unavailable. It uses the same tools and system prompt but drives the
    conversation loop client-side.
    """

    def __init__(
        self,
        model: str = MODEL,
        max_turns: int = 25,
        max_tokens: int = MAX_TOKENS,
        verbose: bool = True,
    ):
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.verbose = verbose

    # Strip the "type" key from tool schemas — the messages API uses
    # only name/description/input_schema, not the managed-agents "custom"
    # type marker.
    @staticmethod
    def _messages_api_tools() -> list[dict[str, Any]]:
        return [
            {k: v for k, v in tool.items() if k != "type"}
            for tool in ALL_TOOLS
        ]

    async def run(self, prompt: str) -> str:
        """Run the agentic loop and return the final text response."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        self._log("Starting local agent loop...")

        for turn in range(1, self.max_turns + 1):
            self._log(f"  Turn {turn}/{self.max_turns}")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=self._messages_api_tools(),
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                final_text = _extract_text(response)
                self._log("  Agent finished.")
                return final_text

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type == "tool_use":
                    self._log(f"    -> {block.name}({_summarize_input(block.input)})")
                    result_str = await execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                self._log(f"  Unexpected stop_reason: {response.stop_reason}")
                final_text = _extract_text(response)
                if final_text.strip():
                    return final_text
                break

        self._log("  Agent reached maximum turns without completing.")
        return "Agent reached maximum turns without completing."

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Public agent facade
# ---------------------------------------------------------------------------

class ResearchSummarizerAgent:
    """
    Research Summarizer Agent — accepts URLs and produces structured summaries.

    Uses the Claude Agent SDK (managed-agents beta API) for the agentic loop,
    with automatic fallback to a local async runner when the managed-agents
    API is unavailable.

    Custom tools (fetch_url, tavily_search) are registered with the agent and
    executed locally when the agent requests them.
    """

    def __init__(
        self,
        model: str = MODEL,
        max_turns: int = 25,
        max_tokens: int = MAX_TOKENS,
        verbose: bool = True,
        use_managed: bool = True,
    ):
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.use_managed = use_managed

    async def summarize(
        self,
        urls: list[str],
        topic: Optional[str] = None,
    ) -> SummaryResult:
        """
        Summarize the content at the given URLs.

        Runs the agentic loop (managed or local), executes tool calls
        (fetch_url, tavily_search), and collects the final structured
        summary.

        Args:
            urls: List of URLs to fetch and summarize.
            topic: Optional topic/context to guide the summarization.

        Returns:
            SummaryResult with structured summary data.
        """
        prompt = self._build_prompt(urls, topic)

        if self.use_managed:
            try:
                runner = ManagedAgentRunner(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    verbose=self.verbose,
                )
                final_text = await runner.run(prompt)
                return _parse_result(final_text)
            except (anthropic.APIError, anthropic.APIConnectionError) as exc:
                if self.verbose:
                    print(
                        f"  Managed-agents API unavailable ({type(exc).__name__}), "
                        "falling back to local runner...",
                        file=sys.stderr,
                    )
                # Fall through to local runner

        runner = LocalAgentRunner(
            model=self.model,
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        final_text = await runner.run(prompt)
        return _parse_result(final_text)

    async def run_interactive(self) -> None:
        """Run the agent in interactive mode, accepting URLs from stdin."""
        print("\n" + "=" * 60)
        print("  Research Summarizer Agent - Interactive Mode")
        print("=" * 60)
        print("\nEnter URLs to summarize (one per line, blank line to finish):")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            urls: list[str] = []
            topic: Optional[str] = None

            while True:
                try:
                    line = input("  URL> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    return

                if line.lower() in ("quit", "exit"):
                    print("\nGoodbye!")
                    return
                if not line:
                    break
                urls.append(line)

            if not urls:
                print("No URLs provided. Try again or type 'quit' to exit.\n")
                continue

            try:
                topic_input = input(
                    "  Topic (optional, press Enter to skip)> "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return

            if topic_input:
                topic = topic_input

            print(f"\nSummarizing {len(urls)} URL(s)...\n")

            try:
                result = await self.summarize(urls, topic=topic)
                _print_result(result)
            except Exception as e:
                print(f"\nError: {e}\n")

    @staticmethod
    def _build_prompt(urls: list[str], topic: Optional[str] = None) -> str:
        url_list = "\n".join(f"  {i+1}. {url}" for i, url in enumerate(urls))
        prompt = (
            f"Please fetch and summarize the following {len(urls)} URL(s):\n\n"
            f"{url_list}"
        )
        if topic:
            prompt += f"\n\nFocus area / topic context: {topic}"
        prompt += (
            "\n\nFetch each URL first, analyze its content, then use web search "
            "if you need additional context. After processing all sources, produce "
            "your final structured JSON summary."
        )
        return prompt


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_text(response: Any) -> str:
    """Extract all text content from a Claude response."""
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


def _parse_result(text: str) -> SummaryResult:
    """Parse the agent's final text response into a SummaryResult."""
    result = SummaryResult(raw_response=text)

    try:
        # Try to find JSON in a code fence first
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            # Find the outermost { ... } block
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            if brace_match:
                data = json.loads(brace_match.group(0))
            else:
                data = json.loads(text)

        result.sources = data.get("sources", data.get("source_summaries", []))
        result.synthesis = data.get(
            "cross_source_synthesis", data.get("synthesis", "")
        )
        result.key_takeaways = data.get("key_takeaways", [])
        result.follow_up = data.get(
            "suggested_follow_up", data.get("follow_up", [])
        )
    except (json.JSONDecodeError, AttributeError):
        pass

    return result


# ---------------------------------------------------------------------------
# Logging / display helpers
# ---------------------------------------------------------------------------

def _summarize_input(input_dict: dict[str, Any]) -> str:
    """Create a short summary of tool input for logging."""
    if "url" in input_dict:
        url = input_dict["url"]
        if len(url) > 60:
            url = url[:57] + "..."
        return f'url="{url}"'
    if "query" in input_dict:
        query = input_dict["query"]
        if len(query) > 50:
            query = query[:47] + "..."
        return f'query="{query}"'
    return json.dumps(input_dict, default=str)[:60]


def _print_result(result: SummaryResult) -> None:
    """Pretty-print a SummaryResult to stdout."""
    divider = "=" * 60
    subdiv = "-" * 60

    print(f"\n{divider}")
    print("  RESEARCH SUMMARY")
    print(f"{divider}")

    if result.sources:
        for i, source in enumerate(result.sources, 1):
            print(f"\n{subdiv}")
            print(f"  Source {i}: {source.get('title', 'Untitled')}")
            print(f"  URL: {source.get('url', source.get('source', 'N/A'))}")
            print(subdiv)

            findings = source.get("key_findings", [])
            if findings:
                print("  Key Findings:")
                for f in findings:
                    print(f"    * {f}")

            methodology = source.get("methodology", "")
            if methodology:
                print(f"  Methodology: {methodology}")

            relevance = source.get("relevance", "")
            if relevance:
                print(f"  Relevance: {relevance}")

    if result.synthesis:
        print(f"\n{subdiv}")
        print("  CROSS-SOURCE SYNTHESIS")
        print(subdiv)
        print(f"  {result.synthesis}")

    if result.key_takeaways:
        print(f"\n{subdiv}")
        print("  KEY TAKEAWAYS")
        print(subdiv)
        for i, takeaway in enumerate(result.key_takeaways, 1):
            print(f"  {i}. {takeaway}")

    if result.follow_up:
        print(f"\n{subdiv}")
        print("  SUGGESTED FOLLOW-UP")
        print(subdiv)
        for q in result.follow_up:
            print(f"    -> {q}")

    if not result.sources and result.raw_response:
        print(f"\n  Raw output:\n{result.raw_response}")

    print(f"\n  Timestamp: {result.timestamp}")
    print(f"{divider}\n")
