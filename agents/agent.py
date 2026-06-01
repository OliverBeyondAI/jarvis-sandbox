#!/usr/bin/env python3
"""
Agent — Core agent logic with managed and local runners.

Provides both a managed-agents runner (Claude Agent SDK) and a local
fallback runner, with automatic failover. Reuses tool infrastructure
from agents.tools.
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

MODEL = "claude-opus-4-7-20250501"  # Upgraded to Opus 4.7
MAX_TOKENS = 8192

DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful AI agent with access to web and file tools.

    You have access to the following tools:

    1. **fetch_url** — Fetch the text content of a web page URL.
    2. **tavily_search** — Search the web for current information on any topic.
    3. **read_file** — Read a file from the local filesystem.
    4. **write_file** — Write content to a file on the local filesystem.

    Use these tools as needed to answer the user's request thoroughly and
    accurately. Cite sources when using information from the web.
""")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Structured output from an agent run."""
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Managed-agents runner (Claude Agent SDK)
# ---------------------------------------------------------------------------

class ManagedAgentRunner:
    """
    Runs an agent via the Claude Agent SDK managed-agents API.

    Flow:
      1. Create an Agent with custom tools.
      2. Create an Environment and Session.
      3. Send a user message and stream events.
      4. Handle tool calls and return results.
      5. Clean up resources.
    """

    def __init__(
        self,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = True,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.verbose = verbose

    async def run(self, prompt: str) -> AgentResult:
        """Send prompt through a managed-agent session and return the result."""
        agent = None
        environment = None
        result = AgentResult()

        try:
            self._log("Creating managed agent...")
            agent = self.client.beta.agents.create(
                model=self.model,
                name="agent",
                description="AI agent with web search and URL fetching tools.",
                system=self.system_prompt,
                tools=ALL_TOOLS,
            )
            self._log(f"  Agent created: {agent.id}")

            self._log("Creating environment...")
            environment = self.client.beta.environments.create(
                name="agent-env",
            )
            self._log(f"  Environment created: {environment.id}")

            self._log("Creating session...")
            session = self.client.beta.sessions.create(
                agent=agent.id,
                environment_id=environment.id,
            )
            self._log(f"  Session created: {session.id}")

            self._log("Sending user message...")
            self.client.beta.sessions.events.send(
                session_id=session.id,
                events=[{
                    "type": "user.message",
                    "content": [{"type": "text", "text": prompt}],
                }],
            )

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

                        result.tool_calls.append({"name": tool_name, "input": tool_input})
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
# Local async agent runner (fallback)
# ---------------------------------------------------------------------------

class LocalAgentRunner:
    """
    Runs the agentic tool-use loop locally using AsyncAnthropic.

    Fallback for environments where the managed-agents API is unavailable.
    Same tools and system prompt, but drives the conversation loop client-side.
    """

    def __init__(
        self,
        model: str = MODEL,
        max_turns: int = 25,
        max_tokens: int = MAX_TOKENS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = True,
    ):
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.verbose = verbose

    @staticmethod
    def _messages_api_tools() -> list[dict[str, Any]]:
        """Strip 'type' key from tool schemas for the messages API."""
        return [
            {k: v for k, v in tool.items() if k != "type"}
            for tool in ALL_TOOLS
        ]

    async def run(self, prompt: str) -> AgentResult:
        """Run the agentic loop and return the result."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]
        result = AgentResult()

        self._log("Starting local agent loop...")

        for turn in range(1, self.max_turns + 1):
            self._log(f"  Turn {turn}/{self.max_turns}")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=self._messages_api_tools(),
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                result.text = _extract_text(response)
                self._log("  Agent finished.")
                return result

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type == "tool_use":
                    self._log(f"    -> {block.name}({_summarize_input(block.input)})")
                    result.tool_calls.append({"name": block.name, "input": block.input})
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
# Public agent facade
# ---------------------------------------------------------------------------

class Agent:
    """
    AI Agent with web search and URL fetching tools.

    Uses the Claude Agent SDK (managed-agents beta API) for the agentic loop,
    with automatic fallback to a local async runner when the managed-agents
    API is unavailable.
    """

    def __init__(
        self,
        model: str = MODEL,
        max_turns: int = 25,
        max_tokens: int = MAX_TOKENS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        verbose: bool = True,
        use_managed: bool = True,
    ):
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.use_managed = use_managed

    async def run(self, prompt: str) -> AgentResult:
        """
        Run the agent with the given prompt.

        Attempts the managed-agents API first, falling back to the local
        runner if unavailable.
        """
        if self.use_managed:
            try:
                runner = ManagedAgentRunner(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system_prompt=self.system_prompt,
                    verbose=self.verbose,
                )
                return await runner.run(prompt)
            except (anthropic.APIError, anthropic.APIConnectionError) as exc:
                if self.verbose:
                    print(
                        f"  Managed-agents API unavailable ({type(exc).__name__}), "
                        "falling back to local runner...",
                        file=sys.stderr,
                    )

        runner = LocalAgentRunner(
            model=self.model,
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            verbose=self.verbose,
        )
        return await runner.run(prompt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(response: Any) -> str:
    """Extract all text content from a Claude response."""
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


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
