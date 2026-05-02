#!/usr/bin/env python3
"""
Sub-Agents — Parallel research workers coordinated by the dispatcher.

Each sub-agent receives a focused research query via a channel, performs
web search and URL fetching using the shared tool infrastructure, and
posts structured findings back through the result channel.

Supports both managed-agents API and local async runner modes.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from agents.tools import ALL_TOOLS, execute_tool

from .channels import FanOutChannel, SubAgentResult, SubAgentTask


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7-20250501"
MAX_TOKENS = 4096

SUB_AGENT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a focused research sub-agent. Your job is to thoroughly research
    a specific query and return structured findings.

    You have access to these tools:
    1. **tavily_search** — Search the web for current information.
    2. **fetch_url** — Fetch and read the full text content of a web page.

    ## Research Protocol

    1. Start with 1-2 targeted web searches on your assigned query.
    2. Read the most promising 2-3 sources using fetch_url.
    3. Synthesize your findings into a clear, factual summary.

    ## Output Format

    When you have completed your research, respond with ONLY a JSON block:

    ```json
    {
      "summary": "Clear 2-4 paragraph summary of your findings",
      "key_facts": ["fact 1", "fact 2", "fact 3"],
      "sources": [
        {"title": "Source Title", "url": "https://..."}
      ]
    }
    ```

    Be thorough but focused. Stick to your assigned query — do not wander
    into tangential topics. Cite your sources accurately.
""")


# ---------------------------------------------------------------------------
# Research sub-agent (local async runner)
# ---------------------------------------------------------------------------

class ResearchSubAgent:
    """
    A single research sub-agent that picks a task from the channel,
    researches it via tool-use, and posts results back.
    """

    def __init__(
        self,
        agent_id: str,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        max_turns: int = 15,
        verbose: bool = True,
    ):
        self.agent_id = agent_id
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.verbose = verbose

    async def run_task(self, task: SubAgentTask) -> SubAgentResult:
        """
        Execute a research task: run the agentic tool-use loop,
        extract structured findings, and return a SubAgentResult.
        """
        start = time.monotonic()
        self._log(f"Starting research: {task.query[:80]}...")

        prompt = f"Research the following query thoroughly:\n\n{task.query}"
        if task.context:
            prompt += f"\n\nAdditional context: {json.dumps(task.context)}"

        messages_api_tools = [
            {k: v for k, v in tool.items() if k != "type"}
            for tool in ALL_TOOLS
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        tool_calls_count = 0

        try:
            for turn in range(1, self.max_turns + 1):
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SUB_AGENT_SYSTEM_PROMPT,
                    tools=messages_api_tools,
                    messages=messages,
                )

                if response.stop_reason == "end_turn":
                    text = self._extract_text(response)
                    duration = time.monotonic() - start
                    self._log(f"Finished in {duration:.1f}s ({tool_calls_count} tool calls)")
                    return self._parse_result(task, text, tool_calls_count, duration)

                # Process tool calls
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results: list[dict[str, Any]] = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_calls_count += 1
                        self._log(f"  [{self.agent_id}] Tool: {block.name}({self._summarize(block.input)})")
                        result_str = await execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                else:
                    text = self._extract_text(response)
                    duration = time.monotonic() - start
                    return self._parse_result(task, text, tool_calls_count, duration)

            duration = time.monotonic() - start
            return SubAgentResult(
                agent_id=self.agent_id,
                query=task.query,
                findings="Research reached maximum turns without completing.",
                tool_calls_count=tool_calls_count,
                duration_seconds=duration,
                error="max_turns_reached",
            )

        except Exception as e:
            duration = time.monotonic() - start
            self._log(f"Error: {type(e).__name__}: {e}")
            return SubAgentResult(
                agent_id=self.agent_id,
                query=task.query,
                error=f"{type(e).__name__}: {e}",
                duration_seconds=duration,
            )

    def _parse_result(
        self,
        task: SubAgentTask,
        text: str,
        tool_calls_count: int,
        duration: float,
    ) -> SubAgentResult:
        """Parse agent output into a structured SubAgentResult."""
        sources: list[dict[str, str]] = []
        findings = text

        # Try to extract JSON from the response
        parsed = self._extract_json(text)
        if parsed:
            findings = parsed.get("summary", text)
            key_facts = parsed.get("key_facts", [])
            if key_facts:
                findings += "\n\nKey Facts:\n" + "\n".join(
                    f"• {f}" for f in key_facts
                )
            sources = parsed.get("sources", [])

        return SubAgentResult(
            agent_id=self.agent_id,
            query=task.query,
            findings=findings,
            sources=sources,
            tool_calls_count=tool_calls_count,
            duration_seconds=duration,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON from agent response (code fences or bare braces)."""
        # Try code-fenced JSON first
        match = re.search(r"```(?:json)?\s*\n?({.*?})\s*\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try bare JSON object
        match = re.search(r"\{[^{}]*\"summary\"[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Try the most aggressive: find outermost braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _extract_text(response: Any) -> str:
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _summarize(input_dict: dict[str, Any]) -> str:
        if "url" in input_dict:
            url = input_dict["url"]
            return f'url="{url[:50]}..."' if len(url) > 50 else f'url="{url}"'
        if "query" in input_dict:
            q = input_dict["query"]
            return f'query="{q[:40]}..."' if len(q) > 40 else f'query="{q}"'
        return json.dumps(input_dict, default=str)[:50]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [{self.agent_id}] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Channel-connected sub-agent worker
# ---------------------------------------------------------------------------

async def run_sub_agent_worker(
    agent_id: str,
    channel: FanOutChannel,
    model: str = MODEL,
    verbose: bool = True,
) -> None:
    """
    Worker coroutine: picks a task from the FanOutChannel, runs research,
    and posts the result back. Designed to be launched with asyncio.gather().
    """
    agent = ResearchSubAgent(agent_id=agent_id, model=model, verbose=verbose)

    try:
        task = await channel.get_task(timeout=30.0)
        result = await agent.run_task(task)
        await channel.submit_result(result)
    except Exception as e:
        error_result = SubAgentResult(
            agent_id=agent_id,
            query="unknown",
            error=f"Worker failed: {type(e).__name__}: {e}",
        )
        await channel.submit_result(error_result)
