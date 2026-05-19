#!/usr/bin/env python3
"""
OpusAgent — Core agent with structured logging, retries, and error handling.

Drives a multi-turn tool-use conversation loop using the Anthropic SDK.
Supports configurable retry logic, structured result output, and
graceful degradation on API errors.
"""

from __future__ import annotations

import asyncio
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import anthropic

from .config import AgentConfig
from .errors import AgentAPIError, MaxTurnsExceeded, ToolExecutionError
from .logging_utils import AgentLogger, LogLevel, get_logger
from .tools import TOOL_SCHEMAS, execute_tool, summarize_input


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert research agent powered by Claude Opus 4.6.

    You have access to the following tools:

    1. **tavily_search** — Search the web for current information.
    2. **fetch_url** — Fetch and read the content of a web page.
    3. **read_file** — Read a file from the local filesystem.
    4. **write_file** — Write content to a file.

    When researching a topic:
    - Search for multiple perspectives and authoritative sources.
    - Cross-reference findings across sources.
    - Cite your sources with URLs.
    - Provide a clear, well-structured response.
    - If you write output to a file, confirm the path and summarize what was written.
""")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Structured output from an agent run."""
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turns_used: int = 0
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OpusAgent:
    """
    Claude Opus 4.6 research agent with tool use.

    Features:
    - Multi-turn agentic loop with tool calls
    - Structured logging with color-coded output
    - Configurable retry on transient API errors
    - Graceful error handling with typed exceptions
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        logger: Optional[AgentLogger] = None,
    ):
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt
        self.logger = logger or get_logger(
            level=LogLevel.INFO if self.config.verbose else LogLevel.WARN,
        )
        self.client = anthropic.AsyncAnthropic()

    async def run(self, prompt: str) -> AgentResult:
        """
        Run the agent with the given prompt.

        Retries on transient API errors up to config.max_retries times.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 2):  # +2 for 0-indexed + initial
            try:
                return await self._run_loop(prompt)
            except AgentAPIError as e:
                last_error = e
                if not e.retryable or attempt > self.config.max_retries:
                    raise
                self.logger.retry(attempt, self.config.max_retries, str(e))
                await asyncio.sleep(self.config.retry_delay * attempt)

        # Should not reach here, but just in case
        raise last_error or AgentAPIError("Agent failed after retries")

    async def _run_loop(self, prompt: str) -> AgentResult:
        """Core agentic tool-use loop."""
        self.logger.reset_timer()
        self.logger.banner(f"Opus Agent — {self.config.model}")
        self.logger.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]
        result = AgentResult()
        t0 = time.monotonic()

        for turn in range(1, self.config.max_turns + 1):
            self.logger.turn(turn, self.config.max_turns)

            try:
                response = await self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=self.system_prompt,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
            except anthropic.RateLimitError as e:
                raise AgentAPIError(str(e), status_code=429, retryable=True) from e
            except anthropic.InternalServerError as e:
                raise AgentAPIError(str(e), status_code=500, retryable=True) from e
            except anthropic.APIError as e:
                status = getattr(e, "status_code", None)
                raise AgentAPIError(str(e), status_code=status, retryable=False) from e

            # Check for end of conversation
            if response.stop_reason == "end_turn":
                result.text = self._extract_text(response)
                result.turns_used = turn
                result.elapsed_seconds = time.monotonic() - t0
                self.logger.info(f"Done in {turn} turn(s), {result.elapsed_seconds:.1f}s")
                return result

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type == "tool_use":
                    self.logger.tool_call(block.name, summarize_input(block.input))
                    result.tool_calls.append({"name": block.name, "input": block.input})

                    result_str = await execute_tool(block.name, block.input)
                    self.logger.tool_result(block.name, len(result_str))

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                # No tool calls and not end_turn — extract what we have
                result.text = self._extract_text(response)
                result.turns_used = turn
                result.elapsed_seconds = time.monotonic() - t0
                return result

        # Exceeded max turns
        result.turns_used = self.config.max_turns
        result.elapsed_seconds = time.monotonic() - t0
        result.text = self._extract_text(response) if response else ""
        result.error = "Exceeded maximum turns"
        self.logger.warn(f"Reached max turns ({self.config.max_turns})")
        raise MaxTurnsExceeded(
            f"Agent exceeded {self.config.max_turns} turns without completing"
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull all text blocks from a Claude response."""
        return "\n".join(
            block.text for block in response.content if hasattr(block, "text")
        )
