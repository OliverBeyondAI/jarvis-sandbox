#!/usr/bin/env python3
"""
Channels — Async communication primitives for multi-agent coordination.

Provides typed, asyncio-based channels that allow a dispatcher to
fan-out work to parallel sub-agents and collect results. Inspired by
Go channels and the Claude Agent SDK Channels concept.

Channel types:
  - Channel:       One-to-one async message passing (asyncio.Queue wrapper).
  - FanOutChannel: Dispatcher sends a task to N sub-agents, collects N results.
  - Mailbox:       Named channel registry for dynamic agent-to-agent routing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------

class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"


@dataclass
class Message(Generic[T]):
    """Envelope for inter-agent communication."""
    type: MessageType
    sender: str
    payload: T
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Channel — basic async queue wrapper
# ---------------------------------------------------------------------------

class Channel(Generic[T]):
    """
    Typed async channel for point-to-point message passing.

    Wraps asyncio.Queue with send/receive semantics and an optional
    capacity bound (0 = unbounded).
    """

    def __init__(self, name: str = "", capacity: int = 0):
        self.name = name
        self._queue: asyncio.Queue[Message[T]] = asyncio.Queue(
            maxsize=capacity
        )
        self._closed = False

    async def send(self, message: Message[T]) -> None:
        if self._closed:
            raise ChannelClosedError(f"Channel '{self.name}' is closed")
        await self._queue.put(message)

    async def receive(self, timeout: float | None = None) -> Message[T]:
        if self._closed and self._queue.empty():
            raise ChannelClosedError(f"Channel '{self.name}' is closed and empty")
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ChannelTimeoutError(
                f"Channel '{self.name}' receive timed out after {timeout}s"
            )

    def close(self) -> None:
        self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# FanOutChannel — dispatch one task to N workers, collect N results
# ---------------------------------------------------------------------------

@dataclass
class SubAgentTask:
    """A unit of work dispatched to a sub-agent."""
    agent_id: str
    query: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentResult:
    """Result returned by a sub-agent through a channel."""
    agent_id: str
    query: str
    findings: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    tool_calls_count: int = 0
    error: str | None = None
    duration_seconds: float = 0.0


class FanOutChannel:
    """
    Fan-out / fan-in coordination channel.

    The dispatcher sends N tasks; each sub-agent picks one, researches it,
    and posts its result back. The dispatcher awaits all N results.
    """

    def __init__(self, name: str = "fanout"):
        self.name = name
        self._task_channel: Channel[SubAgentTask] = Channel(f"{name}/tasks")
        self._result_channel: Channel[SubAgentResult] = Channel(f"{name}/results")
        self._expected = 0

    async def dispatch(self, tasks: list[SubAgentTask]) -> None:
        """Send all tasks into the task channel."""
        self._expected = len(tasks)
        for task in tasks:
            await self._task_channel.send(
                Message(
                    type=MessageType.TASK,
                    sender="dispatcher",
                    payload=task,
                )
            )

    async def get_task(self, timeout: float | None = 30.0) -> SubAgentTask:
        """Called by a sub-agent to pick up a task."""
        msg = await self._task_channel.receive(timeout=timeout)
        return msg.payload

    async def submit_result(self, result: SubAgentResult) -> None:
        """Called by a sub-agent to post its result."""
        await self._result_channel.send(
            Message(
                type=MessageType.RESULT,
                sender=result.agent_id,
                payload=result,
            )
        )

    async def collect_all(
        self, timeout: float = 120.0
    ) -> list[SubAgentResult]:
        """
        Wait for all dispatched sub-agents to report back.

        Returns results in arrival order. Raises ChannelTimeoutError if
        not all results arrive within `timeout` seconds.
        """
        results: list[SubAgentResult] = []
        deadline = asyncio.get_event_loop().time() + timeout

        while len(results) < self._expected:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise ChannelTimeoutError(
                    f"FanOutChannel '{self.name}': collected {len(results)}/{self._expected} "
                    f"results before {timeout}s timeout"
                )
            msg = await self._result_channel.receive(timeout=remaining)
            results.append(msg.payload)

        return results

    def close(self) -> None:
        self._task_channel.close()
        self._result_channel.close()


# ---------------------------------------------------------------------------
# Mailbox — named channel registry
# ---------------------------------------------------------------------------

class Mailbox:
    """
    Named channel registry for dynamic agent-to-agent routing.

    Agents register under a name; other agents can send messages to any
    registered name without holding a direct channel reference.
    """

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}

    def register(self, name: str, capacity: int = 0) -> Channel:
        if name in self._channels:
            return self._channels[name]
        ch: Channel = Channel(name=name, capacity=capacity)
        self._channels[name] = ch
        return ch

    def get(self, name: str) -> Channel:
        if name not in self._channels:
            raise KeyError(f"No channel registered for '{name}'")
        return self._channels[name]

    async def send_to(self, name: str, message: Message) -> None:
        ch = self.get(name)
        await ch.send(message)

    async def receive_from(
        self, name: str, timeout: float | None = None
    ) -> Message:
        ch = self.get(name)
        return await ch.receive(timeout=timeout)

    def close_all(self) -> None:
        for ch in self._channels.values():
            ch.close()
        self._channels.clear()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ChannelError(Exception):
    """Base exception for channel operations."""


class ChannelClosedError(ChannelError):
    """Raised when operating on a closed channel."""


class ChannelTimeoutError(ChannelError):
    """Raised when a channel operation times out."""
