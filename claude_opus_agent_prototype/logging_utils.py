#!/usr/bin/env python3
"""
Logging — Structured, color-coded logging for agent runs.

Provides a lightweight logger with severity levels and optional color
output. Designed for agent workflows where you want clear, readable
console output showing tool calls, retries, and agent decisions.
"""

from __future__ import annotations

import sys
import time
from enum import IntEnum
from typing import TextIO


class LogLevel(IntEnum):
    """Log severity levels."""
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    SILENT = 4


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_SUPPORTS_COLOR = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
}


def _c(style: str, text: str) -> str:
    if not _SUPPORTS_COLOR:
        return text
    return f"{_COLORS.get(style, '')}{text}{_COLORS['reset']}"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class AgentLogger:
    """Structured logger for agent operations."""

    def __init__(
        self,
        name: str = "agent",
        level: LogLevel = LogLevel.INFO,
        stream: TextIO = sys.stderr,
    ):
        self.name = name
        self.level = level
        self.stream = stream
        self._start_time = time.monotonic()

    def _elapsed(self) -> str:
        elapsed = time.monotonic() - self._start_time
        return f"{elapsed:6.1f}s"

    def _emit(self, level: LogLevel, label: str, color: str, msg: str) -> None:
        if level < self.level:
            return
        ts = self._elapsed()
        prefix = _c("dim", f"[{ts}]") + f" {_c(color, label)}"
        print(f"{prefix} {msg}", file=self.stream)

    def debug(self, msg: str) -> None:
        self._emit(LogLevel.DEBUG, "DEBUG", "dim", msg)

    def info(self, msg: str) -> None:
        self._emit(LogLevel.INFO, " INFO", "cyan", msg)

    def warn(self, msg: str) -> None:
        self._emit(LogLevel.WARN, " WARN", "yellow", msg)

    def error(self, msg: str) -> None:
        self._emit(LogLevel.ERROR, "ERROR", "red", msg)

    def tool_call(self, name: str, summary: str) -> None:
        self._emit(LogLevel.INFO, " TOOL", "green", f"{_c('bold', name)}({summary})")

    def tool_result(self, name: str, chars: int) -> None:
        self._emit(LogLevel.DEBUG, "  <--", "dim", f"{name} returned {chars} chars")

    def turn(self, n: int, max_turns: int) -> None:
        self._emit(LogLevel.INFO, " TURN", "blue", f"{n}/{max_turns}")

    def retry(self, attempt: int, max_retries: int, reason: str) -> None:
        self._emit(
            LogLevel.WARN, "RETRY", "yellow",
            f"Attempt {attempt}/{max_retries}: {reason}",
        )

    def banner(self, text: str) -> None:
        line = "=" * 60
        print(f"\n{_c('dim', line)}", file=self.stream)
        print(f"  {_c('bold', text)}", file=self.stream)
        print(f"{_c('dim', line)}", file=self.stream)

    def reset_timer(self) -> None:
        self._start_time = time.monotonic()


def get_logger(
    name: str = "agent",
    level: LogLevel = LogLevel.INFO,
) -> AgentLogger:
    """Create a new agent logger instance."""
    return AgentLogger(name=name, level=level)
