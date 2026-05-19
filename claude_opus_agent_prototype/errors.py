#!/usr/bin/env python3
"""
Errors — Custom exception hierarchy for the agent prototype.

Provides structured error types so callers can handle different failure
modes (API errors, tool failures, config issues) with appropriate retry
or fallback logic.
"""

from __future__ import annotations

from typing import Any, Optional


class AgentError(Exception):
    """Base exception for all agent errors."""

    def __init__(self, message: str, *, context: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class AgentAPIError(AgentError):
    """Raised when the Anthropic API returns an error."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, retryable: bool = False):
        super().__init__(message, context={"status_code": status_code})
        self.status_code = status_code
        self.retryable = retryable


class ToolExecutionError(AgentError):
    """Raised when a tool call fails."""

    def __init__(self, tool_name: str, message: str):
        super().__init__(f"Tool '{tool_name}' failed: {message}", context={"tool": tool_name})
        self.tool_name = tool_name


class MaxTurnsExceeded(AgentError):
    """Raised when the agent exceeds the configured turn limit."""
    pass


class ConfigError(AgentError):
    """Raised for configuration issues (missing keys, invalid values)."""
    pass
