"""
Claude Opus Agent Prototype — Agentic research agent built on Claude Opus 4.6.

Demonstrates multi-step research workflows with tool integration,
structured logging, and robust error handling using the Anthropic SDK.
"""

from .agent import OpusAgent, AgentResult
from .config import AgentConfig
from .logging_utils import get_logger, LogLevel

__all__ = [
    "AgentConfig",
    "AgentResult",
    "LogLevel",
    "OpusAgent",
    "get_logger",
]
