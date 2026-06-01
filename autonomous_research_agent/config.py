"""
Configuration — Frozen dataclass with environment variable defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the Autonomous Research Agent."""

    # Claude model
    model: str = "claude-opus-4-7-20250501"
    max_tokens: int = 16384
    max_agent_turns: int = 40

    # Tavily API
    tavily_api_key: str = field(
        default_factory=lambda: os.environ.get("TAVILY_API_KEY", "")
    )
    tavily_max_results: int = 8
    tavily_search_depth: str = "advanced"

    # Anthropic API
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    # Output
    output_dir: str = field(
        default_factory=lambda: os.environ.get(
            "RESEARCH_OUTPUT_DIR", "./research_reports"
        )
    )

    # Research behavior
    min_searches: int = 4
    min_deep_dives: int = 2
    max_sub_queries: int = 6

    @classmethod
    def from_env(cls) -> Config:
        """Create from environment variables."""
        return cls()

    def validate(self) -> list[str]:
        """Return warnings for missing config."""
        warnings: list[str] = []
        if not self.tavily_api_key:
            warnings.append("TAVILY_API_KEY not set — web search will fail")
        if not self.anthropic_api_key:
            warnings.append("ANTHROPIC_API_KEY not set — Claude calls will fail")
        return warnings
