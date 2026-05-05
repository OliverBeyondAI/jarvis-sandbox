"""
Configuration for the Trend Research multi-agent system.

Centralizes model settings, API keys, storage paths, and agent parameters.
All sensitive values read from environment variables with safe defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the trend research system."""

    # --- Claude model settings ---
    model: str = "claude-opus-4-7-20250501"
    max_tokens: int = 8192
    max_agent_turns: int = 25

    # --- Tavily API ---
    tavily_api_key: str = field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))
    tavily_max_results: int = 10
    tavily_search_depth: str = "advanced"  # "basic" or "advanced"

    # --- S3 storage ---
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get("TREND_RESEARCH_S3_BUCKET", "trend-research-output")
    )
    s3_prefix: str = field(
        default_factory=lambda: os.environ.get("TREND_RESEARCH_S3_PREFIX", "reports/")
    )
    s3_region: str = field(
        default_factory=lambda: os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    # When True, writes to local filesystem instead of S3 (for dev/testing)
    storage_local: bool = field(
        default_factory=lambda: os.environ.get("TREND_RESEARCH_LOCAL_STORAGE", "true").lower() == "true"
    )
    local_storage_dir: str = field(
        default_factory=lambda: os.environ.get("TREND_RESEARCH_LOCAL_DIR", "./trend_reports")
    )

    # --- Research parameters ---
    max_trends_per_report: int = 10
    max_sources_per_trend: int = 8
    content_max_chars: int = 50_000

    # --- Anthropic API ---
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    @classmethod
    def from_env(cls) -> Config:
        """Create a Config instance populated from environment variables."""
        return cls()

    def validate(self) -> list[str]:
        """Return a list of configuration warnings (empty if all OK)."""
        warnings: list[str] = []
        if not self.tavily_api_key:
            warnings.append("TAVILY_API_KEY not set — web search will fail")
        if not self.anthropic_api_key:
            warnings.append("ANTHROPIC_API_KEY not set — Claude calls will fail")
        if not self.storage_local and not self.s3_bucket:
            warnings.append("S3 bucket not configured and local storage disabled")
        return warnings
