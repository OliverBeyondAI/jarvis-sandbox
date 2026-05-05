"""
Configuration for the Memo Generation Agent.

Centralizes model settings, API keys, storage paths, and formatting options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoConfig:
    """Immutable configuration for the memo generation agent."""

    # --- Claude model settings ---
    model: str = "claude-opus-4-7-20250501"
    max_tokens: int = 8192
    max_agent_turns: int = 10

    # --- Anthropic API ---
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    # --- Memo formatting ---
    memo_format: str = field(
        default_factory=lambda: os.environ.get("MEMO_FORMAT", "markdown")
    )  # "markdown" or "html"
    include_appendix: bool = True

    # --- S3 storage ---
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get("MEMO_S3_BUCKET", "trend-research-output")
    )
    s3_prefix: str = field(
        default_factory=lambda: os.environ.get("MEMO_S3_PREFIX", "memos/")
    )
    s3_region: str = field(
        default_factory=lambda: os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )

    # --- Local storage (dev/testing) ---
    storage_local: bool = field(
        default_factory=lambda: os.environ.get("MEMO_LOCAL_STORAGE", "true").lower() == "true"
    )
    local_storage_dir: str = field(
        default_factory=lambda: os.environ.get("MEMO_LOCAL_DIR", "./memo_output")
    )

    @classmethod
    def from_env(cls) -> MemoConfig:
        """Create a MemoConfig instance populated from environment variables."""
        return cls()

    def validate(self) -> list[str]:
        """Return a list of configuration warnings (empty if all OK)."""
        warnings: list[str] = []
        if not self.anthropic_api_key:
            warnings.append("ANTHROPIC_API_KEY not set — Claude calls will fail")
        if not self.storage_local and not self.s3_bucket:
            warnings.append("S3 bucket not configured and local storage disabled")
        if self.memo_format not in ("markdown", "html"):
            warnings.append(f"Unknown memo format: {self.memo_format}")
        return warnings
