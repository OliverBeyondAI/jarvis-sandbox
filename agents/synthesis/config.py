"""
Configuration for the Synthesis Agent.

Extends the trend research config pattern with synthesis-specific settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SynthesisConfig:
    """Immutable configuration for the synthesis agent."""

    # --- Claude model settings ---
    model: str = "claude-opus-4-7-20250501"
    max_tokens: int = 8192
    max_agent_turns: int = 15

    # --- Anthropic API ---
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    # --- Storage ---
    storage_local: bool = field(
        default_factory=lambda: os.environ.get("SYNTHESIS_LOCAL_STORAGE", "true").lower() == "true"
    )
    local_storage_dir: str = field(
        default_factory=lambda: os.environ.get("SYNTHESIS_LOCAL_DIR", "./synthesis_reports")
    )

    # --- S3 storage ---
    s3_bucket: str = field(
        default_factory=lambda: os.environ.get("SYNTHESIS_S3_BUCKET", "trend-research-output")
    )
    s3_prefix: str = field(
        default_factory=lambda: os.environ.get("SYNTHESIS_S3_PREFIX", "synthesis/")
    )
    s3_region: str = field(
        default_factory=lambda: os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )

    @classmethod
    def from_env(cls) -> SynthesisConfig:
        """Create a SynthesisConfig instance populated from environment variables."""
        return cls()

    def validate(self) -> list[str]:
        """Return a list of configuration warnings (empty if all OK)."""
        warnings: list[str] = []
        if not self.anthropic_api_key:
            warnings.append("ANTHROPIC_API_KEY not set — Claude calls will fail")
        if not self.storage_local and not self.s3_bucket:
            warnings.append("S3 bucket not configured and local storage disabled")
        return warnings
