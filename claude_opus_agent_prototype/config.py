#!/usr/bin/env python3
"""
Configuration — Centralized settings for the Claude Opus agent prototype.

All configurable parameters (model, token limits, retry policy, tool
selection) live here. Values can be overridden via environment variables
or constructor arguments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "claude_opus_agent_prototype" / "output"


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for the Opus agent."""

    # Model settings
    model: str = os.getenv("OPUS_AGENT_MODEL", "claude-opus-4-7-20250501")
    max_tokens: int = int(os.getenv("OPUS_AGENT_MAX_TOKENS", "8192"))
    temperature: float = float(os.getenv("OPUS_AGENT_TEMPERATURE", "0.3"))

    # Agent loop
    max_turns: int = int(os.getenv("OPUS_AGENT_MAX_TURNS", "25"))
    max_retries: int = int(os.getenv("OPUS_AGENT_MAX_RETRIES", "2"))
    retry_delay: float = float(os.getenv("OPUS_AGENT_RETRY_DELAY", "1.0"))

    # Output
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    verbose: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
