"""
Configuration and constants for the OphthoFlow Prior Auth Agent.
"""

import os

# ---------------------------------------------------------------------------
# Model Configuration (Bedrock)
# ---------------------------------------------------------------------------

MODEL_BEDROCK = "us.anthropic.claude-opus-4-7-20250501-v1:0"
MODEL_DIRECT = "claude-opus-4-7-20250501"

BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_TOKENS = 8192

# ---------------------------------------------------------------------------
# Agent Defaults
# ---------------------------------------------------------------------------

USE_BEDROCK = os.environ.get(
    "OPHTHOFLOW_USE_BEDROCK", "true"
).lower() in ("1", "true", "yes")
