"""
Memo Generation Agent — Produces formatted internal memos from synthesis data.

Takes the SynthesisReport from Agent 2 (Synthesis Agent) and generates
a polished, stakeholder-ready memo. Stores all pipeline outputs
(research, synthesis, memo) in S3.
"""

from .agent import MemoGenerationAgent, generate_memo, run_memo_pipeline
from .config import MemoConfig
from .models import ArtifactBundle, InternalMemo, MemoAudience, MemoSection
from .storage import MemoStorage

__all__ = [
    "ArtifactBundle",
    "InternalMemo",
    "MemoAudience",
    "MemoConfig",
    "MemoGenerationAgent",
    "MemoSection",
    "MemoStorage",
    "generate_memo",
    "run_memo_pipeline",
]
