"""
Data models for the Memo Generation Agent.

Defines the structured output for the internal memo and the bundled
artifact package that gets stored in S3.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoAudience(str, Enum):
    """Target audience for the memo."""

    EXECUTIVE = "executive"
    ENGINEERING = "engineering"
    PRODUCT = "product"
    ALL_HANDS = "all_hands"


class MemoSection(BaseModel):
    """A section within the formatted memo."""

    heading: str
    content: str
    subsections: list[MemoSection] = Field(default_factory=list)


class InternalMemo(BaseModel):
    """A formatted internal memo generated from synthesis findings."""

    title: str
    subtitle: str = ""
    date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    author: str = "AI Research Pipeline (Agent 3 — Memo Generation)"
    audience: MemoAudience = MemoAudience.PRODUCT
    distribution: str = "Internal — Do Not Distribute"

    # Memo body
    tldr: str = ""  # One-paragraph summary for busy readers
    sections: list[MemoSection] = Field(default_factory=list)

    # Rendered output
    rendered_markdown: str = ""
    rendered_html: str = ""

    # Provenance
    research_source: str = ""
    synthesis_source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_storage_key(self, fmt: str = "markdown") -> str:
        """Generate a storage key for this memo."""
        ts = self.date.strftime("%Y%m%d_%H%M%S")
        ext = "md" if fmt == "markdown" else "html"
        return f"memos/{ts}/memo.{ext}"


class ArtifactBundle(BaseModel):
    """
    Complete artifact bundle containing all pipeline outputs.

    Stored as a single manifest in S3 with paths to each artifact.
    """

    bundle_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Paths/URIs to stored artifacts
    research_report_path: str = ""
    synthesis_report_path: str = ""
    memo_path: str = ""
    memo_html_path: str = ""

    # Inline summaries for quick access
    research_title: str = ""
    synthesis_title: str = ""
    memo_title: str = ""
    executive_summary: str = ""

    # Stats
    trends_analyzed: int = 0
    applications_identified: int = 0
    strategic_themes: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_storage_key(self) -> str:
        """Generate a storage key for this bundle manifest."""
        ts = self.created_at.strftime("%Y%m%d_%H%M%S")
        return f"bundles/{ts}/manifest.json"
