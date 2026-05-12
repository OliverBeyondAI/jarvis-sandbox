"""
Configuration — Brand voice, product/service input, and agent settings.

Frozen dataclass with environment variable defaults and brand voice presets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Brand Voice Presets
# ---------------------------------------------------------------------------

BRAND_VOICE_PRESETS: dict[str, dict[str, Any]] = {
    "professional": {
        "tone": "authoritative, polished, confident",
        "style": "Clear and direct. Use data-driven language. Avoid jargon unless "
                 "speaking to technical audiences. Prefer active voice.",
        "personality": "Trusted advisor — knowledgeable, reliable, forward-thinking",
        "do": [
            "Use strong, decisive language",
            "Lead with value propositions",
            "Back claims with specifics",
            "Address the reader directly",
        ],
        "avoid": [
            "Hyperbole or superlatives without evidence",
            "Slang or overly casual phrasing",
            "Passive constructions",
            "Filler words and hedging",
        ],
    },
    "startup": {
        "tone": "energetic, bold, conversational",
        "style": "Short sentences. Punchy copy. Speak like a smart friend, not a "
                 "corporation. Use metaphors and analogies to simplify complex ideas.",
        "personality": "Ambitious builder — scrappy, transparent, mission-driven",
        "do": [
            "Use conversational language",
            "Show personality and humor where appropriate",
            "Emphasize speed, innovation, and impact",
            "Tell stories and use concrete examples",
        ],
        "avoid": [
            "Corporate buzzwords",
            "Long-winded explanations",
            "Being too formal or stiff",
            "Vague promises without substance",
        ],
    },
    "luxury": {
        "tone": "refined, aspirational, exclusive",
        "style": "Elegant prose with measured pacing. Evoke emotion and sensory "
                 "experience. Every word should feel intentional and curated.",
        "personality": "Curator of excellence — discerning, sophisticated, timeless",
        "do": [
            "Use rich, evocative language",
            "Appeal to aspiration and identity",
            "Emphasize craftsmanship, heritage, and quality",
            "Create a sense of exclusivity",
        ],
        "avoid": [
            "Discounting language or urgency tactics",
            "Technical specifications without context",
            "Casual or colloquial tone",
            "Overexplaining — let the product speak",
        ],
    },
    "technical": {
        "tone": "precise, informative, credible",
        "style": "Lead with capabilities and specifications. Use structured formats "
                 "(bullets, tables, comparisons). Assume an informed audience.",
        "personality": "Expert engineer — detail-oriented, honest, solution-focused",
        "do": [
            "Include specific metrics and benchmarks",
            "Use industry-standard terminology",
            "Compare against alternatives objectively",
            "Provide clear technical differentiators",
        ],
        "avoid": [
            "Marketing fluff without substance",
            "Oversimplifying for a non-technical audience",
            "Emotional appeals over factual ones",
            "Vague performance claims",
        ],
    },
}


# ---------------------------------------------------------------------------
# Product / Service Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductInfo:
    """Describes the product or service the agent is marketing."""

    name: str = "My Product"
    tagline: str = ""
    description: str = ""
    category: str = ""
    target_audience: str = ""
    key_features: list[str] = field(default_factory=list)
    differentiators: list[str] = field(default_factory=list)
    pricing_info: str = ""
    website_url: str = ""
    competitors: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Render product info as a structured block for the system prompt."""
        lines = [
            f"**Product Name:** {self.name}",
        ]
        if self.tagline:
            lines.append(f"**Tagline:** {self.tagline}")
        if self.description:
            lines.append(f"**Description:** {self.description}")
        if self.category:
            lines.append(f"**Category:** {self.category}")
        if self.target_audience:
            lines.append(f"**Target Audience:** {self.target_audience}")
        if self.key_features:
            lines.append("**Key Features:**")
            for f in self.key_features:
                lines.append(f"  - {f}")
        if self.differentiators:
            lines.append("**Differentiators:**")
            for d in self.differentiators:
                lines.append(f"  - {d}")
        if self.pricing_info:
            lines.append(f"**Pricing:** {self.pricing_info}")
        if self.website_url:
            lines.append(f"**Website:** {self.website_url}")
        if self.competitors:
            lines.append(f"**Competitors:** {', '.join(self.competitors)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the Xena Marketing Agent."""

    # Claude model
    model: str = "claude-opus-4-7-20250501"
    max_tokens: int = 16384
    max_agent_turns: int = 30

    # API keys
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    tavily_api_key: str = field(
        default_factory=lambda: os.environ.get("TAVILY_API_KEY", "")
    )

    # Brand voice
    brand_voice_preset: str = "professional"
    brand_voice_custom: dict[str, Any] = field(default_factory=dict)

    # Product
    product: ProductInfo = field(default_factory=ProductInfo)

    # Output
    output_dir: str = field(
        default_factory=lambda: os.environ.get(
            "XENA_OUTPUT_DIR", "./marketing_output"
        )
    )

    # Content generation
    channels: list[str] = field(
        default_factory=lambda: [
            "landing_page",
            "email_sequence",
            "social_media",
            "blog_post",
        ]
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> Config:
        """Create from environment variables with optional overrides."""
        return cls(**overrides)

    @property
    def brand_voice(self) -> dict[str, Any]:
        """Resolve the active brand voice (custom overrides preset)."""
        base = BRAND_VOICE_PRESETS.get(
            self.brand_voice_preset,
            BRAND_VOICE_PRESETS["professional"],
        )
        if self.brand_voice_custom:
            merged = {**base, **self.brand_voice_custom}
            return merged
        return base

    def brand_voice_prompt_block(self) -> str:
        """Render brand voice as a structured block for the system prompt."""
        voice = self.brand_voice
        lines = [
            f"**Tone:** {voice.get('tone', '')}",
            f"**Style:** {voice.get('style', '')}",
            f"**Personality:** {voice.get('personality', '')}",
        ]
        do_items = voice.get("do", [])
        if do_items:
            lines.append("**Do:**")
            for item in do_items:
                lines.append(f"  - {item}")
        avoid_items = voice.get("avoid", [])
        if avoid_items:
            lines.append("**Avoid:**")
            for item in avoid_items:
                lines.append(f"  - {item}")
        return "\n".join(lines)

    def validate(self) -> list[str]:
        """Return warnings for missing config."""
        warnings: list[str] = []
        if not self.anthropic_api_key:
            warnings.append("ANTHROPIC_API_KEY not set — Claude calls will fail")
        if not self.tavily_api_key:
            warnings.append("TAVILY_API_KEY not set — market research will fail")
        if self.brand_voice_preset not in BRAND_VOICE_PRESETS:
            warnings.append(
                f"Unknown brand voice preset '{self.brand_voice_preset}' — "
                f"falling back to 'professional'"
            )
        return warnings
