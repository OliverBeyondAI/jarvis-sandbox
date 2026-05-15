"""
Tool Schemas — Anthropic custom tool definitions for the managed-agents API.

These schemas are used when creating the managed agent (non-MCP path).
The MCP server provides the same tools via the MCP protocol, but these
schemas are needed for the managed-agents API which requires explicit
tool definitions at agent creation time.

The agent supports two modes:
  1. MCP mode (default) — tools are discovered from the MCP server
  2. Managed mode — tools are defined here and executed locally
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic custom tool format for managed-agents API)
# ---------------------------------------------------------------------------

MARKET_RESEARCH_TOOL: dict[str, Any] = {
    "name": "market_research",
    "type": "custom",
    "description": (
        "Search the web for market intelligence — trends, competitor activity, "
        "audience insights, and industry data. Use targeted queries to gather "
        "information that will inform the marketing strategy and messaging."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Targeted search query for market research.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (1-10).",
                "default": 5,
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "default": "advanced",
            },
        },
        "required": ["query"],
    },
}

FETCH_URL_TOOL: dict[str, Any] = {
    "name": "fetch_url",
    "type": "custom",
    "description": (
        "Fetch the full text content of a web page. Use this to deep-dive "
        "into competitor websites, industry reports, or product pages to "
        "extract messaging patterns, positioning, and market data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch content from.",
            },
        },
        "required": ["url"],
    },
}

ANALYZE_MARKET_TOOL: dict[str, Any] = {
    "name": "analyze_market",
    "type": "custom",
    "description": (
        "Record structured market analysis findings — audience segments, "
        "competitor positioning, messaging opportunities, and market gaps."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "description": "Current analysis phase.",
                "enum": [
                    "market_landscape",
                    "competitor_analysis",
                    "audience_research",
                    "messaging_strategy",
                    "content_planning",
                ],
            },
            "insights": {
                "type": "array",
                "description": "List of market insights from this phase.",
                "items": {
                    "type": "object",
                    "properties": {
                        "headline": {
                            "type": "string",
                            "description": "One-line insight summary.",
                        },
                        "detail": {
                            "type": "string",
                            "description": "Supporting detail or evidence.",
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "market_trend",
                                "competitor_intel",
                                "audience_insight",
                                "messaging_opportunity",
                                "positioning_gap",
                                "risk",
                            ],
                            "default": "market_trend",
                        },
                        "source_urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs supporting this insight.",
                        },
                    },
                    "required": ["headline"],
                },
            },
            "gaps": {
                "type": "array",
                "description": "Research gaps needing additional investigation.",
                "items": {"type": "string"},
            },
        },
        "required": ["phase", "insights"],
    },
}

DRAFT_CONTENT_TOOL: dict[str, Any] = {
    "name": "draft_content",
    "type": "custom",
    "description": (
        "Record a drafted piece of marketing content for a specific channel."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": "Marketing channel for this content.",
                "enum": [
                    "landing_page",
                    "email_sequence",
                    "social_media",
                    "blog_post",
                    "ad_copy",
                    "press_release",
                    "product_description",
                ],
            },
            "title": {
                "type": "string",
                "description": "Title or headline for the content piece.",
            },
            "body": {
                "type": "string",
                "description": "The full content body (markdown formatted).",
            },
            "cta": {
                "type": "string",
                "description": "Call-to-action text.",
            },
            "notes": {
                "type": "string",
                "description": "Internal notes about this content piece.",
            },
        },
        "required": ["channel", "title", "body"],
    },
}

GENERATE_IMAGE_TOOL: dict[str, Any] = {
    "name": "generate_image",
    "type": "custom",
    "description": (
        "Generate a social media-ready marketing image with embedded text and "
        "branding using OpenAI GPT Image 1. Creates professional visuals with "
        "rendered headlines, taglines, and brand elements optimized for specific "
        "social media platforms with correct dimensions and text-safe zones."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "concept": {
                "type": "string",
                "description": (
                    "The core visual concept — describe the scene, mood, and "
                    "composition (e.g. 'Futuristic workspace with holographic "
                    "project boards, cool blue tones, tech-forward feel')."
                ),
            },
            "headline": {
                "type": "string",
                "description": (
                    "Primary text to render on the image (3-8 words). "
                    "This is the main marketing message or hook."
                ),
            },
            "tagline": {
                "type": "string",
                "description": (
                    "Secondary text below the headline — a brief value "
                    "proposition or call-to-action."
                ),
            },
            "brand_name": {
                "type": "string",
                "description": (
                    "Brand or product name for subtle branding (watermark "
                    "or logo-style text in a corner)."
                ),
            },
            "platform": {
                "type": "string",
                "description": (
                    "Target platform — determines dimensions and text-safe zones."
                ),
                "enum": [
                    "instagram_post",
                    "instagram_story",
                    "twitter_post",
                    "linkedin_post",
                    "facebook_ad",
                    "blog_header",
                    "landing_hero",
                ],
                "default": "instagram_post",
            },
            "color_palette": {
                "type": "string",
                "description": (
                    "Brand colors as dominant palette "
                    "(e.g. 'deep navy #1B2A4A, electric blue #3B82F6, white')."
                ),
            },
            "style": {
                "type": "string",
                "enum": ["natural", "vivid", "minimal", "editorial"],
                "default": "natural",
                "description": (
                    "Visual style: natural (photorealistic), vivid (bold), "
                    "minimal (clean whitespace), editorial (magazine quality)."
                ),
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1024x1536", "1536x1024"],
                "description": (
                    "Override image dimensions. If empty, uses the platform's default."
                ),
            },
            "channel": {
                "type": "string",
                "description": "Marketing channel label for file naming.",
            },
        },
        "required": ["concept"],
    },
}

SAVE_CAMPAIGN_TOOL: dict[str, Any] = {
    "name": "save_campaign",
    "type": "custom",
    "description": (
        "Save the complete marketing campaign as a markdown file. "
        "Call this exactly ONCE when all content has been finalized."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename for the campaign (e.g. 'acme_launch_campaign.md').",
            },
            "content": {
                "type": "string",
                "description": "Full markdown content of the campaign document.",
            },
        },
        "required": ["filename", "content"],
    },
}

# All tools for the managed-agents API
ALL_TOOLS: list[dict[str, Any]] = [
    MARKET_RESEARCH_TOOL,
    FETCH_URL_TOOL,
    ANALYZE_MARKET_TOOL,
    DRAFT_CONTENT_TOOL,
    GENERATE_IMAGE_TOOL,
    SAVE_CAMPAIGN_TOOL,
]

# Tool names for quick lookup
TOOL_NAMES: set[str] = {t["name"] for t in ALL_TOOLS}
