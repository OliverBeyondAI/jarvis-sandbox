#!/usr/bin/env python3
"""
MCP Server — Marketing tools exposed via the Model Context Protocol.

Provides six tools for the marketing agent workflow:
  1. market_research  — Web search for market intelligence (Tavily)
  2. fetch_url        — Read full web page content
  3. analyze_market   — Record structured market analysis findings
  4. draft_content    — Record a drafted marketing content piece
  5. generate_image   — Social media-ready images with text/branding (OpenAI GPT Image 1)
  6. save_campaign    — Save the complete campaign document

Run with:
    python mcp_server.py                  # stdio transport (for MCP clients)
    python mcp_server.py --transport sse  # SSE transport (HTTP, port 8000)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="xena-marketing-tools",
    instructions=(
        "This server provides marketing research, content drafting, image "
        "generation, and campaign management tools for the Xena Marketing Agent."
    ),
)


# ---------------------------------------------------------------------------
# Tool: market_research
# ---------------------------------------------------------------------------

@mcp.tool()
def market_research(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> str:
    """Search the web for market intelligence using Tavily.

    Searches for market trends, competitor activity, audience insights,
    and industry data. Use targeted queries to gather information that
    will inform the marketing strategy and messaging.

    Args:
        query: Targeted search query for market research
            (e.g. "AI productivity tools market trends 2025",
            "competitor analysis project management SaaS").
        max_results: Maximum results to return (1-10). Defaults to 5.
        search_depth: Search depth — "basic" for fast results or
            "advanced" for deeper analysis. Defaults to "advanced".

    Returns:
        JSON string with query results including title, URL, snippet,
        and relevance score for each result.
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient()
        results = client.search(
            query=query,
            max_results=min(max_results, 10),
            search_depth=search_depth,
        )
        items = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "relevance_score": min(max(r.get("score", 0), 0.0), 1.0),
            }
            for r in results.get("results", [])
        ]
        return json.dumps({
            "query": query,
            "results": items,
            "result_count": len(items),
        }, indent=2)
    except ImportError:
        return json.dumps({
            "query": query,
            "error": "tavily-python not installed. Install with: pip install tavily-python",
        })
    except Exception as e:
        return json.dumps({"query": query, "error": f"Search failed: {e}"})


# ---------------------------------------------------------------------------
# Tool: fetch_url
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_url(url: str) -> str:
    """Fetch the full text content of a web page.

    Use this to deep-dive into competitor websites, industry reports,
    or product pages to extract messaging patterns, positioning, and
    market data. HTML is stripped to return clean text.

    Args:
        url: The URL to fetch content from (e.g. "https://competitor.com/pricing").

    Returns:
        JSON string with the page content, status code, content type,
        and content length. Content is truncated at 50,000 characters.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup

        with httpx.Client(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "MCPXenaMarketingAgent/1.0",
                "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
            },
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            text = response.text
            if "text/html" in content_type:
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)

            max_chars = 50_000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Content truncated at 50,000 characters]"

            return json.dumps({
                "url": url,
                "status": response.status_code,
                "content_type": content_type.split(";")[0].strip(),
                "content": text,
                "length": len(text),
            })
    except Exception as e:
        return json.dumps({"url": url, "error": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Tool: analyze_market
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_market(
    phase: str,
    insights: list[dict[str, Any]],
    gaps: list[str] | None = None,
) -> str:
    """Record structured market analysis findings.

    Use this after each research phase to organize insights before
    generating content. Tracks audience segments, competitor positioning,
    messaging opportunities, and market gaps.

    Args:
        phase: Current analysis phase. One of: "market_landscape",
            "competitor_analysis", "audience_research",
            "messaging_strategy", "content_planning".
        insights: List of insight objects, each with:
            - headline (str, required): One-line insight summary
            - detail (str): Supporting detail or evidence
            - type (str): One of "market_trend", "competitor_intel",
              "audience_insight", "messaging_opportunity", "positioning_gap", "risk"
            - source_urls (list[str]): URLs supporting this insight
        gaps: Research gaps needing additional investigation.

    Returns:
        JSON string confirming the insights were recorded.
    """
    return json.dumps({
        "phase": phase,
        "insights_recorded": len(insights),
        "insights": insights,
        "gaps": gaps or [],
        "status": "recorded",
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: draft_content
# ---------------------------------------------------------------------------

@mcp.tool()
def draft_content(
    channel: str,
    title: str,
    body: str,
    cta: str = "",
    notes: str = "",
) -> str:
    """Record a drafted piece of marketing content for a specific channel.

    Use this to save each content piece as you generate it — landing page
    copy, email sequences, social posts, blog outlines, ad copy, etc.

    Args:
        channel: Marketing channel. One of: "landing_page", "email_sequence",
            "social_media", "blog_post", "ad_copy", "press_release",
            "product_description".
        title: Title or headline for the content piece.
        body: The full content body (markdown formatted).
        cta: Call-to-action text (e.g. "Start your free trial").
        notes: Internal notes about this content piece.

    Returns:
        JSON string confirming the content was drafted.
    """
    return json.dumps({
        "channel": channel,
        "title": title,
        "body_length": len(body),
        "cta": cta,
        "notes": notes,
        "status": "drafted",
    }, indent=2)


# ---------------------------------------------------------------------------
# Social media platform presets
# ---------------------------------------------------------------------------

SOCIAL_PRESETS: dict[str, dict[str, Any]] = {
    "instagram_post": {
        "size": "1024x1024",
        "label": "Instagram Post (1:1)",
        "text_placement": "center",
        "safe_zone": "keep text within the central 80% of the image",
    },
    "instagram_story": {
        "size": "1024x1536",
        "label": "Instagram Story / Reels (9:16)",
        "text_placement": "upper_third",
        "safe_zone": "keep text in the upper 40% — bottom is covered by UI overlays",
    },
    "twitter_post": {
        "size": "1536x1024",
        "label": "Twitter / X Post (3:2)",
        "text_placement": "center_left",
        "safe_zone": "keep text in the left two-thirds, leave right side for visual balance",
    },
    "linkedin_post": {
        "size": "1536x1024",
        "label": "LinkedIn Post (3:2)",
        "text_placement": "center",
        "safe_zone": "keep text within central 70%, professional clean margins",
    },
    "facebook_ad": {
        "size": "1024x1024",
        "label": "Facebook Ad (1:1)",
        "text_placement": "center",
        "safe_zone": "minimal text — Facebook penalizes images with >20% text coverage",
    },
    "blog_header": {
        "size": "1536x1024",
        "label": "Blog Header (3:2)",
        "text_placement": "center",
        "safe_zone": "keep text centered with generous padding for overlay readability",
    },
    "landing_hero": {
        "size": "1536x1024",
        "label": "Landing Page Hero (3:2)",
        "text_placement": "left",
        "safe_zone": "keep text in the left half — right side for product imagery or negative space",
    },
}


def _build_marketing_prompt(
    concept: str,
    headline: str = "",
    tagline: str = "",
    brand_name: str = "",
    color_palette: str = "",
    platform: str = "",
    style: str = "natural",
) -> str:
    """Build a detailed image generation prompt from marketing inputs.

    Combines the visual concept with text rendering instructions, brand
    elements, and platform-specific layout guidance into a single prompt
    optimized for GPT Image 1's text rendering capabilities.
    """
    parts: list[str] = []

    # Core visual concept
    parts.append(f"Create a professional marketing image: {concept}")

    # Embedded text instructions (GPT Image 1 can render text)
    text_elements: list[str] = []
    if headline:
        text_elements.append(
            f'Display the headline "{headline}" in bold, modern sans-serif typography '
            f"as the primary text element. Make it large, highly legible, and visually prominent."
        )
    if tagline:
        text_elements.append(
            f'Include the tagline "{tagline}" in a smaller, complementary font below the headline.'
        )
    if brand_name and brand_name.lower() not in (headline or "").lower():
        text_elements.append(
            f'Include the brand name "{brand_name}" as a subtle watermark or logo-style text '
            f"in a corner of the image."
        )
    if text_elements:
        parts.append(
            "TEXT RENDERING (render this text directly on the image, not as a placeholder): "
            + " ".join(text_elements)
        )

    # Color palette / brand colors
    if color_palette:
        parts.append(f"Color palette: {color_palette}. Use these as the dominant colors.")

    # Platform-specific layout
    preset = SOCIAL_PRESETS.get(platform, {})
    if preset:
        parts.append(
            f"Layout optimized for {preset['label']}. "
            f"Text placement: {preset.get('text_placement', 'center')}. "
            f"{preset.get('safe_zone', '')}"
        )

    # Style guidance
    style_instructions = {
        "natural": (
            "Photorealistic, clean, and polished. Professional commercial photography aesthetic. "
            "Soft, even lighting with subtle depth of field."
        ),
        "vivid": (
            "Bold, vibrant, and eye-catching. High contrast colors with dynamic composition. "
            "Stylized and attention-grabbing for social media feeds."
        ),
        "minimal": (
            "Minimalist design with generous whitespace. Simple geometric elements. "
            "Clean typography on a subtle gradient or solid background."
        ),
        "editorial": (
            "Editorial magazine quality. Sophisticated composition with intentional negative space. "
            "Muted, cohesive color grading."
        ),
    }
    parts.append(f"Visual style: {style_instructions.get(style, style_instructions['natural'])}")

    # Universal quality instructions
    parts.append(
        "The image must look like a professional agency-produced marketing asset. "
        "All text must be spelled correctly, sharp, and fully readable. "
        "No placeholder text, lorem ipsum, or garbled characters."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool: generate_image
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_image(
    concept: str,
    headline: str = "",
    tagline: str = "",
    brand_name: str = "",
    platform: str = "instagram_post",
    color_palette: str = "",
    style: str = "natural",
    size: str = "",
    channel: str = "",
) -> str:
    """Generate a social media-ready marketing image with embedded text and branding.

    Creates professional marketing visuals with rendered headline text, taglines,
    and brand elements. Optimized for specific social media platforms with
    correct dimensions and text-safe zones. Uses OpenAI GPT Image 1 for
    high-quality text rendering directly in the image.

    Args:
        concept: The core visual concept for the image. Describe the scene,
            mood, and composition you want.
            (e.g. "Futuristic workspace with holographic project boards floating
            above a clean desk, cool blue and purple tones, tech-forward feel")
        headline: Primary text to render on the image — the main marketing
            message or hook. Keep it short (3-8 words) for maximum impact.
            (e.g. "Ship Faster with AI", "Energy Without the Crash")
        tagline: Secondary text rendered below the headline. Supports the
            headline with a brief value proposition or call-to-action.
            (e.g. "Try FlowDesk free for 14 days", "Powered by adaptogens")
        brand_name: Brand or product name to include as a subtle branding
            element (watermark or logo-style text in a corner).
        platform: Target social media platform — determines image dimensions
            and text-safe zones. One of: "instagram_post" (1:1),
            "instagram_story" (9:16), "twitter_post" (3:2), "linkedin_post" (3:2),
            "facebook_ad" (1:1), "blog_header" (3:2), "landing_hero" (3:2).
            Defaults to "instagram_post".
        color_palette: Brand colors to use as dominant palette.
            (e.g. "deep navy #1B2A4A, electric blue #3B82F6, white #FFFFFF")
        style: Visual style preset. One of: "natural" (photorealistic),
            "vivid" (bold and vibrant), "minimal" (clean whitespace),
            "editorial" (magazine quality). Defaults to "natural".
        size: Override image dimensions (e.g. "1024x1024", "1024x1536",
            "1536x1024"). If empty, uses the platform's default size.
        channel: Marketing channel label for file naming (e.g. "social_media",
            "landing_page"). Falls back to platform name.

    Returns:
        JSON string with the image file path, resolved prompt, platform info,
        and generation metadata, or an error if image generation is unavailable.
    """
    # Resolve size from platform preset if not explicitly set
    preset = SOCIAL_PRESETS.get(platform, {})
    resolved_size = size or preset.get("size", "1024x1024")
    resolved_channel = channel or platform

    # Build the full prompt with text/branding instructions
    full_prompt = _build_marketing_prompt(
        concept=concept,
        headline=headline,
        tagline=tagline,
        brand_name=brand_name,
        color_palette=color_palette,
        platform=platform,
        style=style,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return json.dumps({
            "status": "skipped",
            "concept": concept,
            "headline": headline,
            "platform": platform,
            "resolved_prompt": full_prompt,
            "reason": "OPENAI_API_KEY not set — image generation unavailable",
        })

    try:
        import base64
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        result = client.images.generate(
            model="gpt-image-1",
            prompt=full_prompt,
            n=1,
            size=resolved_size,
            quality="high",
        )

        # Save the image
        output_dir = Path(
            os.environ.get("XENA_OUTPUT_DIR", "./marketing_output")
        ) / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        channel_slug = resolved_channel.replace(" ", "_")
        filename = f"{channel_slug}_{timestamp}.png"
        filepath = output_dir / filename

        # gpt-image-1 returns base64 by default
        image_data = result.data[0]
        if hasattr(image_data, "b64_json") and image_data.b64_json:
            image_bytes = base64.b64decode(image_data.b64_json)
            filepath.write_bytes(image_bytes)
        elif hasattr(image_data, "url") and image_data.url:
            import httpx

            resp = httpx.get(image_data.url)
            filepath.write_bytes(resp.content)

        return json.dumps({
            "status": "generated",
            "file_path": str(filepath.resolve()),
            "concept": concept,
            "headline": headline,
            "tagline": tagline,
            "brand_name": brand_name,
            "platform": platform,
            "platform_label": preset.get("label", platform),
            "size": resolved_size,
            "style": style,
            "color_palette": color_palette,
            "resolved_prompt": full_prompt,
            "model": "gpt-image-1",
        }, indent=2)
    except ImportError:
        return json.dumps({
            "status": "error",
            "error": "openai package not installed. Install with: pip install openai",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "concept": concept,
            "headline": headline,
            "error": f"{type(e).__name__}: {e}",
        })


# ---------------------------------------------------------------------------
# Tool: save_campaign
# ---------------------------------------------------------------------------

@mcp.tool()
def save_campaign(filename: str, content: str) -> str:
    """Save the complete marketing campaign as a markdown file.

    Call this exactly ONCE when all content has been drafted, reviewed,
    and finalized. The content should be the full campaign document
    compiled from all drafted content pieces.

    Args:
        filename: Filename for the campaign document
            (e.g. "acme_launch_campaign.md"). The .md extension is
            added automatically if missing.
        content: Full markdown content of the campaign document.

    Returns:
        JSON string confirming the file was saved with its path and size.
    """
    target_dir = os.environ.get("XENA_OUTPUT_DIR", "./marketing_output")
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".md"):
        filename = f"{filename}.md"

    filepath = path / filename
    filepath.write_text(content, encoding="utf-8")

    return json.dumps({
        "saved": True,
        "path": str(filepath.resolve()),
        "size_bytes": len(content.encode("utf-8")),
    }, indent=2)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    mcp.run(transport=transport)
