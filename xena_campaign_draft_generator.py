"""
Xena Campaign Draft Generator — AWS Lambda Function

Takes a JSON input representing a new marketing campaign and generates
social media post drafts for Twitter/X, LinkedIn, and Instagram using
Claude (Opus 4.7) on Amazon Bedrock, following a predefined brand voice.

Example event:
{
    "campaign_name": "Xena Summer Launch 2026",
    "product_name": "Xena Analytics Pro",
    "product_description": "AI-powered analytics platform for growth teams",
    "target_audience": "B2B SaaS marketing leaders and growth engineers",
    "key_messages": [
        "Real-time insights in seconds, not hours",
        "AI that explains the why behind the numbers",
        "Integrates with your existing stack in one click"
    ],
    "campaign_objective": "Drive awareness and free trial signups",
    "tone_override": null
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Brand voice (Xena default)
# ---------------------------------------------------------------------------

XENA_BRAND_VOICE = {
    "company": "Xena",
    "tagline": "Intelligence, accelerated.",
    "voice_attributes": [
        "Confident — we speak with authority backed by data",
        "Approachable — no jargon, explain complex topics simply",
        "Forward-looking — focus on what's next, not what was",
        "Human — we're a team of builders, not a faceless corporation",
    ],
    "tone_by_channel": {
        "twitter": "Punchy, conversational, witty. Max 280 chars for main hook.",
        "linkedin": "Professional, thought-leadership, concise. Lead with insight.",
        "instagram": "Visual-first, inspiring, casual. Emphasize the image concept.",
    },
    "writing_rules": {
        "do": [
            "Use active voice",
            "Lead with the insight, not the setup",
            "Include a clear call-to-action",
            "Use data and specifics over vague claims",
            "Keep sentences under 25 words",
        ],
        "dont": [
            "Use buzzwords without substance (synergy, disrupt, etc.)",
            "Over-use exclamation marks (max one per post)",
            "Make unsubstantiated claims",
            "Use ALL CAPS for emphasis",
        ],
    },
    "hashtags": {
        "always_use": ["#Xena", "#IntelligenceAccelerated"],
        "category_tags": [
            "#DataDriven",
            "#AIInsights",
            "#GrowthMarketing",
            "#Analytics",
            "#FutureOfWork",
        ],
    },
    "emoji_usage": "Sparingly (max 2-3 per post). Prefer: rocket, lightning, chart-up, lightbulb, globe.",
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_prompt(event: dict[str, Any]) -> str:
    """Build the Claude prompt from the campaign input and brand voice."""
    campaign_name = event.get("campaign_name", "Untitled Campaign")
    product_name = event.get("product_name", "Xena")
    product_description = event.get("product_description", "")
    target_audience = event.get("target_audience", "")
    key_messages = event.get("key_messages", [])
    campaign_objective = event.get("campaign_objective", "")
    tone_override = event.get("tone_override")

    brand = XENA_BRAND_VOICE
    voice_section = "\n".join(f"- {attr}" for attr in brand["voice_attributes"])
    do_rules = "\n".join(f"- {r}" for r in brand["writing_rules"]["do"])
    dont_rules = "\n".join(f"- {r}" for r in brand["writing_rules"]["dont"])
    key_msgs = "\n".join(f"- {m}" for m in key_messages) if key_messages else "- (none provided)"
    always_tags = " ".join(brand["hashtags"]["always_use"])
    category_tags = " ".join(brand["hashtags"]["category_tags"])

    tone_note = ""
    if tone_override:
        tone_note = f"\n**Tone Override:** {tone_override}\n"

    return f"""You are Xena's senior social media strategist. Generate a set of social media post drafts for a new marketing campaign.

## Campaign Brief
- **Campaign:** {campaign_name}
- **Product:** {product_name}
- **Description:** {product_description}
- **Target Audience:** {target_audience}
- **Objective:** {campaign_objective}

### Key Messages
{key_msgs}

## Brand Voice
**Company:** {brand["company"]}
**Tagline:** "{brand["tagline"]}"

### Voice Attributes
{voice_section}

### Channel Tones
- **Twitter/X:** {brand["tone_by_channel"]["twitter"]}
- **LinkedIn:** {brand["tone_by_channel"]["linkedin"]}
- **Instagram:** {brand["tone_by_channel"]["instagram"]}
{tone_note}
### Writing Rules
**Do:**
{do_rules}

**Don't:**
{dont_rules}

### Hashtags
Always include 1-2 from: {always_tags}
Pick 1-2 relevant from: {category_tags}

### Emoji
{brand["emoji_usage"]}

## Your Task

Generate exactly **3 post drafts** — one each for Twitter/X, LinkedIn, and Instagram.

Return ONLY valid JSON (no markdown fencing, no commentary) with this exact structure:

{{
  "campaign_name": "{campaign_name}",
  "product_name": "{product_name}",
  "generated_at": "<ISO 8601 timestamp>",
  "drafts": [
    {{
      "platform": "twitter",
      "hook": "<the opening line / main hook, max 280 chars>",
      "body": "<full post text including hook>",
      "hashtags": ["#tag1", "#tag2", "#tag3"],
      "suggested_image": "<brief image concept description>",
      "cta": "<call to action>",
      "character_count": <int>,
      "notes": "<strategic rationale for this post>"
    }},
    {{
      "platform": "linkedin",
      "hook": "<bold opening line>",
      "body": "<full post text, 150-300 words>",
      "hashtags": ["#tag1", "#tag2", "#tag3"],
      "suggested_image": "<image concept>",
      "cta": "<call to action>",
      "character_count": <int>,
      "notes": "<strategic rationale>"
    }},
    {{
      "platform": "instagram",
      "hook": "<opening line>",
      "body": "<caption text, visual-first tone>",
      "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"],
      "suggested_image": "<detailed image concept with composition, colors, mood>",
      "cta": "<call to action>",
      "character_count": <int>,
      "notes": "<strategic rationale>"
    }}
  ]
}}

Make each post unique in style and angle while staying on-message. The Twitter post should be punchy and concise, the LinkedIn post should establish thought leadership, and the Instagram post should paint a visual picture. Every post must include a clear CTA."""


# ---------------------------------------------------------------------------
# Bedrock invocation
# ---------------------------------------------------------------------------


def _invoke_bedrock(prompt: str) -> dict[str, Any]:
    """Call Claude on Bedrock and parse the JSON response."""
    bedrock = boto3.client("bedrock-runtime")

    model_id = "us.anthropic.claude-opus-4-7-20250501-v1:0"

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    response = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body),
    )

    response_body = json.loads(response["body"].read())
    raw_text = response_body["content"][0]["text"]

    # Strip markdown fences if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    drafts = json.loads(cleaned)

    # Attach token usage metadata
    usage = response_body.get("usage", {})
    drafts["token_usage"] = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }

    return drafts


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """
    AWS Lambda entry point.

    Accepts a campaign JSON payload and returns social media post drafts
    for Twitter/X, LinkedIn, and Instagram.

    Parameters
    ----------
    event : dict
        Campaign specification. Required fields:
        - campaign_name: str
        - product_name: str
        - product_description: str
        - target_audience: str
        - key_messages: list[str]
        - campaign_objective: str
        Optional:
        - tone_override: str | None

    context : LambdaContext
        AWS Lambda context (unused).

    Returns
    -------
    dict with statusCode, headers, and body (JSON string of drafts).
    """
    logger.info("Received campaign event: %s", json.dumps(event, default=str))

    # Validate required fields
    required = ["campaign_name", "product_name"]
    missing = [f for f in required if not event.get(f)]
    if missing:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": f"Missing required fields: {', '.join(missing)}",
            }),
        }

    try:
        prompt = _build_prompt(event)
        logger.info("Prompt built (%d chars), invoking Bedrock...", len(prompt))

        drafts = _invoke_bedrock(prompt)
        logger.info(
            "Drafts generated: %d posts, %d input tokens, %d output tokens",
            len(drafts.get("drafts", [])),
            drafts.get("token_usage", {}).get("input_tokens", 0),
            drafts.get("token_usage", {}).get("output_tokens", 0),
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(drafts, indent=2, default=str),
        }

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Claude response as JSON: %s", exc)
        return {
            "statusCode": 502,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Model returned invalid JSON",
                "detail": str(exc),
            }),
        }
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Internal error",
                "detail": str(exc),
            }),
        }


# ---------------------------------------------------------------------------
# Local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_event = {
        "campaign_name": "Xena Summer Launch 2026",
        "product_name": "Xena Analytics Pro",
        "product_description": (
            "AI-powered analytics platform that gives growth teams "
            "real-time insights with natural language queries. No SQL required."
        ),
        "target_audience": "B2B SaaS marketing leaders and growth engineers",
        "key_messages": [
            "Real-time insights in seconds, not hours",
            "AI that explains the why behind the numbers",
            "Integrates with your existing stack in one click",
        ],
        "campaign_objective": "Drive awareness and free trial signups for Q3 launch",
    }

    result = lambda_handler(sample_event)
    print(json.dumps(json.loads(result["body"]), indent=2))
