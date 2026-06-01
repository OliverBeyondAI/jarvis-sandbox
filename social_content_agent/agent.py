"""Core social content generation agent using the Claude API."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import anthropic

from .models import (
    BrandVoice,
    ProductInfo,
    SocialPost,
    WeeklyContentPlan,
)

TOOL_DEFINITIONS = [
    {
        "name": "analyze_audience",
        "description": (
            "Analyze the target audience to identify their social media habits, "
            "preferred platforms, content preferences, and engagement patterns. "
            "Use this to inform the content strategy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "audience_description": {
                    "type": "string",
                    "description": "Description of the target audience",
                },
                "analysis": {
                    "type": "string",
                    "description": "Your detailed analysis of this audience's social media behavior",
                },
                "preferred_platforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ranked list of best platforms for this audience",
                },
                "peak_engagement_times": {
                    "type": "object",
                    "description": "Best posting times by day of week",
                    "additionalProperties": {"type": "string"},
                },
                "content_preferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of content this audience engages with most",
                },
            },
            "required": ["audience_description", "analysis", "preferred_platforms"],
        },
    },
    {
        "name": "define_content_strategy",
        "description": (
            "Define the overarching content strategy for the week, including "
            "content pillars, themes per day, and how they build on each other "
            "to create a cohesive narrative arc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_overview": {
                    "type": "string",
                    "description": "High-level strategy explanation",
                },
                "content_pillars": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 content pillars that guide the week's content",
                },
                "weekly_arc": {
                    "type": "string",
                    "description": "How the week's content builds a narrative arc",
                },
                "day_themes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "integer"},
                            "day_label": {"type": "string"},
                            "theme": {"type": "string"},
                            "platform": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["day", "day_label", "theme", "platform"],
                    },
                    "description": "Theme and platform assignment for each day",
                },
            },
            "required": ["strategy_overview", "content_pillars", "day_themes"],
        },
    },
    {
        "name": "generate_post",
        "description": (
            "Generate a single social media post with caption, image description, "
            "hashtags, and posting recommendations. Each post should align with "
            "the content strategy and brand voice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "integer", "description": "Day number (1-7)"},
                "day_label": {"type": "string", "description": "Day of week"},
                "platform": {"type": "string", "description": "Target platform"},
                "theme": {"type": "string", "description": "Content theme for this post"},
                "caption": {
                    "type": "string",
                    "description": "The full post caption/copy",
                },
                "image_description": {
                    "type": "string",
                    "description": (
                        "Detailed description of the suggested image/visual. "
                        "Include composition, colors, mood, subjects, and style."
                    ),
                },
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant hashtags for the post",
                },
                "best_time_to_post": {
                    "type": "string",
                    "description": "Recommended posting time with timezone",
                },
                "engagement_hook": {
                    "type": "string",
                    "description": "A question or CTA to drive engagement",
                },
            },
            "required": [
                "day", "day_label", "platform", "theme",
                "caption", "image_description", "hashtags",
            ],
        },
    },
    {
        "name": "finalize_plan",
        "description": (
            "Finalize the weekly content plan after all posts are generated. "
            "Review for consistency, variety, and brand alignment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "review_notes": {
                    "type": "string",
                    "description": "Final review notes on the content plan",
                },
                "adjustments_made": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any adjustments made during review",
                },
            },
            "required": ["review_notes"],
        },
    },
]


def build_system_prompt(product: ProductInfo, brand_voice: BrandVoice) -> str:
    """Build the system prompt with product and brand context."""
    return f"""You are a world-class social media strategist and content creator. Your task is to generate a full 7-day social media content plan for a product/service.

## Product/Service Information
{product.to_prompt()}

## Brand Voice Specification
{brand_voice.to_prompt()}

## Your Workflow

You MUST follow these steps in order, using the provided tools:

1. **Audience Analysis** — Use `analyze_audience` to deeply understand the target audience's social media behavior, preferred platforms, and content preferences.

2. **Content Strategy** — Use `define_content_strategy` to create a cohesive week-long content strategy with content pillars, daily themes, and a narrative arc that builds momentum through the week.

3. **Post Generation** — Use `generate_post` exactly 7 times (once per day, Monday through Sunday) to create each day's content. Each post must include:
   - A compelling caption tailored to the platform
   - A detailed image/visual description (composition, colors, mood, subjects, style)
   - Relevant hashtags
   - Best posting time
   - An engagement hook (question or CTA)

4. **Finalization** — Use `finalize_plan` to review the complete plan for consistency and quality.

## Content Guidelines

- Each day should feature a DIFFERENT platform (rotate through the most relevant ones)
- Vary content themes: mix educational, promotional, storytelling, engagement, and inspirational content
- Image descriptions should be vivid and specific enough for a designer or AI image generator to create
- Captions should match the brand voice exactly — tone, emoji usage, and hashtag style must be consistent
- Build a narrative arc across the week: awareness → interest → engagement → conversion
- Include platform-specific best practices (character limits, format preferences, etc.)
- Make every post feel authentic, not corporate or templated

Begin by analyzing the audience, then build your strategy, then generate all 7 posts, then finalize."""


@dataclass
class AgentMetrics:
    """Track agent execution metrics."""
    total_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def _log(msg: str) -> None:
    """Log to stderr so stdout stays clean for output."""
    print(msg, file=sys.stderr)


def execute_tool(name: str, input_data: dict) -> str:
    """Execute a tool call and return the result as a string."""
    # All tools are "thinking" tools — the agent structures its work through them.
    # We simply acknowledge the tool call and return the input for the agent to use.
    if name == "analyze_audience":
        _log(f"  [tool] Analyzing audience: {input_data.get('audience_description', '')[:80]}...")
        return json.dumps({"status": "success", "message": "Audience analysis recorded."})

    elif name == "define_content_strategy":
        pillars = input_data.get("content_pillars", [])
        _log(f"  [tool] Content strategy defined with {len(pillars)} pillars")
        return json.dumps({"status": "success", "message": "Content strategy recorded."})

    elif name == "generate_post":
        day = input_data.get("day", "?")
        platform = input_data.get("platform", "?")
        _log(f"  [tool] Generated post: Day {day} — {platform}")
        return json.dumps({"status": "success", "message": f"Post for day {day} recorded."})

    elif name == "finalize_plan":
        _log(f"  [tool] Plan finalized")
        return json.dumps({"status": "success", "message": "Plan finalized."})

    else:
        return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})


def run_agent(
    product: ProductInfo,
    brand_voice: BrandVoice,
    model: str = "claude-opus-4-7-20250501",
    max_turns: int = 30,
) -> WeeklyContentPlan:
    """Run the social content agent and return a WeeklyContentPlan."""
    client = anthropic.Anthropic()
    system_prompt = build_system_prompt(product, brand_voice)
    messages = [{"role": "user", "content": "Generate the 7-day social media content plan now. Follow all steps in order."}]

    metrics = AgentMetrics()
    plan = WeeklyContentPlan(product=product, brand_voice=brand_voice)

    _log(f"\n{'='*60}")
    _log(f"Social Content Agent — Generating 7-day plan for {product.name}")
    _log(f"Model: {model}")
    _log(f"{'='*60}\n")

    for turn in range(max_turns):
        metrics.total_turns += 1
        _log(f"[Turn {turn + 1}]")

        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        metrics.input_tokens += response.usage.input_tokens
        metrics.output_tokens += response.usage.output_tokens

        # Process response content
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                _log(f"  [agent] {block.text[:120]}...")

            elif block.type == "tool_use":
                metrics.tool_calls += 1
                tool_input = block.input

                # Capture structured data from tool calls
                if block.name == "define_content_strategy":
                    plan.strategy_notes = tool_input.get("strategy_overview", "")
                    plan.content_pillars = tool_input.get("content_pillars", [])

                elif block.name == "generate_post":
                    post = SocialPost(
                        day=tool_input.get("day", 0),
                        day_label=tool_input.get("day_label", ""),
                        platform=tool_input.get("platform", ""),
                        theme=tool_input.get("theme", ""),
                        caption=tool_input.get("caption", ""),
                        image_description=tool_input.get("image_description", ""),
                        hashtags=tool_input.get("hashtags", []),
                        best_time_to_post=tool_input.get("best_time_to_post", ""),
                        engagement_hook=tool_input.get("engagement_hook", ""),
                    )
                    plan.posts.append(post)

                result_str = execute_tool(block.name, tool_input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

        # If the agent is done (no more tool calls), break
        if response.stop_reason == "end_turn":
            _log(f"\n[Agent completed in {metrics.total_turns} turns, {metrics.tool_calls} tool calls]")
            _log(f"[Tokens: {metrics.input_tokens:,} input, {metrics.output_tokens:,} output]")
            break

        # Continue the conversation with tool results
        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    return plan
