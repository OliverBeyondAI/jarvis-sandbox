"""
Social Content Agent — 7-day social media content generation using the
Anthropic beta managed-agents API.

Uses Claude Opus 4.7 by default with custom tool dispatch for structured
content generation: audience analysis, strategy definition, post generation,
and plan finalization.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .models import (
    BrandVoice,
    SocialPost,
    SocialProductInfo,
    WeeklyContentPlan,
)

# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-opus-4-7-20250501"

# ---------------------------------------------------------------------------
# Tool Definitions (type: "custom" for managed-agents API)
# ---------------------------------------------------------------------------

SOCIAL_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "analyze_audience",
        "type": "custom",
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
        "type": "custom",
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
        "type": "custom",
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
        "type": "custom",
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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def build_social_system_prompt(product: SocialProductInfo, brand_voice: BrandVoice) -> str:
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


# ---------------------------------------------------------------------------
# Tool execution (thinking tools — acknowledge and return)
# ---------------------------------------------------------------------------


def _execute_social_tool(name: str, input_data: dict) -> str:
    """Execute a social content tool call and return acknowledgment."""
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class SocialAgentMetrics:
    """Track agent execution metrics."""
    total_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Main agent runner (managed-agents API)
# ---------------------------------------------------------------------------


def run_social_agent(
    product: SocialProductInfo,
    brand_voice: BrandVoice,
    model: str = DEFAULT_MODEL,
) -> WeeklyContentPlan:
    """Run the social content agent via managed-agents API and return a WeeklyContentPlan."""
    client = anthropic.Anthropic()
    system_prompt = build_social_system_prompt(product, brand_voice)

    metrics = SocialAgentMetrics()
    plan = WeeklyContentPlan(product=product, brand_voice=brand_voice)

    _log(f"\n{'='*60}")
    _log(f"Social Content Agent — Generating 7-day plan for {product.name}")
    _log(f"Model: {model}")
    _log(f"API: Managed Agents (beta)")
    _log(f"{'='*60}\n")

    # Step 1: Create managed agent
    agent = client.beta.agents.create(
        model=model,
        name="social-content-agent",
        description="7-day social media content plan generator",
        system=system_prompt,
        tools=SOCIAL_TOOL_DEFINITIONS,
    )
    _log(f"  Agent created: {agent.id}")

    # Step 2: Create environment
    environment = client.beta.environments.create(
        name="social-content-env",
    )

    # Step 3: Create session
    session = client.beta.sessions.create(
        agent={"agent_id": agent.id},
        environment_id=environment.id,
    )
    _log(f"  Session created: {session.id}")

    # Step 4: Send user message
    client.beta.sessions.events.send(
        session_id=session.id,
        events=[{
            "type": "user.message",
            "content": [{
                "type": "text",
                "text": "Generate the 7-day social media content plan now. Follow all steps in order.",
            }],
        }],
    )

    # Step 5: Event loop — stream events, handle tool calls
    _run_social_event_loop(client, session.id, plan, metrics)

    _log(f"\n[Agent completed: {metrics.tool_calls} tool calls]")
    _log(f"[Tokens: {metrics.input_tokens:,} input, {metrics.output_tokens:,} output]")

    # Clean up
    try:
        client.beta.sessions.delete(session_id=session.id)
        client.beta.environments.delete(environment_id=environment.id)
        client.beta.agents.archive(agent_id=agent.id)
    except Exception:
        pass

    return plan


def _run_social_event_loop(
    client: anthropic.Anthropic,
    session_id: str,
    plan: WeeklyContentPlan,
    metrics: SocialAgentMetrics,
) -> None:
    """Stream events from managed session, dispatching custom tool calls."""

    while True:
        pending_tool_calls: list[dict[str, Any]] = []

        with client.beta.sessions.events.stream(
            session_id=session_id,
            timeout=600.0,
        ) as stream:
            for event in stream:
                if event.type == "agent.message":
                    for block in event.content:
                        if hasattr(block, "text") and block.text:
                            snippet = block.text[:120]
                            if len(block.text) > 120:
                                snippet += "..."
                            _log(f"  [agent] {snippet}")

                elif event.type == "agent.custom_tool_use":
                    metrics.tool_calls += 1
                    tool_input = dict(event.input)

                    # Capture structured data
                    if event.name == "define_content_strategy":
                        plan.strategy_notes = tool_input.get("strategy_overview", "")
                        plan.content_pillars = tool_input.get("content_pillars", [])

                    elif event.name == "generate_post":
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

                    pending_tool_calls.append({
                        "id": event.id,
                        "name": event.name,
                        "input": tool_input,
                    })

                elif event.type == "span.model_request_end":
                    metrics.total_turns += 1
                    if hasattr(event, "model_usage") and event.model_usage:
                        usage = event.model_usage
                        if hasattr(usage, "input_tokens"):
                            metrics.input_tokens += usage.input_tokens or 0
                        if hasattr(usage, "output_tokens"):
                            metrics.output_tokens += usage.output_tokens or 0

                elif event.type == "session.status_idle":
                    stop_type = getattr(event.stop_reason, "type", "unknown")
                    if stop_type == "end_turn":
                        return
                    elif stop_type == "requires_action":
                        break
                    else:
                        return

                elif event.type in ("session.error", "session.status_terminated"):
                    return

        # Execute pending tools and send results
        if not pending_tool_calls:
            return

        tool_result_events = []
        for tc in pending_tool_calls:
            result_str = _execute_social_tool(tc["name"], tc["input"])
            tool_result_events.append({
                "type": "user.custom_tool_result",
                "custom_tool_use_id": tc["id"],
                "content": [{"type": "text", "text": result_str}],
            })

        client.beta.sessions.events.send(
            session_id=session_id,
            events=tool_result_events,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Log to stderr so stdout stays clean for output."""
    print(msg, file=sys.stderr)
