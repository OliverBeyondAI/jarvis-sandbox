"""Feedback loop for iterative content refinement using managed-agents API.

The agent evaluates generated posts against brand voice guidelines and
engagement best practices, then refines posts that don't meet the quality
threshold until they pass or the maximum number of rounds is reached.

Uses the Anthropic beta managed-agents API for both evaluator and refiner
agent sessions.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from typing import Any

import anthropic

from .models import (
    BrandVoice,
    PostFeedback,
    SocialProductInfo,
    RefinedContentPlan,
    RefinementRound,
    SocialPost,
    WeeklyContentPlan,
)
from .social_agent import DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASS_THRESHOLD = 7  # Overall score >= 7 means the post passes
DEFAULT_MAX_ROUNDS = 3

# ---------------------------------------------------------------------------
# Tool definitions for the evaluate phase
# ---------------------------------------------------------------------------

EVALUATE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "evaluate_post",
        "type": "custom",
        "description": (
            "Evaluate a single social media post against brand voice guidelines "
            "and engagement best practices. Provide scores (1-10) and specific "
            "actionable feedback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "integer", "description": "Day number of the post being evaluated"},
                "platform": {"type": "string", "description": "Platform of the post"},
                "brand_voice_score": {
                    "type": "integer",
                    "description": "Score 1-10: How well does the post match the brand voice spec?",
                },
                "engagement_score": {
                    "type": "integer",
                    "description": "Score 1-10: How likely is this post to drive engagement?",
                },
                "clarity_score": {
                    "type": "integer",
                    "description": "Score 1-10: How clear and compelling is the messaging?",
                },
                "overall_score": {
                    "type": "integer",
                    "description": "Score 1-10: Overall quality assessment",
                },
                "brand_voice_issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific ways the post deviates from brand voice guidelines",
                },
                "engagement_issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific engagement weaknesses",
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What the post does well — preserve these in any refinement",
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete, actionable suggestions for improving the post",
                },
                "verdict": {
                    "type": "string",
                    "enum": ["pass", "refine"],
                    "description": "pass if overall_score >= 7 and no critical issues; refine otherwise",
                },
            },
            "required": [
                "day", "platform", "brand_voice_score", "engagement_score",
                "clarity_score", "overall_score", "brand_voice_issues",
                "engagement_issues", "strengths", "suggestions", "verdict",
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool definitions for the refine phase
# ---------------------------------------------------------------------------

REFINE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "refine_post",
        "type": "custom",
        "description": (
            "Submit a refined version of a social media post that addresses "
            "all feedback issues while preserving identified strengths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "integer", "description": "Day number (1-7)"},
                "day_label": {"type": "string", "description": "Day of week"},
                "platform": {"type": "string", "description": "Target platform"},
                "theme": {"type": "string", "description": "Content theme"},
                "caption": {
                    "type": "string",
                    "description": "The refined post caption/copy",
                },
                "image_description": {
                    "type": "string",
                    "description": "Refined image/visual description",
                },
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Refined hashtags",
                },
                "best_time_to_post": {
                    "type": "string",
                    "description": "Recommended posting time",
                },
                "engagement_hook": {
                    "type": "string",
                    "description": "Refined engagement hook (question or CTA)",
                },
            },
            "required": [
                "day", "day_label", "platform", "theme",
                "caption", "image_description", "hashtags",
                "engagement_hook",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _post_to_text(post: SocialPost) -> str:
    """Render a post as readable text for the evaluator."""
    parts = [
        f"Day {post.day} ({post.day_label}) — {post.platform}",
        f"Theme: {post.theme}",
        f"Caption:\n{post.caption}",
        f"Image Description:\n{post.image_description}",
        f"Hashtags: {' '.join(post.hashtags)}",
    ]
    if post.best_time_to_post:
        parts.append(f"Best Time: {post.best_time_to_post}")
    if post.engagement_hook:
        parts.append(f"Engagement Hook: {post.engagement_hook}")
    return "\n".join(parts)


def _build_evaluate_system(product: SocialProductInfo, brand_voice: BrandVoice) -> str:
    return f"""You are a senior social media strategist acting as a QUALITY REVIEWER. You are evaluating social media posts for brand voice alignment and engagement potential.

## Brand Voice Guidelines (the standard you evaluate against)
{brand_voice.to_prompt()}

## Product Context
{product.to_prompt()}

## Evaluation Criteria

### Brand Voice (score 1-10)
- Does the tone match the specified brand voice?
- Is the personality consistent with guidelines?
- Does emoji usage match the "{brand_voice.emoji_usage}" directive?
- Does hashtag usage match the "{brand_voice.hashtag_style}" directive?
- Are "do use" elements present: {', '.join(brand_voice.do_use) if brand_voice.do_use else 'N/A'}?
- Are "avoid" elements absent: {', '.join(brand_voice.avoid) if brand_voice.avoid else 'N/A'}?

### Engagement Potential (score 1-10)
- Is there a strong opening hook that stops the scroll?
- Is the CTA/engagement hook clear and compelling?
- Does the post feel native to the platform?
- Would the target audience want to share, save, or comment?
- Is the image description compelling and specific enough?

### Clarity (score 1-10)
- Is the value proposition clear within the first line or two?
- Is the messaging concise and free of fluff?
- Is the image description detailed enough for a designer/AI to execute?

## Instructions
You will receive a set of social media posts. Use the `evaluate_post` tool ONCE for EACH post. Be specific in your feedback — vague feedback like "could be better" is not useful.

A post PASSES (verdict: "pass") if overall_score >= 7 and there are no critical brand voice violations.
A post needs REFINEMENT (verdict: "refine") otherwise."""


def _build_refine_system(
    product: SocialProductInfo,
    brand_voice: BrandVoice,
) -> str:
    return f"""You are a senior social media copywriter. Your job is to REFINE social media posts based on specific feedback from a quality reviewer.

## Brand Voice Guidelines (you MUST follow these)
{brand_voice.to_prompt()}

## Product Context
{product.to_prompt()}

## Instructions
You will receive posts that need refinement along with detailed feedback. For each post, use the `refine_post` tool to submit an improved version that:

1. **Addresses every issue** listed in the feedback
2. **Preserves all strengths** the reviewer identified
3. **Follows every suggestion** unless it contradicts the brand voice
4. **Stays true to the brand voice** — this is non-negotiable
5. **Maintains the same theme and platform** — don't change the content strategy, only the execution

Be bold in your refinements. A timid edit that barely changes anything is worse than a confident rewrite that nails the brand voice."""


# ---------------------------------------------------------------------------
# Managed-agents event loop helper
# ---------------------------------------------------------------------------


def _run_session_loop(
    client: anthropic.Anthropic,
    session_id: str,
    tool_handler,
) -> tuple[int, int]:
    """Run a managed-agents event loop, dispatching tool calls via tool_handler.

    Returns (input_tokens, output_tokens).
    """
    total_in = 0
    total_out = 0

    while True:
        pending_tool_calls: list[dict[str, Any]] = []

        with client.beta.sessions.events.stream(
            session_id=session_id,
            timeout=600.0,
        ) as stream:
            for event in stream:
                if event.type == "agent.custom_tool_use":
                    pending_tool_calls.append({
                        "id": event.id,
                        "name": event.name,
                        "input": dict(event.input),
                    })

                elif event.type == "span.model_request_end":
                    if hasattr(event, "model_usage") and event.model_usage:
                        usage = event.model_usage
                        if hasattr(usage, "input_tokens"):
                            total_in += usage.input_tokens or 0
                        if hasattr(usage, "output_tokens"):
                            total_out += usage.output_tokens or 0

                elif event.type == "session.status_idle":
                    stop_type = getattr(event.stop_reason, "type", "unknown")
                    if stop_type == "requires_action":
                        break
                    return total_in, total_out

                elif event.type in ("session.error", "session.status_terminated"):
                    return total_in, total_out

        if not pending_tool_calls:
            return total_in, total_out

        tool_result_events = []
        for tc in pending_tool_calls:
            result = tool_handler(tc["name"], tc["input"])
            tool_result_events.append({
                "type": "user.custom_tool_result",
                "custom_tool_use_id": tc["id"],
                "content": [{"type": "text", "text": result if isinstance(result, str) else json.dumps(result)}],
            })

        client.beta.sessions.events.send(
            session_id=session_id,
            events=tool_result_events,
        )


# ---------------------------------------------------------------------------
# Evaluate phase
# ---------------------------------------------------------------------------

def _evaluate_posts(
    client: anthropic.Anthropic,
    posts: list[SocialPost],
    product: SocialProductInfo,
    brand_voice: BrandVoice,
    model: str,
) -> tuple[list[PostFeedback], int, int]:
    """Run the evaluator agent on a list of posts via managed-agents API."""
    system = _build_evaluate_system(product, brand_voice)

    # Create evaluator agent
    agent = client.beta.agents.create(
        model=model,
        name="post-evaluator",
        description="Social media post quality evaluator",
        system=system,
        tools=EVALUATE_TOOLS,
    )
    environment = client.beta.environments.create(name="eval-env")
    session = client.beta.sessions.create(
        agent={"agent_id": agent.id},
        environment_id=environment.id,
    )

    posts_text = "\n\n---\n\n".join(_post_to_text(p) for p in sorted(posts, key=lambda p: p.day))
    user_msg = f"Evaluate each of the following {len(posts)} social media posts:\n\n{posts_text}"

    client.beta.sessions.events.send(
        session_id=session.id,
        events=[{
            "type": "user.message",
            "content": [{"type": "text", "text": user_msg}],
        }],
    )

    feedbacks: list[PostFeedback] = []

    def handle_evaluate(name: str, inp: dict) -> str:
        if name == "evaluate_post":
            fb = PostFeedback(
                day=inp.get("day", 0),
                platform=inp.get("platform", ""),
                brand_voice_score=inp.get("brand_voice_score", 5),
                engagement_score=inp.get("engagement_score", 5),
                clarity_score=inp.get("clarity_score", 5),
                overall_score=inp.get("overall_score", 5),
                brand_voice_issues=inp.get("brand_voice_issues", []),
                engagement_issues=inp.get("engagement_issues", []),
                strengths=inp.get("strengths", []),
                suggestions=inp.get("suggestions", []),
                verdict=inp.get("verdict", "refine"),
            )
            feedbacks.append(fb)
            _log(f"  [evaluate] Day {fb.day} ({fb.platform}): "
                 f"voice={fb.brand_voice_score} engage={fb.engagement_score} "
                 f"clarity={fb.clarity_score} overall={fb.overall_score} -> {fb.verdict}")
        return json.dumps({"status": "recorded"})

    total_in, total_out = _run_session_loop(client, session.id, handle_evaluate)

    # Clean up
    try:
        client.beta.sessions.delete(session_id=session.id)
        client.beta.environments.delete(environment_id=environment.id)
        client.beta.agents.archive(agent_id=agent.id)
    except Exception:
        pass

    return feedbacks, total_in, total_out


# ---------------------------------------------------------------------------
# Refine phase
# ---------------------------------------------------------------------------

def _refine_posts(
    client: anthropic.Anthropic,
    posts_to_refine: list[SocialPost],
    feedback_map: dict[int, PostFeedback],
    product: SocialProductInfo,
    brand_voice: BrandVoice,
    model: str,
) -> tuple[list[SocialPost], int, int]:
    """Run the refiner agent on posts that need improvement via managed-agents API."""
    system = _build_refine_system(product, brand_voice)

    agent = client.beta.agents.create(
        model=model,
        name="post-refiner",
        description="Social media post refiner",
        system=system,
        tools=REFINE_TOOLS,
    )
    environment = client.beta.environments.create(name="refine-env")
    session = client.beta.sessions.create(
        agent={"agent_id": agent.id},
        environment_id=environment.id,
    )

    blocks = []
    for post in sorted(posts_to_refine, key=lambda p: p.day):
        fb = feedback_map[post.day]
        blocks.append(
            f"### Post to Refine — Day {post.day}\n\n"
            f"**Current Post:**\n{_post_to_text(post)}\n\n"
            f"**Feedback:**\n"
            f"- Brand Voice Score: {fb.brand_voice_score}/10\n"
            f"- Engagement Score: {fb.engagement_score}/10\n"
            f"- Clarity Score: {fb.clarity_score}/10\n"
            f"- Overall Score: {fb.overall_score}/10\n"
            f"- Brand Voice Issues: {'; '.join(fb.brand_voice_issues) if fb.brand_voice_issues else 'None'}\n"
            f"- Engagement Issues: {'; '.join(fb.engagement_issues) if fb.engagement_issues else 'None'}\n"
            f"- Strengths (preserve these): {'; '.join(fb.strengths) if fb.strengths else 'None'}\n"
            f"- Suggestions: {'; '.join(fb.suggestions) if fb.suggestions else 'None'}\n"
        )

    user_msg = (
        f"Refine the following {len(posts_to_refine)} posts based on the reviewer feedback. "
        f"Use `refine_post` for each one.\n\n" + "\n---\n\n".join(blocks)
    )

    client.beta.sessions.events.send(
        session_id=session.id,
        events=[{
            "type": "user.message",
            "content": [{"type": "text", "text": user_msg}],
        }],
    )

    refined: list[SocialPost] = []

    def handle_refine(name: str, inp: dict) -> str:
        if name == "refine_post":
            post = SocialPost(
                day=inp.get("day", 0),
                day_label=inp.get("day_label", ""),
                platform=inp.get("platform", ""),
                theme=inp.get("theme", ""),
                caption=inp.get("caption", ""),
                image_description=inp.get("image_description", ""),
                hashtags=inp.get("hashtags", []),
                best_time_to_post=inp.get("best_time_to_post", ""),
                engagement_hook=inp.get("engagement_hook", ""),
            )
            refined.append(post)
            _log(f"  [refine] Day {post.day} ({post.platform}): refined caption ({len(post.caption)} chars)")
        return json.dumps({"status": "recorded"})

    total_in, total_out = _run_session_loop(client, session.id, handle_refine)

    # Clean up
    try:
        client.beta.sessions.delete(session_id=session.id)
        client.beta.environments.delete(environment_id=environment.id)
        client.beta.agents.archive(agent_id=agent.id)
    except Exception:
        pass

    return refined, total_in, total_out


# ---------------------------------------------------------------------------
# Main feedback loop
# ---------------------------------------------------------------------------

def run_feedback_loop(
    plan: WeeklyContentPlan,
    model: str = DEFAULT_MODEL,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    pass_threshold: int = PASS_THRESHOLD,
) -> RefinedContentPlan:
    """Run the iterative feedback loop on a content plan.

    Each round:
    1. Evaluate all current posts against brand voice + engagement criteria
    2. Posts scoring >= pass_threshold pass through unchanged
    3. Posts below threshold are refined by a separate agent call
    4. Repeat until all posts pass or max_rounds is reached
    """
    client = anthropic.Anthropic()
    product = plan.product
    brand_voice = plan.brand_voice

    result = RefinedContentPlan(
        original_plan=deepcopy(plan),
        final_plan=deepcopy(plan),
    )

    current_posts = list(plan.posts)
    total_in = 0
    total_out = 0

    _log(f"\n{'='*60}")
    _log(f"Feedback Loop — Evaluating & refining {len(current_posts)} posts")
    _log(f"Model: {model} | Max rounds: {max_rounds} | Pass threshold: {pass_threshold}/10")
    _log(f"API: Managed Agents (beta)")
    _log(f"{'='*60}\n")

    for round_num in range(1, max_rounds + 1):
        _log(f"\n--- Round {round_num} ---\n")

        rnd = RefinementRound(
            round_number=round_num,
            original_posts=deepcopy(current_posts),
        )

        # Phase 1: Evaluate
        _log("[Phase 1: Evaluating posts]")
        feedbacks, eval_in, eval_out = _evaluate_posts(
            client, current_posts, product, brand_voice, model,
        )
        total_in += eval_in
        total_out += eval_out
        rnd.feedback = feedbacks

        # Build feedback lookup by day
        feedback_map = {fb.day: fb for fb in feedbacks}

        # Determine which posts need refinement
        posts_to_refine = []
        passed_posts = []
        for post in current_posts:
            fb = feedback_map.get(post.day)
            if fb and fb.verdict == "refine":
                posts_to_refine.append(post)
            else:
                passed_posts.append(post)

        rnd.posts_passed_count = len(passed_posts)
        rnd.posts_refined_count = len(posts_to_refine)

        _log(f"\n  Passed: {len(passed_posts)} | Need refinement: {len(posts_to_refine)}")

        # Phase 2: Refine posts that didn't pass
        if posts_to_refine:
            _log("\n[Phase 2: Refining posts]")
            refined_posts, ref_in, ref_out = _refine_posts(
                client, posts_to_refine, feedback_map, product, brand_voice, model,
            )
            total_in += ref_in
            total_out += ref_out
            rnd.refined_posts = refined_posts
            result.total_refinements += len(refined_posts)

            # Merge: keep passed posts, replace refined ones
            refined_map = {p.day: p for p in refined_posts}
            current_posts = []
            for post in rnd.original_posts:
                if post.day in refined_map:
                    current_posts.append(refined_map[post.day])
                else:
                    current_posts.append(post)
        else:
            rnd.refined_posts = []
            _log("\n  All posts passed! No refinement needed.")

        result.rounds.append(rnd)

        # If all posts passed, we're done
        if not posts_to_refine:
            _log(f"\n[All posts passed in round {round_num}]")
            break

    # Build final plan
    result.final_plan.posts = sorted(current_posts, key=lambda p: p.day)
    result.input_tokens = total_in
    result.output_tokens = total_out

    _log(f"\n{'='*60}")
    _log(f"Feedback loop complete: {len(result.rounds)} rounds, {result.total_refinements} refinements")
    _log(f"Tokens: {total_in:,} input, {total_out:,} output")
    _log(f"{'='*60}\n")

    return result
