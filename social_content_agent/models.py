"""Data models for social media content generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
from datetime import datetime


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"


class ContentTheme(str, Enum):
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    BEHIND_THE_SCENES = "behind_the_scenes"
    USER_GENERATED = "user_generated"
    STORYTELLING = "storytelling"
    ENGAGEMENT = "engagement"
    INSPIRATIONAL = "inspirational"


@dataclass
class BrandVoice:
    """Specification for brand tone and personality."""
    tone: str  # e.g., "professional yet approachable"
    personality: str  # e.g., "witty, knowledgeable, empathetic"
    do_use: list[str] = field(default_factory=list)  # phrases/styles to use
    avoid: list[str] = field(default_factory=list)  # phrases/styles to avoid
    hashtag_style: str = "moderate"  # minimal, moderate, heavy
    emoji_usage: str = "moderate"  # none, minimal, moderate, heavy

    def to_prompt(self) -> str:
        lines = [
            f"Tone: {self.tone}",
            f"Personality: {self.personality}",
            f"Hashtag style: {self.hashtag_style}",
            f"Emoji usage: {self.emoji_usage}",
        ]
        if self.do_use:
            lines.append(f"Do use: {', '.join(self.do_use)}")
        if self.avoid:
            lines.append(f"Avoid: {', '.join(self.avoid)}")
        return "\n".join(lines)


@dataclass
class ProductInfo:
    """Description of the product or service to promote."""
    name: str
    description: str
    target_audience: str
    key_features: list[str] = field(default_factory=list)
    differentiators: list[str] = field(default_factory=list)
    website_url: str = ""
    category: str = ""

    def to_prompt(self) -> str:
        lines = [
            f"Product/Service: {self.name}",
            f"Description: {self.description}",
            f"Target Audience: {self.target_audience}",
            f"Category: {self.category}" if self.category else "",
        ]
        if self.key_features:
            lines.append(f"Key Features: {', '.join(self.key_features)}")
        if self.differentiators:
            lines.append(f"Differentiators: {', '.join(self.differentiators)}")
        if self.website_url:
            lines.append(f"Website: {self.website_url}")
        return "\n".join(line for line in lines if line)


@dataclass
class SocialPost:
    """A single social media post."""
    day: int  # 1-7
    day_label: str  # e.g., "Monday"
    platform: str
    theme: str
    caption: str
    image_description: str
    hashtags: list[str] = field(default_factory=list)
    best_time_to_post: str = ""
    engagement_hook: str = ""


@dataclass
class WeeklyContentPlan:
    """A full week of social media content."""
    product: ProductInfo
    brand_voice: BrandVoice
    posts: list[SocialPost] = field(default_factory=list)
    strategy_notes: str = ""
    content_pillars: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON for the HTML viewer."""
        return json.dumps({
            "product": {
                "name": self.product.name,
                "description": self.product.description,
                "target_audience": self.product.target_audience,
                "category": self.product.category,
            },
            "brand_voice": {
                "tone": self.brand_voice.tone,
                "personality": self.brand_voice.personality,
            },
            "strategy_notes": self.strategy_notes,
            "content_pillars": self.content_pillars,
            "posts": [
                {
                    "day": p.day,
                    "day_label": p.day_label,
                    "platform": p.platform,
                    "theme": p.theme,
                    "caption": p.caption,
                    "image_description": p.image_description,
                    "hashtags": p.hashtags,
                    "best_time_to_post": p.best_time_to_post,
                    "engagement_hook": p.engagement_hook,
                }
                for p in self.posts
            ],
        }, indent=2)

    def to_markdown(self) -> str:
        """Render as a polished markdown document."""
        lines = [
            f"# 7-Day Social Media Content Plan",
            f"## {self.product.name}",
            "",
            f"**Target Audience:** {self.product.target_audience}",
            f"**Brand Voice:** {self.brand_voice.tone} | {self.brand_voice.personality}",
            "",
        ]
        if self.strategy_notes:
            lines += ["### Strategy Overview", "", self.strategy_notes, ""]
        if self.content_pillars:
            lines += ["### Content Pillars", ""]
            for pillar in self.content_pillars:
                lines.append(f"- {pillar}")
            lines.append("")

        lines.append("---")
        lines.append("")

        for post in sorted(self.posts, key=lambda p: p.day):
            lines += [
                f"## Day {post.day} — {post.day_label}",
                f"**Platform:** {post.platform} | **Theme:** {post.theme}",
                "",
                f"### Caption",
                post.caption,
                "",
                f"### Suggested Image",
                post.image_description,
                "",
            ]
            if post.hashtags:
                lines.append(f"**Hashtags:** {' '.join(post.hashtags)}")
            if post.best_time_to_post:
                lines.append(f"**Best Time to Post:** {post.best_time_to_post}")
            if post.engagement_hook:
                lines.append(f"**Engagement Hook:** {post.engagement_hook}")
            lines += ["", "---", ""]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feedback Loop Models
# ---------------------------------------------------------------------------


@dataclass
class PostFeedback:
    """Evaluation feedback for a single social media post."""
    day: int
    platform: str
    brand_voice_score: int  # 1-10
    engagement_score: int  # 1-10
    clarity_score: int  # 1-10
    overall_score: int  # 1-10
    brand_voice_issues: list[str] = field(default_factory=list)
    engagement_issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    verdict: str = ""  # "pass" or "refine"


@dataclass
class RefinementRound:
    """One round of evaluation + refinement for the full plan."""
    round_number: int
    feedback: list[PostFeedback] = field(default_factory=list)
    original_posts: list[SocialPost] = field(default_factory=list)
    refined_posts: list[SocialPost] = field(default_factory=list)
    posts_refined_count: int = 0
    posts_passed_count: int = 0


@dataclass
class RefinedContentPlan:
    """A content plan with full feedback loop history."""
    original_plan: WeeklyContentPlan
    final_plan: WeeklyContentPlan
    rounds: list[RefinementRound] = field(default_factory=list)
    total_refinements: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_json(self) -> str:
        """Serialize the full refinement history to JSON."""
        return json.dumps({
            "product": {
                "name": self.original_plan.product.name,
                "description": self.original_plan.product.description,
                "target_audience": self.original_plan.product.target_audience,
                "category": self.original_plan.product.category,
            },
            "brand_voice": {
                "tone": self.original_plan.brand_voice.tone,
                "personality": self.original_plan.brand_voice.personality,
                "hashtag_style": self.original_plan.brand_voice.hashtag_style,
                "emoji_usage": self.original_plan.brand_voice.emoji_usage,
                "do_use": self.original_plan.brand_voice.do_use,
                "avoid": self.original_plan.brand_voice.avoid,
            },
            "strategy_notes": self.final_plan.strategy_notes,
            "content_pillars": self.final_plan.content_pillars,
            "total_refinements": self.total_refinements,
            "total_rounds": len(self.rounds),
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "rounds": [
                {
                    "round_number": r.round_number,
                    "posts_refined": r.posts_refined_count,
                    "posts_passed": r.posts_passed_count,
                    "feedback": [
                        {
                            "day": fb.day,
                            "platform": fb.platform,
                            "scores": {
                                "brand_voice": fb.brand_voice_score,
                                "engagement": fb.engagement_score,
                                "clarity": fb.clarity_score,
                                "overall": fb.overall_score,
                            },
                            "brand_voice_issues": fb.brand_voice_issues,
                            "engagement_issues": fb.engagement_issues,
                            "strengths": fb.strengths,
                            "suggestions": fb.suggestions,
                            "verdict": fb.verdict,
                        }
                        for fb in r.feedback
                    ],
                    "original_posts": [
                        {
                            "day": p.day,
                            "day_label": p.day_label,
                            "platform": p.platform,
                            "theme": p.theme,
                            "caption": p.caption,
                            "image_description": p.image_description,
                            "hashtags": p.hashtags,
                            "best_time_to_post": p.best_time_to_post,
                            "engagement_hook": p.engagement_hook,
                        }
                        for p in r.original_posts
                    ],
                    "refined_posts": [
                        {
                            "day": p.day,
                            "day_label": p.day_label,
                            "platform": p.platform,
                            "theme": p.theme,
                            "caption": p.caption,
                            "image_description": p.image_description,
                            "hashtags": p.hashtags,
                            "best_time_to_post": p.best_time_to_post,
                            "engagement_hook": p.engagement_hook,
                        }
                        for p in r.refined_posts
                    ],
                }
                for r in self.rounds
            ],
            "final_posts": [
                {
                    "day": p.day,
                    "day_label": p.day_label,
                    "platform": p.platform,
                    "theme": p.theme,
                    "caption": p.caption,
                    "image_description": p.image_description,
                    "hashtags": p.hashtags,
                    "best_time_to_post": p.best_time_to_post,
                    "engagement_hook": p.engagement_hook,
                }
                for p in sorted(self.final_plan.posts, key=lambda p: p.day)
            ],
        }, indent=2)

    def to_markdown(self) -> str:
        """Render the refinement process as markdown."""
        lines = [
            "# Social Content Feedback Loop Report",
            f"## {self.original_plan.product.name}",
            "",
            f"**Target Audience:** {self.original_plan.product.target_audience}",
            f"**Brand Voice:** {self.original_plan.brand_voice.tone}",
            f"**Refinement Rounds:** {len(self.rounds)}",
            f"**Total Posts Refined:** {self.total_refinements}",
            "",
            "---",
            "",
        ]

        for rnd in self.rounds:
            lines += [
                f"## Round {rnd.round_number}",
                f"Posts refined: {rnd.posts_refined_count} | Posts passed: {rnd.posts_passed_count}",
                "",
            ]
            for fb in sorted(rnd.feedback, key=lambda f: f.day):
                lines += [
                    f"### Day {fb.day} — {fb.platform}",
                    f"**Scores:** Brand Voice {fb.brand_voice_score}/10 | "
                    f"Engagement {fb.engagement_score}/10 | "
                    f"Clarity {fb.clarity_score}/10 | "
                    f"Overall {fb.overall_score}/10",
                    f"**Verdict:** {'Passed' if fb.verdict == 'pass' else 'Needs Refinement'}",
                    "",
                ]
                if fb.strengths:
                    lines.append("**Strengths:**")
                    for s in fb.strengths:
                        lines.append(f"- {s}")
                    lines.append("")
                if fb.brand_voice_issues or fb.engagement_issues:
                    lines.append("**Issues:**")
                    for issue in fb.brand_voice_issues + fb.engagement_issues:
                        lines.append(f"- {issue}")
                    lines.append("")
                if fb.suggestions:
                    lines.append("**Suggestions:**")
                    for s in fb.suggestions:
                        lines.append(f"- {s}")
                    lines.append("")
            lines += ["---", ""]

        lines += [
            "## Final Polished Posts",
            "",
        ]
        for post in sorted(self.final_plan.posts, key=lambda p: p.day):
            lines += [
                f"### Day {post.day} — {post.day_label}",
                f"**Platform:** {post.platform} | **Theme:** {post.theme}",
                "",
                "#### Caption",
                post.caption,
                "",
                "#### Suggested Image",
                post.image_description,
                "",
            ]
            if post.hashtags:
                lines.append(f"**Hashtags:** {' '.join(post.hashtags)}")
            if post.best_time_to_post:
                lines.append(f"**Best Time to Post:** {post.best_time_to_post}")
            if post.engagement_hook:
                lines.append(f"**Engagement Hook:** {post.engagement_hook}")
            lines += ["", "---", ""]

        return "\n".join(lines)
