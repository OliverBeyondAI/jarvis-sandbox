#!/usr/bin/env python3
"""End-to-end demo: Full SMB content pipeline for a local bakery.

Demonstrates:
  1. Brand voice adaptation — defining a unique bakery persona
  2. Week-long content generation — 7-day social media plan via managed-agents API
  3. Feedback-driven refinement — iterative evaluation and polishing
  4. Structured output — JSON + Markdown + HTML report saved to demo_output/

Usage:
  python demo_bakery_pipeline.py [--model MODEL] [--max-rounds N] [--skip-refine]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

# ── Local imports ────────────────────────────────────────────────────────────
from xena_marketing_agent.models import (
    BrandVoice,
    SocialProductInfo,
    WeeklyContentPlan,
    RefinedContentPlan,
)
from xena_marketing_agent.social_agent import run_social_agent, DEFAULT_MODEL
from xena_marketing_agent.feedback_loop import run_feedback_loop


# ═══════════════════════════════════════════════════════════════════════════
# 1. SAMPLE SMB DEFINITION — Sweet Crumb Bakery
# ═══════════════════════════════════════════════════════════════════════════

BAKERY_PRODUCT = SocialProductInfo(
    name="Sweet Crumb Bakery",
    description=(
        "A neighborhood artisan bakery specializing in handcrafted sourdough breads, "
        "French pastries, and custom celebration cakes. Everything is baked fresh daily "
        "using locally sourced, organic ingredients. Known for their signature rosemary "
        "olive oil sourdough and seasonal fruit danishes. Family-owned for 12 years, "
        "they host weekend baking workshops and partner with local coffee roasters."
    ),
    target_audience=(
        "Local food lovers aged 25-55 within a 10-mile radius. Mix of young "
        "professionals grabbing morning pastries, parents ordering birthday cakes, "
        "and weekend brunch enthusiasts. They value quality ingredients, community "
        "connection, and supporting small businesses. Active on Instagram, Facebook, "
        "and occasionally TikTok."
    ),
    key_features=[
        "Handcrafted sourdough with 12-year-old starter",
        "French-trained pastry chef on staff",
        "Custom celebration cakes (48-hour notice)",
        "Weekend baking workshops for all skill levels",
        "Locally sourced organic flour, butter, and seasonal fruit",
        "Daily rotating specials board",
    ],
    differentiators=[
        "Only bakery in town using a 12-year sourdough starter",
        "Baking workshops create community, not just customers",
        "Partnerships with 3 local farms for seasonal ingredients",
        "Every loaf and pastry made from scratch — no mixes, no shortcuts",
    ],
    category="Local Food & Beverage / Bakery",
    website_url="https://sweetcrumbbakery.example.com",
)

BAKERY_BRAND_VOICE = BrandVoice(
    tone=(
        "Warm, neighborly, and genuine — like chatting with the baker who "
        "remembers your usual order and saves you the last almond croissant"
    ),
    personality=(
        "Passionate about craft, community-oriented, a little flour-dusted "
        "and playful, knowledgeable without being pretentious"
    ),
    do_use=[
        "sensory language (golden crust, buttery layers, warm from the oven)",
        "behind-the-scenes peeks at the baking process",
        "community and neighborhood references",
        "invitations to visit, taste, and join workshops",
        "storytelling about ingredients and traditions",
    ],
    avoid=[
        "corporate or chain-bakery language",
        "aggressive sales tactics or urgency pressure",
        "overly polished or stock-photo aesthetic",
        "health-shaming or diet culture references",
        "generic food cliches ('yummy in my tummy')",
    ],
    hashtag_style="moderate",
    emoji_usage="moderate",
)


# ═══════════════════════════════════════════════════════════════════════════
# 2. BRAND VOICE ADAPTATION DEMO
# ═══════════════════════════════════════════════════════════════════════════

VOICE_VARIANTS = {
    "neighborhood": BAKERY_BRAND_VOICE,
    "upscale": BrandVoice(
        tone="Refined, artisanal, and quietly confident — a patisserie, not just a bakery",
        personality="Sophisticated, detail-obsessed, subtly luxurious",
        do_use=[
            "French pastry terminology (viennoiserie, laminage, crumb structure)",
            "ingredient provenance and sourcing stories",
            "minimalist, elegant visual descriptions",
            "understated confidence in craftsmanship",
        ],
        avoid=[
            "casual slang or overly friendly tone",
            "discount or deal language",
            "excessive emojis",
            "corporate chain references",
        ],
        hashtag_style="minimal",
        emoji_usage="minimal",
    ),
    "playful": BrandVoice(
        tone="Fun, energetic, and deliciously irreverent — life's too short for boring bread",
        personality="Witty, enthusiastic, a little cheeky, always hungry",
        do_use=[
            "food puns and baking humor",
            "enthusiastic exclamations",
            "interactive questions and polls",
            "behind-the-scenes chaos and fun moments",
        ],
        avoid=[
            "corporate speak",
            "dry or formal language",
            "diet culture messaging",
            "anything that feels like a lecture",
        ],
        hashtag_style="heavy",
        emoji_usage="heavy",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# 3. OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    """Print progress to stderr."""
    print(msg, file=sys.stderr)


def save_json(data: dict, path: str) -> None:
    """Write a dict as formatted JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    _log(f"  Saved: {path}")


def save_markdown(text: str, path: str) -> None:
    """Write markdown content to a file."""
    with open(path, "w") as f:
        f.write(text)
    _log(f"  Saved: {path}")


def build_pipeline_summary(
    plan: WeeklyContentPlan,
    refined: RefinedContentPlan | None,
    duration_sec: float,
    voice_name: str,
) -> dict:
    """Build a comprehensive JSON summary of the entire pipeline run."""
    summary = {
        "pipeline": "Sweet Crumb Bakery — End-to-End Content Pipeline",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration_sec, 1),
        "voice_profile": voice_name,
        "product": {
            "name": plan.product.name,
            "description": plan.product.description,
            "target_audience": plan.product.target_audience,
            "category": plan.product.category,
            "key_features": plan.product.key_features,
            "differentiators": plan.product.differentiators,
        },
        "brand_voice": {
            "tone": plan.brand_voice.tone,
            "personality": plan.brand_voice.personality,
            "hashtag_style": plan.brand_voice.hashtag_style,
            "emoji_usage": plan.brand_voice.emoji_usage,
            "do_use": plan.brand_voice.do_use,
            "avoid": plan.brand_voice.avoid,
        },
        "content_plan": {
            "strategy_notes": plan.strategy_notes,
            "content_pillars": plan.content_pillars,
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
                for p in sorted(plan.posts, key=lambda p: p.day)
            ],
        },
    }

    if refined:
        summary["refinement"] = {
            "total_rounds": len(refined.rounds),
            "total_refinements": refined.total_refinements,
            "token_usage": {
                "input_tokens": refined.input_tokens,
                "output_tokens": refined.output_tokens,
            },
            "rounds": [
                {
                    "round_number": r.round_number,
                    "posts_passed": r.posts_passed_count,
                    "posts_refined": r.posts_refined_count,
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
                            "verdict": fb.verdict,
                            "strengths": fb.strengths,
                            "suggestions": fb.suggestions,
                        }
                        for fb in r.feedback
                    ],
                }
                for r in refined.rounds
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
                for p in sorted(refined.final_plan.posts, key=lambda p: p.day)
            ],
        }

    return summary


def build_full_markdown(
    plan: WeeklyContentPlan,
    refined: RefinedContentPlan | None,
    duration_sec: float,
    voice_name: str,
) -> str:
    """Build a comprehensive markdown report of the pipeline run."""
    lines = [
        "# Sweet Crumb Bakery — Content Pipeline Report",
        "",
        f"*Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*",
        f"*Pipeline duration: {duration_sec:.1f}s | Voice profile: {voice_name}*",
        "",
        "---",
        "",
        "## Business Profile",
        "",
        f"**{plan.product.name}**",
        "",
        f"> {plan.product.description}",
        "",
        f"**Target Audience:** {plan.product.target_audience}",
        "",
        "**Key Features:**",
    ]
    for feat in plan.product.key_features:
        lines.append(f"- {feat}")
    lines += [
        "",
        "**Differentiators:**",
    ]
    for diff in plan.product.differentiators:
        lines.append(f"- {diff}")
    lines += [
        "",
        "---",
        "",
        "## Brand Voice Configuration",
        "",
        f"**Tone:** {plan.brand_voice.tone}",
        "",
        f"**Personality:** {plan.brand_voice.personality}",
        "",
        f"**Emoji Usage:** {plan.brand_voice.emoji_usage} | **Hashtag Style:** {plan.brand_voice.hashtag_style}",
        "",
    ]
    if plan.brand_voice.do_use:
        lines.append("**Do Use:**")
        for item in plan.brand_voice.do_use:
            lines.append(f"- {item}")
        lines.append("")
    if plan.brand_voice.avoid:
        lines.append("**Avoid:**")
        for item in plan.brand_voice.avoid:
            lines.append(f"- {item}")
        lines.append("")
    lines += ["---", ""]

    if plan.strategy_notes:
        lines += ["## Content Strategy", "", plan.strategy_notes, ""]
    if plan.content_pillars:
        lines += ["### Content Pillars", ""]
        for pillar in plan.content_pillars:
            lines.append(f"- {pillar}")
        lines += ["", "---", ""]

    lines += ["## 7-Day Content Plan (Initial Generation)", ""]
    for post in sorted(plan.posts, key=lambda p: p.day):
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

    if refined:
        lines += [
            "## Feedback-Driven Refinement",
            "",
            f"**Rounds completed:** {len(refined.rounds)}",
            f"**Total posts refined:** {refined.total_refinements}",
            f"**Refinement tokens:** {refined.input_tokens:,} input / {refined.output_tokens:,} output",
            "",
        ]

        for rnd in refined.rounds:
            lines += [
                f"### Round {rnd.round_number}",
                f"Passed: {rnd.posts_passed_count} | Refined: {rnd.posts_refined_count}",
                "",
            ]
            for fb in sorted(rnd.feedback, key=lambda f: f.day):
                emoji = "pass" if fb.verdict == "pass" else "refine"
                lines += [
                    f"#### Day {fb.day} — {fb.platform} [{emoji}]",
                    f"Scores: Voice {fb.brand_voice_score}/10 | "
                    f"Engage {fb.engagement_score}/10 | "
                    f"Clarity {fb.clarity_score}/10 | "
                    f"Overall {fb.overall_score}/10",
                    "",
                ]
                if fb.strengths:
                    lines.append("**Strengths:** " + "; ".join(fb.strengths))
                if fb.suggestions:
                    lines.append("**Suggestions:** " + "; ".join(fb.suggestions))
                lines.append("")
            lines += ["---", ""]

        lines += ["## Final Polished Posts (After Refinement)", ""]
        for post in sorted(refined.final_plan.posts, key=lambda p: p.day):
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


def build_html_report(summary: dict) -> str:
    """Generate a self-contained HTML report from the pipeline summary."""
    json_data = json.dumps(summary, indent=2, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sweet Crumb Bakery — Content Pipeline Report</title>
<style>
:root {{
  --bg: #faf8f5;
  --card: #ffffff;
  --text: #2d2a26;
  --text-secondary: #6b6560;
  --accent: #c4703f;
  --accent-light: #f5e6d8;
  --accent-dark: #a05a2f;
  --border: #e8e2db;
  --green: #3a8a5c;
  --green-light: #e6f4ec;
  --amber: #b8860b;
  --amber-light: #fef9e7;
  --blue: #4a7fb5;
  --blue-light: #e8f0fa;
  --red: #c0392b;
  --red-light: #fde8e6;
  --radius: 12px;
  --shadow: 0 2px 12px rgba(0,0,0,0.06);
  --shadow-hover: 0 4px 20px rgba(0,0,0,0.1);
  --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}}
.hero {{
  background: linear-gradient(135deg, #c4703f 0%, #a05a2f 50%, #7a4425 100%);
  color: white;
  padding: 3rem 2rem 2.5rem;
  text-align: center;
  position: relative;
  overflow: hidden;
}}
.hero::before {{
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
  animation: shimmer 20s ease-in-out infinite;
}}
@keyframes shimmer {{
  0%, 100% {{ transform: translate(0, 0); }}
  50% {{ transform: translate(5%, 5%); }}
}}
.hero h1 {{ font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; position: relative; letter-spacing: -0.02em; }}
.hero .subtitle {{ font-size: 1.05rem; opacity: 0.9; position: relative; }}
.hero .meta {{ display: flex; gap: 1.5rem; justify-content: center; margin-top: 1.2rem; font-size: 0.85rem; opacity: 0.8; position: relative; flex-wrap: wrap; }}
.hero .meta span {{ display: flex; align-items: center; gap: 0.3rem; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }}
.section-title {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--accent); color: var(--accent-dark); }}
.card {{ background: var(--card); border-radius: var(--radius); border: 1px solid var(--border); padding: 1.5rem; margin-bottom: 1.2rem; box-shadow: var(--shadow); transition: box-shadow 0.2s ease; }}
.card:hover {{ box-shadow: var(--shadow-hover); }}
.card h3 {{ font-size: 1.1rem; margin-bottom: 0.6rem; color: var(--accent-dark); }}
.badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }}
.badge-platform {{ background: var(--blue-light); color: var(--blue); }}
.badge-pass {{ background: var(--green-light); color: var(--green); }}
.badge-refine {{ background: var(--amber-light); color: var(--amber); }}
.badge-theme {{ background: var(--accent-light); color: var(--accent-dark); }}
.pillars {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.8rem 0; }}
.pillar {{ background: var(--accent-light); color: var(--accent-dark); padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }}
.voice-box {{ background: linear-gradient(135deg, var(--accent-light), #fff); border-left: 4px solid var(--accent); padding: 1rem 1.2rem; border-radius: 0 var(--radius) var(--radius) 0; margin-bottom: 1rem; font-style: italic; color: var(--text-secondary); }}
.post-grid {{ display: grid; gap: 1.2rem; }}
.post-card {{ background: var(--card); border-radius: var(--radius); border: 1px solid var(--border); overflow: hidden; box-shadow: var(--shadow); transition: box-shadow 0.2s, transform 0.2s; }}
.post-card:hover {{ box-shadow: var(--shadow-hover); transform: translateY(-2px); }}
.post-header {{ padding: 1rem 1.2rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); background: linear-gradient(to right, var(--accent-light), transparent); }}
.post-header .day-info {{ font-weight: 700; font-size: 1rem; }}
.post-header .badges {{ display: flex; gap: 0.4rem; flex-wrap: wrap; }}
.post-body {{ padding: 1.2rem; }}
.post-body .caption {{ white-space: pre-wrap; line-height: 1.7; margin-bottom: 1rem; font-size: 0.95rem; }}
.post-body .image-desc {{ background: #f8f5f1; padding: 0.8rem 1rem; border-radius: 8px; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.8rem; border-left: 3px solid var(--accent); }}
.post-body .image-desc::before {{ content: 'Image concept: '; font-weight: 600; color: var(--accent-dark); }}
.hashtags {{ display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.5rem; }}
.hashtag {{ background: var(--blue-light); color: var(--blue); padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.78rem; }}
.post-footer {{ padding: 0.6rem 1.2rem; border-top: 1px solid var(--border); display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-secondary); background: #fafaf8; }}
.scores-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.6rem; margin: 0.8rem 0; }}
.score-box {{ text-align: center; padding: 0.6rem; border-radius: 8px; background: #f8f5f1; }}
.score-box .score-value {{ font-size: 1.4rem; font-weight: 700; }}
.score-box .score-label {{ font-size: 0.72rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }}
.score-high {{ color: var(--green); }}
.score-mid {{ color: var(--amber); }}
.score-low {{ color: var(--red); }}
.refinement-round {{ margin-bottom: 2rem; }}
.round-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding: 0.8rem 1rem; background: var(--card); border-radius: var(--radius); border: 1px solid var(--border); }}
.round-header h3 {{ margin: 0; font-size: 1.1rem; }}
.round-stats {{ display: flex; gap: 1rem; font-size: 0.85rem; }}
.tabs {{ display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 2px solid var(--border); }}
.tab {{ padding: 0.7rem 1.5rem; cursor: pointer; font-weight: 600; font-size: 0.9rem; color: var(--text-secondary); border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; background: none; border-top: none; border-left: none; border-right: none; font-family: var(--font); }}
.tab:hover {{ color: var(--accent); }}
.tab.active {{ color: var(--accent-dark); border-bottom-color: var(--accent); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.features-list {{ list-style: none; padding: 0; }}
.features-list li {{ padding: 0.4rem 0; padding-left: 1.2rem; position: relative; font-size: 0.9rem; }}
.features-list li::before {{ content: ''; position: absolute; left: 0; top: 0.75rem; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }}
.comparison-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }}
@media (max-width: 768px) {{
  .hero h1 {{ font-size: 1.6rem; }}
  .hero .meta {{ flex-direction: column; gap: 0.5rem; }}
  .container {{ padding: 1rem; }}
  .post-header {{ flex-direction: column; align-items: flex-start; gap: 0.5rem; }}
  .tabs {{ overflow-x: auto; }}
  .tab {{ white-space: nowrap; padding: 0.5rem 1rem; font-size: 0.82rem; }}
  .scores-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>

<div class="hero">
  <h1>Sweet Crumb Bakery</h1>
  <p class="subtitle">End-to-End Content Pipeline Report</p>
  <div class="meta">
    <span id="meta-date"></span>
    <span id="meta-duration"></span>
    <span id="meta-voice"></span>
    <span id="meta-posts"></span>
  </div>
</div>

<div class="container">
  <div class="tabs">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="content">Content Plan</button>
    <button class="tab" data-tab="refinement">Refinement</button>
    <button class="tab" data-tab="final">Final Posts</button>
  </div>

  <div id="tab-overview" class="tab-content active">
    <h2 class="section-title">Business Profile</h2>
    <div class="card">
      <p id="biz-description"></p>
      <br>
      <p><strong>Target Audience:</strong> <span id="biz-audience"></span></p>
    </div>
    <div class="comparison-grid">
      <div class="card">
        <h3>Key Features</h3>
        <ul class="features-list" id="features-list"></ul>
      </div>
      <div class="card">
        <h3>Differentiators</h3>
        <ul class="features-list" id="diff-list"></ul>
      </div>
    </div>
    <h2 class="section-title" style="margin-top: 1.5rem;">Brand Voice</h2>
    <div class="voice-box" id="voice-tone"></div>
    <div class="card">
      <p><strong>Personality:</strong> <span id="voice-personality"></span></p>
      <p style="margin-top:0.5rem;"><strong>Emoji:</strong> <span id="voice-emoji"></span> | <strong>Hashtags:</strong> <span id="voice-hashtags"></span></p>
      <div class="comparison-grid" style="margin-top:1rem;">
        <div>
          <h3 style="color:var(--green);">Do Use</h3>
          <ul class="features-list" id="do-use-list"></ul>
        </div>
        <div>
          <h3 style="color:var(--red);">Avoid</h3>
          <ul class="features-list" id="avoid-list"></ul>
        </div>
      </div>
    </div>
    <h2 class="section-title" style="margin-top: 1.5rem;">Content Strategy</h2>
    <div class="card">
      <p id="strategy-notes"></p>
      <div class="pillars" id="pillars-container"></div>
    </div>
  </div>

  <div id="tab-content" class="tab-content">
    <h2 class="section-title">7-Day Content Plan (Initial Generation)</h2>
    <div class="post-grid" id="initial-posts"></div>
  </div>

  <div id="tab-refinement" class="tab-content">
    <h2 class="section-title">Feedback-Driven Refinement</h2>
    <div id="refinement-container"></div>
  </div>

  <div id="tab-final" class="tab-content">
    <h2 class="section-title">Final Polished Posts</h2>
    <div class="post-grid" id="final-posts"></div>
  </div>
</div>

<script>
const DATA = {json_data};

function scoreClass(v) {{ return v >= 8 ? 'score-high' : v >= 6 ? 'score-mid' : 'score-low'; }}

function renderPost(post) {{
  const hashtags = (post.hashtags || []).map(h => `<span class="hashtag">${{h}}</span>`).join('');
  return `
    <div class="post-card">
      <div class="post-header">
        <span class="day-info">Day ${{post.day}} &mdash; ${{post.day_label}}</span>
        <div class="badges">
          <span class="badge badge-platform">${{post.platform}}</span>
          <span class="badge badge-theme">${{post.theme}}</span>
        </div>
      </div>
      <div class="post-body">
        <div class="caption">${{post.caption}}</div>
        <div class="image-desc">${{post.image_description}}</div>
        <div class="hashtags">${{hashtags}}</div>
      </div>
      <div class="post-footer">
        <span>${{post.best_time_to_post || ''}}</span>
        <span>${{post.engagement_hook || ''}}</span>
      </div>
    </div>`;
}}

function renderFeedback(fb) {{
  return `
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
        <h3 style="margin:0;">Day ${{fb.day}} &mdash; ${{fb.platform}}</h3>
        <span class="badge ${{fb.verdict === 'pass' ? 'badge-pass' : 'badge-refine'}}">${{fb.verdict}}</span>
      </div>
      <div class="scores-grid">
        <div class="score-box"><div class="score-value ${{scoreClass(fb.scores.brand_voice)}}">${{fb.scores.brand_voice}}</div><div class="score-label">Voice</div></div>
        <div class="score-box"><div class="score-value ${{scoreClass(fb.scores.engagement)}}">${{fb.scores.engagement}}</div><div class="score-label">Engage</div></div>
        <div class="score-box"><div class="score-value ${{scoreClass(fb.scores.clarity)}}">${{fb.scores.clarity}}</div><div class="score-label">Clarity</div></div>
        <div class="score-box"><div class="score-value ${{scoreClass(fb.scores.overall)}}">${{fb.scores.overall}}</div><div class="score-label">Overall</div></div>
      </div>
      ${{fb.strengths && fb.strengths.length ? '<p style="font-size:0.85rem;margin-top:0.5rem;"><strong style="color:var(--green);">Strengths:</strong> ' + fb.strengths.join('; ') + '</p>' : ''}}
      ${{fb.suggestions && fb.suggestions.length ? '<p style="font-size:0.85rem;margin-top:0.3rem;"><strong style="color:var(--amber);">Suggestions:</strong> ' + fb.suggestions.join('; ') + '</p>' : ''}}
    </div>`;
}}

document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  }});
}});

const d = DATA;
const ts = new Date(d.timestamp);
document.getElementById('meta-date').textContent = ts.toLocaleDateString('en-US', {{ month: 'long', day: 'numeric', year: 'numeric' }});
document.getElementById('meta-duration').textContent = d.duration_seconds + 's';
document.getElementById('meta-voice').textContent = 'Voice: ' + d.voice_profile;
document.getElementById('meta-posts').textContent = d.content_plan.posts.length + ' posts';

document.getElementById('biz-description').textContent = d.product.description;
document.getElementById('biz-audience').textContent = d.product.target_audience;

const fl = document.getElementById('features-list');
(d.product.key_features || []).forEach(f => {{ const li = document.createElement('li'); li.textContent = f; fl.appendChild(li); }});
const dl = document.getElementById('diff-list');
(d.product.differentiators || []).forEach(f => {{ const li = document.createElement('li'); li.textContent = f; dl.appendChild(li); }});

document.getElementById('voice-tone').textContent = d.brand_voice.tone;
document.getElementById('voice-personality').textContent = d.brand_voice.personality;
document.getElementById('voice-emoji').textContent = d.brand_voice.emoji_usage;
document.getElementById('voice-hashtags').textContent = d.brand_voice.hashtag_style;

const dul = document.getElementById('do-use-list');
(d.brand_voice.do_use || []).forEach(f => {{ const li = document.createElement('li'); li.textContent = f; dul.appendChild(li); }});
const avl = document.getElementById('avoid-list');
(d.brand_voice.avoid || []).forEach(f => {{ const li = document.createElement('li'); li.textContent = f; avl.appendChild(li); }});

document.getElementById('strategy-notes').textContent = d.content_plan.strategy_notes || 'Strategy generated by the agent.';
const pc = document.getElementById('pillars-container');
(d.content_plan.content_pillars || []).forEach(p => {{
  const span = document.createElement('span');
  span.className = 'pillar';
  span.textContent = p;
  pc.appendChild(span);
}});

document.getElementById('initial-posts').innerHTML = (d.content_plan.posts || []).map(renderPost).join('');

const rc = document.getElementById('refinement-container');
if (d.refinement) {{
  d.refinement.rounds.forEach(round => {{
    rc.innerHTML += `
      <div class="refinement-round">
        <div class="round-header">
          <h3>Round ${{round.round_number}}</h3>
          <div class="round-stats">
            <span class="badge badge-pass">${{round.posts_passed}} passed</span>
            <span class="badge badge-refine">${{round.posts_refined}} refined</span>
          </div>
        </div>
        ${{(round.feedback || []).map(renderFeedback).join('')}}
      </div>`;
  }});
}} else {{
  rc.innerHTML = '<div class="card"><p>Refinement was skipped for this run.</p></div>';
}}

const fp = document.getElementById('final-posts');
if (d.refinement && d.refinement.final_posts) {{
  fp.innerHTML = d.refinement.final_posts.map(renderPost).join('');
}} else {{
  fp.innerHTML = (d.content_plan.posts || []).map(renderPost).join('');
  const tab = document.querySelector('[data-tab="final"]');
  tab.textContent = 'Final Posts (no refinement)';
}}
</script>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    model: str = DEFAULT_MODEL,
    max_rounds: int = 3,
    skip_refine: bool = False,
    voice_name: str = "neighborhood",
    output_dir: str = "demo_output",
) -> None:
    """Run the complete end-to-end pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    brand_voice = VOICE_VARIANTS[voice_name]
    start = time.time()

    # ── Phase 1: Content Generation ───────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("  PHASE 1: 7-Day Content Generation")
    _log(f"  SMB: {BAKERY_PRODUCT.name}")
    _log(f"  Voice: {voice_name}")
    _log(f"  Model: {model}")
    _log(f"  API: Managed Agents (beta)")
    _log("=" * 70)

    plan = run_social_agent(
        product=BAKERY_PRODUCT,
        brand_voice=brand_voice,
        model=model,
    )

    gen_duration = time.time() - start
    _log(f"\n  Generation complete: {len(plan.posts)} posts in {gen_duration:.1f}s")

    # ── Phase 2: Feedback-Driven Refinement ───────────────────────────────
    refined = None
    if not skip_refine:
        _log("\n" + "=" * 70)
        _log("  PHASE 2: Feedback-Driven Refinement")
        _log(f"  Max rounds: {max_rounds} | Pass threshold: 7/10")
        _log("=" * 70)

        refined = run_feedback_loop(
            plan=plan,
            model=model,
            max_rounds=max_rounds,
        )
    else:
        _log("\n  [Skipping refinement phase]")

    total_duration = time.time() - start

    # ── Phase 3: Save Outputs ─────────────────────────────────────────────
    _log("\n" + "=" * 70)
    _log("  PHASE 3: Saving Outputs")
    _log("=" * 70 + "\n")

    summary = build_pipeline_summary(plan, refined, total_duration, voice_name)

    json_path = os.path.join(output_dir, "pipeline_output.json")
    save_json(summary, json_path)

    md_text = build_full_markdown(plan, refined, total_duration, voice_name)
    md_path = os.path.join(output_dir, "pipeline_report.md")
    save_markdown(md_text, md_path)

    html_content = build_html_report(summary)
    html_path = os.path.join(output_dir, "pipeline_report.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    _log(f"  Saved: {html_path}")

    if refined:
        refine_json_path = os.path.join(output_dir, "refinement_detail.json")
        with open(refine_json_path, "w") as f:
            f.write(refined.to_json())
        _log(f"  Saved: {refine_json_path}")

    _log("\n" + "=" * 70)
    _log("  PIPELINE COMPLETE")
    _log(f"  Duration: {total_duration:.1f}s")
    _log(f"  Posts generated: {len(plan.posts)}")
    if refined:
        _log(f"  Refinement rounds: {len(refined.rounds)}")
        _log(f"  Posts refined: {refined.total_refinements}")
    _log(f"\n  Output directory: {output_dir}/")
    _log(f"    - pipeline_output.json   (structured data)")
    _log(f"    - pipeline_report.md     (readable report)")
    _log(f"    - pipeline_report.html   (visual report)")
    if refined:
        _log(f"    - refinement_detail.json (full refinement history)")
    _log("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweet Crumb Bakery — End-to-End Content Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full pipeline with default settings
  python demo_bakery_pipeline.py

  # Skip refinement (faster, generation only)
  python demo_bakery_pipeline.py --skip-refine

  # Use a different voice profile
  python demo_bakery_pipeline.py --voice upscale

  # Custom model and output directory
  python demo_bakery_pipeline.py --model claude-opus-4-7-20250501 --output-dir my_output
""",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum refinement rounds (default: 3)",
    )
    parser.add_argument(
        "--skip-refine",
        action="store_true",
        help="Skip the feedback refinement phase",
    )
    parser.add_argument(
        "--voice",
        choices=list(VOICE_VARIANTS.keys()),
        default="neighborhood",
        help="Brand voice variant: neighborhood (default), upscale, or playful",
    )
    parser.add_argument(
        "--output-dir",
        default="demo_output",
        help="Output directory (default: demo_output/)",
    )
    args = parser.parse_args()

    run_pipeline(
        model=args.model,
        max_rounds=args.max_rounds,
        skip_refine=args.skip_refine,
        voice_name=args.voice,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
