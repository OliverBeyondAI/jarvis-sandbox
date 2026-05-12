"""CLI entry point for the Social Content Agent."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .models import BrandVoice, ProductInfo, RefinedContentPlan, WeeklyContentPlan
from .agent import run_agent
from .feedback_loop import run_feedback_loop


# ---------------------------------------------------------------------------
# Pre-built demo scenarios
# ---------------------------------------------------------------------------

DEMO_SCENARIOS = {
    "wellness_app": {
        "product": ProductInfo(
            name="Serenity — Mindfulness & Sleep App",
            description=(
                "A premium mindfulness and sleep improvement app featuring AI-personalized "
                "meditation sessions, sleep stories narrated by calming voices, breathing "
                "exercises, and sleep quality tracking. Helps users build consistent "
                "mindfulness habits and achieve deeper, more restorative sleep."
            ),
            target_audience=(
                "Stressed professionals aged 28-45 who struggle with sleep quality "
                "and work-life balance. Tech-savvy, health-conscious, willing to invest "
                "in self-improvement tools. Active on Instagram and LinkedIn."
            ),
            key_features=[
                "AI-personalized meditation plans",
                "Sleep stories with 50+ narrators",
                "Smart alarm with sleep cycle tracking",
                "Breathing exercises for anxiety relief",
                "Progress tracking and streak system",
            ],
            differentiators=[
                "AI adapts to your stress patterns",
                "Only app combining meditation + sleep tracking",
                "Clinically validated techniques",
            ],
            category="Health & Wellness",
            website_url="https://serenity-app.example.com",
        ),
        "brand_voice": BrandVoice(
            tone="Warm, calming, and encouraging — like a supportive friend who also happens to be a wellness expert",
            personality="Empathetic, knowledgeable, gently motivating, never preachy",
            do_use=[
                "gentle encouragement",
                "science-backed language",
                "personal 'you' address",
                "calming imagery",
            ],
            avoid=[
                "aggressive sales language",
                "fear-based messaging",
                "clinical jargon",
                "all-caps emphasis",
            ],
            hashtag_style="moderate",
            emoji_usage="moderate",
        ),
    },
    "saas_tool": {
        "product": ProductInfo(
            name="ShipFast — Developer Deployment Platform",
            description=(
                "A modern deployment platform for developer teams that turns git pushes "
                "into production deployments in under 60 seconds. Features preview environments, "
                "automatic rollbacks, built-in monitoring, and team collaboration tools."
            ),
            target_audience=(
                "Software engineering teams and DevOps engineers at startups and mid-size "
                "companies (50-500 employees). Frustrated with complex CI/CD pipelines. "
                "Active on Twitter/X and LinkedIn. Value speed and developer experience."
            ),
            key_features=[
                "60-second deployments from git push",
                "Automatic preview environments for every PR",
                "One-click rollbacks",
                "Built-in error tracking and monitoring",
                "Team permissions and audit logs",
            ],
            differentiators=[
                "Fastest deployment times in the industry",
                "Zero-config setup for most frameworks",
                "Free tier with generous limits",
            ],
            category="Developer Tools / SaaS",
            website_url="https://shipfast.example.dev",
        ),
        "brand_voice": BrandVoice(
            tone="Confident, technical but accessible, with dry developer humor",
            personality="Sharp, efficient, slightly irreverent, deeply technical",
            do_use=[
                "developer slang and references",
                "concrete metrics and numbers",
                "code snippets in posts",
                "memes and tech humor",
            ],
            avoid=[
                "buzzwords without substance",
                "enterprise jargon",
                "overly formal language",
                "generic stock photo vibes",
            ],
            hashtag_style="minimal",
            emoji_usage="minimal",
        ),
    },
    "coffee_brand": {
        "product": ProductInfo(
            name="Ember & Oak — Small-Batch Coffee Roasters",
            description=(
                "An artisan coffee roastery offering single-origin, ethically sourced beans "
                "roasted in small batches. Direct trade relationships with farmers in Colombia, "
                "Ethiopia, and Guatemala. Subscription boxes and limited seasonal blends."
            ),
            target_audience=(
                "Coffee enthusiasts aged 25-40 who care about quality, sustainability, and "
                "the story behind their cup. Willing to pay premium for exceptional coffee. "
                "Active on Instagram and TikTok. Appreciate craft and artisanship."
            ),
            key_features=[
                "Single-origin beans from 3 regions",
                "Roasted within 48 hours of shipping",
                "Direct trade with farmer cooperatives",
                "Monthly subscription with tasting notes",
                "Seasonal limited-edition blends",
            ],
            differentiators=[
                "Direct farmer relationships (not just 'fair trade')",
                "Roast-to-ship freshness guarantee",
                "Each bag includes farmer's story and brew guide",
            ],
            category="Food & Beverage / E-commerce",
            website_url="https://emberandoak.example.com",
        ),
        "brand_voice": BrandVoice(
            tone="Warm, passionate, and artisanal — storytelling meets coffee expertise",
            personality="Curious, passionate, community-oriented, authentic",
            do_use=[
                "sensory language (aroma, flavor notes)",
                "storytelling about origins and farmers",
                "cozy, inviting imagery",
                "behind-the-scenes of roasting process",
            ],
            avoid=[
                "generic coffee clichés ('but first, coffee')",
                "corporate language",
                "discount-heavy messaging",
                "stock photography aesthetic",
            ],
            hashtag_style="moderate",
            emoji_usage="moderate",
        ),
    },
}


BRAND_VOICE_PRESETS = {
    "professional": BrandVoice(
        tone="Professional, authoritative, and trustworthy",
        personality="Expert, reliable, data-driven",
        do_use=["statistics", "industry terminology", "thought leadership"],
        avoid=["slang", "excessive emojis", "informal language"],
        hashtag_style="minimal",
        emoji_usage="minimal",
    ),
    "startup": BrandVoice(
        tone="Energetic, bold, and disruptive",
        personality="Ambitious, innovative, community-driven",
        do_use=["action verbs", "future-focused language", "community references"],
        avoid=["corporate jargon", "passive voice", "cautious hedging"],
        hashtag_style="moderate",
        emoji_usage="moderate",
    ),
    "luxury": BrandVoice(
        tone="Refined, aspirational, and exclusive",
        personality="Sophisticated, curated, understated",
        do_use=["sensory language", "exclusivity cues", "elegant phrasing"],
        avoid=["discount language", "urgency tactics", "casual slang"],
        hashtag_style="minimal",
        emoji_usage="none",
    ),
    "friendly": BrandVoice(
        tone="Warm, approachable, and conversational",
        personality="Helpful, cheerful, relatable",
        do_use=["questions", "personal stories", "inclusive language"],
        avoid=["technical jargon", "formal language", "sales pressure"],
        hashtag_style="moderate",
        emoji_usage="heavy",
    ),
}


def run_demo(scenario_name: str, model: str = "claude-sonnet-4-20250514") -> WeeklyContentPlan:
    """Run a pre-built demo scenario."""
    if scenario_name not in DEMO_SCENARIOS:
        print(f"Unknown demo: {scenario_name}", file=sys.stderr)
        print(f"Available demos: {', '.join(DEMO_SCENARIOS.keys())}", file=sys.stderr)
        sys.exit(1)

    scenario = DEMO_SCENARIOS[scenario_name]
    return run_agent(
        product=scenario["product"],
        brand_voice=scenario["brand_voice"],
        model=model,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Social Content Agent — Generate a 7-day social media content plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run a demo scenario
  python -m social_content_agent --demo wellness_app

  # Custom product with preset voice
  python -m social_content_agent \\
    --product-name "Acme Widget" \\
    --product-desc "A revolutionary widget for modern teams" \\
    --audience "Product managers at tech companies" \\
    --voice startup

  # Output as JSON
  python -m social_content_agent --demo coffee_brand --json

  # Output as HTML viewer
  python -m social_content_agent --demo saas_tool --html output.html
""",
    )
    parser.add_argument(
        "--demo",
        choices=list(DEMO_SCENARIOS.keys()),
        help="Run a pre-built demo scenario",
    )
    parser.add_argument("--product-name", help="Product/service name")
    parser.add_argument("--product-desc", help="Product/service description")
    parser.add_argument("--audience", help="Target audience description")
    parser.add_argument("--features", nargs="+", help="Key features (space-separated)")
    parser.add_argument(
        "--voice",
        choices=list(BRAND_VOICE_PRESETS.keys()),
        default="startup",
        help="Brand voice preset (default: startup)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Run the feedback loop to iteratively evaluate and refine generated posts",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum refinement rounds (default: 3, used with --refine)",
    )
    parser.add_argument(
        "--pass-threshold",
        type=int,
        default=7,
        help="Score threshold for a post to pass (1-10, default: 7, used with --refine)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--html", metavar="FILE", help="Output as HTML viewer to FILE")
    parser.add_argument(
        "--output", "-o", metavar="FILE", help="Save markdown output to FILE"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.demo and not args.product_name:
        parser.error("Either --demo or --product-name is required")

    # Run agent
    if args.demo:
        plan = run_demo(args.demo, model=args.model)
    else:
        if not args.product_desc:
            parser.error("--product-desc is required when not using --demo")
        if not args.audience:
            parser.error("--audience is required when not using --demo")

        product = ProductInfo(
            name=args.product_name,
            description=args.product_desc,
            target_audience=args.audience,
            key_features=args.features or [],
        )
        brand_voice = BRAND_VOICE_PRESETS[args.voice]
        plan = run_agent(product=product, brand_voice=brand_voice, model=args.model)

    # Optionally run feedback loop
    if args.refine:
        refined = run_feedback_loop(
            plan,
            model=args.model,
            max_rounds=args.max_rounds,
            pass_threshold=args.pass_threshold,
        )
        if args.json:
            print(refined.to_json())
        elif args.html:
            _write_feedback_viewer(refined, args.html)
            print(f"Feedback viewer written to {args.html}", file=sys.stderr)
        else:
            md = refined.to_markdown()
            if args.output:
                with open(args.output, "w") as f:
                    f.write(md)
                print(f"Markdown saved to {args.output}", file=sys.stderr)
            else:
                print(md)
    else:
        # Output
        if args.json:
            print(plan.to_json())
        elif args.html:
            _write_html_viewer(plan, args.html)
            print(f"HTML viewer written to {args.html}", file=sys.stderr)
        else:
            md = plan.to_markdown()
            if args.output:
                with open(args.output, "w") as f:
                    f.write(md)
                print(f"Markdown saved to {args.output}", file=sys.stderr)
            else:
                print(md)


def _write_feedback_viewer(refined: RefinedContentPlan, path: str) -> None:
    """Write an HTML feedback viewer with the refinement data embedded."""
    viewer_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_path = os.path.join(viewer_dir, "feedback_viewer.html")

    with open(viewer_path, "r") as f:
        html = f.read()

    json_data = refined.to_json()
    html = html.replace("__REFINEMENT_DATA_PLACEHOLDER__", json_data)

    with open(path, "w") as f:
        f.write(html)


def _write_html_viewer(plan: WeeklyContentPlan, path: str) -> None:
    """Write an HTML viewer with the plan data embedded."""
    viewer_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_path = os.path.join(viewer_dir, "viewer.html")

    with open(viewer_path, "r") as f:
        html = f.read()

    # Embed the plan data into the HTML
    json_data = plan.to_json()
    html = html.replace("__PLAN_DATA_PLACEHOLDER__", json_data)

    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
