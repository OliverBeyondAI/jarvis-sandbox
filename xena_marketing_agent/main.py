#!/usr/bin/env python3
"""
Xena Marketing Agent — CLI Entry Point

Two modes:
  1. Campaign mode (default) — Full marketing campaign generation with research
  2. Social mode — 7-day social media content plan with optional feedback refinement

Usage:
    # Campaign mode
    python -m xena_marketing_agent "Launch campaign for our new AI productivity tool"
    python -m xena_marketing_agent --demo saas_launch
    python -m xena_marketing_agent --demo wellness_brand --voice luxury

    # Social media mode
    python -m xena_marketing_agent social --demo wellness_app
    python -m xena_marketing_agent social --demo coffee_brand --refine
    python -m xena_marketing_agent social --product-name "Acme" --product-desc "..." --audience "..."

    python -m xena_marketing_agent --list-demos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from .config import Config, ProductInfo
from .models import BrandVoice, SocialProductInfo


# ===========================================================================
# Campaign Demo Scenarios
# ===========================================================================

CAMPAIGN_DEMOS = {
    "saas_launch": {
        "title": "B2B SaaS Product Launch",
        "prompt": textwrap.dedent("""\
            Create a full marketing campaign for the launch of FlowDesk —
            an AI-powered project management platform for engineering teams.

            Generate compelling content across all channels that positions
            FlowDesk as the intelligent alternative to legacy tools like Jira
            and Asana. Emphasize the AI-native workflow automation, natural
            language task creation, and predictive sprint planning features.
        """),
        "product": ProductInfo(
            name="FlowDesk",
            tagline="Project management that thinks ahead",
            description=(
                "AI-powered project management platform for engineering teams. "
                "Natural language task creation, predictive sprint planning, "
                "and intelligent workflow automation."
            ),
            category="B2B SaaS / Developer Tools",
            target_audience="Engineering managers, tech leads, and CTOs at "
                            "mid-market and enterprise companies (50-5000 employees)",
            key_features=[
                "Natural language task creation and updates",
                "AI-powered sprint planning with velocity predictions",
                "Automated standup summaries and blockers detection",
                "Smart dependency mapping across teams",
                "Real-time codebase integration (GitHub, GitLab)",
            ],
            differentiators=[
                "AI-native — not a bolt-on feature",
                "Learns team patterns to improve estimates over time",
                "Zero-config GitHub integration with PR-to-task linking",
            ],
            pricing_info="Free tier, Pro at $12/user/month, Enterprise custom",
            competitors=["Jira", "Asana", "Linear", "Shortcut"],
        ),
        "voice": "startup",
        "channels": ["landing_page", "email_sequence", "social_media", "blog_post"],
    },
    "wellness_brand": {
        "title": "D2C Wellness Brand",
        "prompt": textwrap.dedent("""\
            Create a marketing campaign for Solara — a premium adaptogen
            supplement line targeting busy professionals who want sustained
            energy without caffeine crashes.

            The campaign should feel elevated and aspirational while being
            grounded in science. Position Solara as the sophisticated choice
            for high-performers who care about what they put in their bodies.
        """),
        "product": ProductInfo(
            name="Solara",
            tagline="Sustained energy, naturally elevated",
            description=(
                "Premium adaptogen supplement line combining clinically-studied "
                "mushroom extracts and adaptogens for sustained energy, focus, "
                "and stress resilience — without caffeine or stimulants."
            ),
            category="D2C Wellness / Supplements",
            target_audience="Health-conscious professionals aged 28-45 who "
                            "value premium quality and scientific backing",
            key_features=[
                "Clinical-grade lion's mane, ashwagandha, and rhodiola",
                "Third-party tested for purity and potency",
                "No caffeine, no stimulants, no crash",
                "Sustainably sourced, vegan, non-GMO",
                "30-day and 90-day subscription options",
            ],
            differentiators=[
                "Clinical dosages (not pixie-dusted)",
                "Transparent sourcing with farm-to-capsule traceability",
                "Designed by a neuroscientist and naturopathic doctor",
            ],
            pricing_info="$49/month (30-day) or $129/quarter (save 12%)",
            competitors=["Athletic Greens", "Mud/Wtr", "Four Sigmatic", "Onnit"],
        ),
        "voice": "luxury",
        "channels": ["landing_page", "email_sequence", "social_media", "ad_copy"],
    },
    "dev_tool": {
        "title": "Developer Tool Launch",
        "prompt": textwrap.dedent("""\
            Create a marketing campaign for VectorForge — an open-source
            vector database optimized for AI agent memory and retrieval.

            Target AI/ML engineers and developers building LLM-powered
            applications. Emphasize performance benchmarks, ease of
            integration, and the open-source community angle.
        """),
        "product": ProductInfo(
            name="VectorForge",
            tagline="The vector database built for AI agents",
            description=(
                "Open-source vector database purpose-built for AI agent memory, "
                "RAG pipelines, and semantic search. Sub-millisecond retrieval "
                "at billion-scale with a single-binary deployment."
            ),
            category="Developer Tools / Infrastructure",
            target_audience="AI/ML engineers, backend developers, and platform "
                            "teams building LLM-powered applications",
            key_features=[
                "Sub-millisecond vector search at billion-scale",
                "Single binary — no JVM, no cluster management",
                "Native Python, TypeScript, and Go SDKs",
                "Built-in hybrid search (vector + full-text + metadata)",
                "Agent memory primitives (sessions, threads, summaries)",
            ],
            differentiators=[
                "10x faster than Pinecone on standard benchmarks",
                "Single binary deployment — production-ready in 5 minutes",
                "Purpose-built agent memory API (not just a vector store)",
            ],
            pricing_info="Open-source (Apache 2.0), Cloud hosted at $0.10/GB/month",
            competitors=["Pinecone", "Weaviate", "Qdrant", "ChromaDB"],
        ),
        "voice": "technical",
        "channels": ["landing_page", "blog_post", "social_media", "product_description"],
    },
}


# ===========================================================================
# Social Media Demo Scenarios
# ===========================================================================

SOCIAL_DEMOS = {
    "wellness_app": {
        "product": SocialProductInfo(
            name="Serenity — Mindfulness & Sleep App",
            description=(
                "A premium mindfulness and sleep improvement app featuring AI-personalized "
                "meditation sessions, sleep stories narrated by calming voices, breathing "
                "exercises, and sleep quality tracking."
            ),
            target_audience=(
                "Stressed professionals aged 28-45 who struggle with sleep quality "
                "and work-life balance. Tech-savvy, health-conscious, active on Instagram and LinkedIn."
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
            do_use=["gentle encouragement", "science-backed language", "personal 'you' address", "calming imagery"],
            avoid=["aggressive sales language", "fear-based messaging", "clinical jargon", "all-caps emphasis"],
            hashtag_style="moderate",
            emoji_usage="moderate",
        ),
    },
    "saas_tool": {
        "product": SocialProductInfo(
            name="ShipFast — Developer Deployment Platform",
            description=(
                "A modern deployment platform for developer teams that turns git pushes "
                "into production deployments in under 60 seconds."
            ),
            target_audience=(
                "Software engineering teams and DevOps engineers at startups and mid-size "
                "companies. Frustrated with complex CI/CD pipelines. Active on Twitter/X and LinkedIn."
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
            do_use=["developer slang and references", "concrete metrics and numbers", "code snippets in posts", "memes and tech humor"],
            avoid=["buzzwords without substance", "enterprise jargon", "overly formal language", "generic stock photo vibes"],
            hashtag_style="minimal",
            emoji_usage="minimal",
        ),
    },
    "coffee_brand": {
        "product": SocialProductInfo(
            name="Ember & Oak — Small-Batch Coffee Roasters",
            description=(
                "An artisan coffee roastery offering single-origin, ethically sourced beans "
                "roasted in small batches. Direct trade relationships with farmers."
            ),
            target_audience=(
                "Coffee enthusiasts aged 25-40 who care about quality, sustainability, and "
                "the story behind their cup. Active on Instagram and TikTok."
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
            do_use=["sensory language (aroma, flavor notes)", "storytelling about origins and farmers", "cozy, inviting imagery", "behind-the-scenes of roasting process"],
            avoid=["generic coffee cliches ('but first, coffee')", "corporate language", "discount-heavy messaging", "stock photography aesthetic"],
            hashtag_style="moderate",
            emoji_usage="moderate",
        ),
    },
}


SOCIAL_VOICE_PRESETS = {
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


# ===========================================================================
# Campaign Runner
# ===========================================================================


def run_campaign(
    prompt: str,
    product: ProductInfo | None = None,
    voice: str = "professional",
    channels: list[str] | None = None,
    output_dir: str | None = None,
    model: str | None = None,
    output_json: bool = False,
) -> None:
    """Run the Xena Marketing Agent on a campaign brief."""
    from .agent import XenaMarketingAgent

    overrides: dict = {}
    if product:
        overrides["product"] = product
    if voice:
        overrides["brand_voice_preset"] = voice
    if channels:
        overrides["channels"] = channels
    if output_dir:
        overrides["output_dir"] = output_dir
    if model:
        overrides["model"] = model

    config = Config.from_env(**overrides)

    # Validate config
    warnings = config.validate()
    for w in warnings:
        print(f"[warning] {w}", file=sys.stderr)

    agent = XenaMarketingAgent(config=config)

    print("\n" + "=" * 70, file=sys.stderr)
    print("  XENA MARKETING AGENT", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Product:    {config.product.name}", file=sys.stderr)
    print(f"  Voice:      {config.brand_voice_preset}", file=sys.stderr)
    print(f"  Channels:   {', '.join(config.channels)}", file=sys.stderr)
    print(f"  Model:      {config.model}", file=sys.stderr)
    print(f"  Output:     {config.output_dir}", file=sys.stderr)
    print(f"  API:        Managed Agents (beta)", file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)

    result = agent.run(prompt)

    if output_json:
        print(
            json.dumps(
                {
                    "text": result.text,
                    "campaign_path": result.campaign_path,
                    "search_count": result.search_count,
                    "fetch_count": result.fetch_count,
                    "analyze_count": result.analyze_count,
                    "draft_count": result.draft_count,
                    "content_pieces": result.content_pieces,
                    "insights_count": len(result.insights),
                    "tool_calls_count": len(result.tool_calls),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "duration_seconds": result.duration_seconds,
                    "timestamp": result.timestamp,
                },
                indent=2,
                default=str,
            )
        )
    else:
        if result.campaign_path:
            print(f"\nCampaign saved to: {result.campaign_path}")
        print(f"\nCampaign completed in {result.duration_seconds:.1f}s")
        print(
            f"  Searches: {result.search_count} | "
            f"Pages read: {result.fetch_count} | "
            f"Analysis phases: {result.analyze_count} | "
            f"Content pieces: {result.draft_count}"
        )

        if result.content_pieces:
            print("\n  Content generated:")
            for piece in result.content_pieces:
                print(f"    - [{piece['channel']}] {piece['title']}")

        if result.text:
            print(f"\n{'=' * 70}")
            print(result.text)


# ===========================================================================
# Social Content Runner
# ===========================================================================


def run_social(args: argparse.Namespace) -> None:
    """Run the social content agent with optional feedback refinement."""
    from .social_agent import run_social_agent, DEFAULT_MODEL
    from .feedback_loop import run_feedback_loop
    from .models import RefinedContentPlan, WeeklyContentPlan

    model = args.model or DEFAULT_MODEL

    if args.demo:
        if args.demo not in SOCIAL_DEMOS:
            print(f"Unknown social demo: {args.demo}", file=sys.stderr)
            print(f"Available: {', '.join(SOCIAL_DEMOS.keys())}", file=sys.stderr)
            sys.exit(1)
        scenario = SOCIAL_DEMOS[args.demo]
        product = scenario["product"]
        brand_voice = scenario["brand_voice"]
    else:
        if not args.product_name or not args.product_desc or not args.audience:
            print("Error: --product-name, --product-desc, and --audience are required for custom products", file=sys.stderr)
            sys.exit(1)
        product = SocialProductInfo(
            name=args.product_name,
            description=args.product_desc,
            target_audience=args.audience,
            key_features=args.features or [],
        )
        brand_voice = SOCIAL_VOICE_PRESETS.get(args.voice, SOCIAL_VOICE_PRESETS["startup"])

    plan = run_social_agent(product=product, brand_voice=brand_voice, model=model)

    if args.refine:
        refined = run_feedback_loop(
            plan,
            model=model,
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


def _write_feedback_viewer(refined, path: str) -> None:
    """Write an HTML feedback viewer with the refinement data embedded."""
    viewer_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_path = os.path.join(viewer_dir, "feedback_viewer.html")
    with open(viewer_path, "r") as f:
        html = f.read()
    json_data = refined.to_json()
    html = html.replace("__REFINEMENT_DATA_PLACEHOLDER__", json_data)
    with open(path, "w") as f:
        f.write(html)


def _write_html_viewer(plan, path: str) -> None:
    """Write an HTML viewer with the plan data embedded."""
    viewer_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_path = os.path.join(viewer_dir, "viewer.html")
    with open(viewer_path, "r") as f:
        html = f.read()
    json_data = plan.to_json()
    html = html.replace("__PLAN_DATA_PLACEHOLDER__", json_data)
    with open(path, "w") as f:
        f.write(html)


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="xena_marketing_agent",
        description=(
            "Xena Marketing Agent — autonomous marketing content generator "
            "powered by Claude Opus 4.7 and the Anthropic managed-agents API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Campaign mode (default):
              python -m xena_marketing_agent "Launch campaign for our AI writing tool"
              python -m xena_marketing_agent --demo saas_launch
              python -m xena_marketing_agent --demo wellness_brand --voice luxury

            Social media mode:
              python -m xena_marketing_agent social --demo wellness_app
              python -m xena_marketing_agent social --demo coffee_brand --refine
              python -m xena_marketing_agent social --demo saas_tool --html output.html

            Campaign demos: saas_launch, wellness_brand, dev_tool
            Social demos:   wellness_app, saas_tool, coffee_brand
            Brand voices:   professional, startup, luxury, technical, friendly
        """),
    )

    subparsers = parser.add_subparsers(dest="mode")

    # --- Social subcommand ---
    social_parser = subparsers.add_parser(
        "social",
        help="Generate a 7-day social media content plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    social_parser.add_argument("--demo", choices=list(SOCIAL_DEMOS.keys()), help="Run a pre-built social demo")
    social_parser.add_argument("--product-name", help="Product/service name")
    social_parser.add_argument("--product-desc", help="Product/service description")
    social_parser.add_argument("--audience", help="Target audience description")
    social_parser.add_argument("--features", nargs="+", help="Key features")
    social_parser.add_argument("--voice", choices=list(SOCIAL_VOICE_PRESETS.keys()), default="startup", help="Brand voice preset")
    social_parser.add_argument("--model", default=None, help="Claude model (default: claude-opus-4-7-20250501)")
    social_parser.add_argument("--refine", action="store_true", help="Run feedback loop for iterative refinement")
    social_parser.add_argument("--max-rounds", type=int, default=3, help="Max refinement rounds (default: 3)")
    social_parser.add_argument("--pass-threshold", type=int, default=7, help="Score threshold for pass (1-10)")
    social_parser.add_argument("--json", action="store_true", help="Output as JSON")
    social_parser.add_argument("--html", metavar="FILE", help="Output as HTML viewer to FILE")
    social_parser.add_argument("--output", "-o", metavar="FILE", help="Save markdown output to FILE")

    # --- Campaign args (top-level) ---
    parser.add_argument("prompt", nargs="?", help="Marketing brief or campaign description")
    parser.add_argument("--demo", type=str, choices=list(CAMPAIGN_DEMOS.keys()), default=None, help="Run a pre-built campaign demo")
    parser.add_argument("--list-demos", action="store_true", help="List all available demos and exit")
    parser.add_argument("--voice", "-v", type=str, default=None, choices=["professional", "startup", "luxury", "technical"], help="Brand voice preset")
    parser.add_argument("--channels", "-c", type=str, nargs="+", default=None, help="Marketing channels")
    parser.add_argument("--product-name", type=str, default=None, help="Product name")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--model", "-m", type=str, default=None, help="Override Claude model")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # List demos
    if args.list_demos:
        print("\nCampaign demos:\n")
        for key, demo in CAMPAIGN_DEMOS.items():
            print(f"  {key:20s} {demo['title']}")
        print("\nSocial media demos:\n")
        for key in SOCIAL_DEMOS:
            scenario = SOCIAL_DEMOS[key]
            print(f"  {key:20s} {scenario['product'].name}")
        print(f"\nUsage:")
        print(f"  python -m xena_marketing_agent --demo <name>")
        print(f"  python -m xena_marketing_agent social --demo <name>\n")
        return

    # Social mode
    if args.mode == "social":
        run_social(args)
        return

    # Campaign mode
    product = None
    voice = args.voice or "professional"
    channels = args.channels

    if args.demo:
        demo = CAMPAIGN_DEMOS[args.demo]
        prompt = demo["prompt"]
        product = demo.get("product")
        voice = args.voice or demo.get("voice", "professional")
        channels = args.channels or demo.get("channels")
        print(f"Running demo: {demo['title']}", file=sys.stderr)
    elif args.prompt:
        prompt = args.prompt
        if args.product_name:
            product = ProductInfo(name=args.product_name)
    else:
        parser.print_help()
        print(
            "\nError: Provide a prompt, --demo, or use 'social' subcommand.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_campaign(
        prompt=prompt,
        product=product,
        voice=voice,
        channels=channels,
        output_dir=args.output,
        model=args.model,
        output_json=args.json,
    )


if __name__ == "__main__":
    main()
