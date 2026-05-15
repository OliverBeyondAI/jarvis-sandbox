#!/usr/bin/env python3
"""
MCP Xena Marketing Agent — CLI Entry Point

Usage:
    # Run a demo campaign
    python -m agents.mcp_xena_marketing_agent --demo saas_launch
    python -m agents.mcp_xena_marketing_agent --demo wellness_brand --voice luxury
    python -m agents.mcp_xena_marketing_agent --demo dev_tool --thinking deep

    # Custom campaign
    python -m agents.mcp_xena_marketing_agent "Launch campaign for our AI writing tool"

    # MCP server discovery (no API key needed)
    python -m agents.mcp_xena_marketing_agent --discover

    # List available demos
    python -m agents.mcp_xena_marketing_agent --list-demos
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

from .config import Config, ProductInfo
from .viewer_gen import generate_html_viewer


# ===========================================================================
# Demo Scenarios
# ===========================================================================

DEMOS: dict[str, dict[str, Any]] = {
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
            target_audience=(
                "Engineering managers, tech leads, and CTOs at "
                "mid-market and enterprise companies (50-5000 employees)"
            ),
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
            target_audience=(
                "Health-conscious professionals aged 28-45 who "
                "value premium quality and scientific backing"
            ),
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
            target_audience=(
                "AI/ML engineers, backend developers, and platform "
                "teams building LLM-powered applications"
            ),
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
# MCP Server Discovery (no API key needed)
# ===========================================================================


async def discover_tools() -> None:
    """Connect to the MCP server and list discovered tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_script = str(Path(__file__).parent / "mcp_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
    )

    print("=" * 60)
    print("  MCP Xena Marketing Agent — Tool Discovery")
    print("=" * 60)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.list_tools()

            print(f"\nDiscovered {len(result.tools)} tools:\n")
            total_bytes = 0
            for tool in result.tools:
                schema_json = json.dumps(tool.inputSchema, indent=2)
                schema_bytes = len(schema_json)
                total_bytes += schema_bytes
                desc = (tool.description or "").split("\n")[0]
                print(f"  {tool.name}")
                print(f"    Description: {desc[:80]}")
                print(f"    Schema size: {schema_bytes} bytes (~{schema_bytes // 4} tokens)")
                print()

            print(f"Total schema payload: {total_bytes} bytes (~{total_bytes // 4} tokens)")
            print(f"MCP server: {server_script}")
            print()


# ===========================================================================
# Campaign Runner
# ===========================================================================


async def run_campaign(
    prompt: str,
    product: ProductInfo | None = None,
    voice: str = "professional",
    channels: list[str] | None = None,
    output_dir: str | None = None,
    model: str | None = None,
    thinking: str | None = None,
    no_images: bool = False,
    output_json: bool = False,
    html_output: bool = False,
) -> None:
    """Run the MCP Xena Marketing Agent on a campaign brief."""
    from .agent import MCPXenaMarketingAgent

    overrides: dict[str, Any] = {}
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
    if thinking:
        overrides["thinking_preset"] = thinking
    if no_images:
        overrides["generate_images"] = False

    config = Config.from_env(**overrides)

    # Validate config
    warnings = config.validate()
    for w in warnings:
        print(f"[warning] {w}", file=sys.stderr)

    agent = MCPXenaMarketingAgent(config=config)

    print("\n" + "=" * 70, file=sys.stderr)
    print("  MCP XENA MARKETING AGENT", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Product:    {config.product.name}", file=sys.stderr)
    print(f"  Voice:      {config.brand_voice_preset}", file=sys.stderr)
    print(f"  Channels:   {', '.join(config.channels)}", file=sys.stderr)
    print(f"  Model:      {config.model}", file=sys.stderr)
    print(f"  Thinking:   {config.thinking_preset} "
          f"({config.resolved_thinking_budget} tokens)", file=sys.stderr)
    print(f"  Images:     {'enabled' if config.generate_images else 'disabled'}",
          file=sys.stderr)
    print(f"  Output:     {config.output_dir}", file=sys.stderr)
    print(f"  API:        MCP + Claude Agent SDK", file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)

    result = await agent.run(prompt)

    result_dict = {
        "product_name": config.product.name,
        "product_tagline": config.product.tagline,
        "brand_voice": config.brand_voice_preset,
        "channels": config.channels,
        "prompt": prompt,
        "text": result.text,
        "campaign_path": result.campaign_path,
        "search_count": result.search_count,
        "fetch_count": result.fetch_count,
        "analyze_count": result.analyze_count,
        "draft_count": result.draft_count,
        "image_count": result.image_count,
        "content_pieces": result.content_pieces,
        "images_generated": result.images_generated,
        "insights": result.insights,
        "tool_calls": [
            {"name": tc["name"], "input_keys": list(tc["input"].keys())}
            for tc in result.tool_calls
        ],
        "mcp_tools_discovered": result.mcp_tools_discovered,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "duration_seconds": round(result.duration_seconds, 1),
        "timestamp": result.timestamp,
    }

    # Generate HTML viewer if requested
    if html_output:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = str(out_dir / "campaign_viewer.html")
        generated = generate_html_viewer(result_dict, html_path)
        print(f"\nHTML Viewer: {generated}", file=sys.stderr)

    if output_json:
        print(json.dumps(result_dict, indent=2, default=str))
    else:
        if result.campaign_path:
            print(f"\nCampaign saved to: {result.campaign_path}")
        print(f"\nCampaign completed in {result.duration_seconds:.1f}s")
        print(
            f"  Searches: {result.search_count} | "
            f"Pages read: {result.fetch_count} | "
            f"Analysis phases: {result.analyze_count} | "
            f"Content pieces: {result.draft_count} | "
            f"Images: {result.image_count}"
        )

        if result.content_pieces:
            print("\n  Content generated:")
            for piece in result.content_pieces:
                print(f"    - [{piece['channel']}] {piece['title']}")

        if result.images_generated:
            print("\n  Images generated:")
            for img in result.images_generated:
                print(f"    - [{img['channel']}] {img['file_path']}")

        if result.text:
            print(f"\n{'=' * 70}")
            print(result.text)


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mcp_xena_marketing_agent",
        description=(
            "MCP Xena Marketing Agent — autonomous marketing content generator "
            "powered by Claude Agent SDK, MCP tool servers, OpenAI image "
            "generation, and Tavily market research."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m agents.mcp_xena_marketing_agent --demo saas_launch
              python -m agents.mcp_xena_marketing_agent --demo wellness_brand --voice luxury
              python -m agents.mcp_xena_marketing_agent --demo dev_tool --thinking deep
              python -m agents.mcp_xena_marketing_agent "Launch campaign for our new AI tool"
              python -m agents.mcp_xena_marketing_agent --discover

            Demos:          saas_launch, wellness_brand, dev_tool
            Brand voices:   professional, startup, luxury, technical
            Thinking modes: minimal (4K), balanced (10K), deep (20K)
        """),
    )

    parser.add_argument(
        "prompt", nargs="?",
        help="Marketing brief or campaign description",
    )
    parser.add_argument(
        "--demo", type=str, choices=list(DEMOS.keys()),
        help="Run a pre-built campaign demo",
    )
    parser.add_argument(
        "--list-demos", action="store_true",
        help="List all available demos and exit",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Discover MCP server tools (no API key needed) and exit",
    )
    parser.add_argument(
        "--voice", "-v", type=str, default=None,
        choices=["professional", "startup", "luxury", "technical"],
        help="Brand voice preset",
    )
    parser.add_argument(
        "--channels", "-c", type=str, nargs="+",
        help="Marketing channels to generate content for",
    )
    parser.add_argument(
        "--thinking", "-t", type=str, default=None,
        choices=["minimal", "balanced", "deep"],
        help="Extended thinking budget preset (default: balanced)",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Disable image generation",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for campaign files",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Override Claude model",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Generate an HTML campaign viewer alongside the output",
    )

    args = parser.parse_args()

    # Discover mode
    if args.discover:
        asyncio.run(discover_tools())
        return

    # List demos
    if args.list_demos:
        print("\nAvailable campaign demos:\n")
        for key, demo in DEMOS.items():
            print(f"  {key:20s} {demo['title']}")
        print(f"\nUsage:")
        print(f"  python -m agents.mcp_xena_marketing_agent --demo <name>")
        print(f"  python -m agents.mcp_xena_marketing_agent --demo <name> --thinking deep")
        print()
        return

    # Campaign mode
    product = None
    voice = args.voice or "professional"
    channels = args.channels

    if args.demo:
        demo = DEMOS[args.demo]
        prompt = demo["prompt"]
        product = demo.get("product")
        voice = args.voice or demo.get("voice", "professional")
        channels = args.channels or demo.get("channels")
        print(f"Running demo: {demo['title']}", file=sys.stderr)
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        print(
            "\nError: Provide a prompt, --demo, or --discover.",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(run_campaign(
        prompt=prompt,
        product=product,
        voice=voice,
        channels=channels,
        output_dir=args.output,
        model=args.model,
        thinking=args.thinking,
        no_images=args.no_images,
        output_json=args.json,
        html_output=args.html,
    ))


if __name__ == "__main__":
    main()
