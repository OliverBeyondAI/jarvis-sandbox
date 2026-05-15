#!/usr/bin/env python3
"""
MCP Xena Marketing Agent — End-to-End Workflow Runner

Standalone script that demonstrates the complete marketing agent pipeline:
  1. Connect to MCP server and discover tools
  2. Research a topic via Tavily web search
  3. Generate social media copy from research insights
  4. Create an accompanying image with text overlay
  5. Output a complete social media post package

Usage:
    # Discover MCP tools (no API key needed)
    python run_workflow.py --discover

    # Run a demo campaign (requires ANTHROPIC_API_KEY + TAVILY_API_KEY)
    python run_workflow.py --demo saas_launch

    # Run with a custom topic
    python run_workflow.py "AI-powered developer tools market trends"

    # Run with HTML output viewer
    python run_workflow.py --demo dev_tool --html

    # Dry-run: show config and system prompt without calling APIs
    python run_workflow.py --demo saas_launch --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# Ensure the package is importable when running standalone
_this_dir = Path(__file__).resolve().parent
_agents_dir = _this_dir.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from mcp_xena_marketing_agent.config import Config, ProductInfo
from mcp_xena_marketing_agent.agent import MCPXenaMarketingAgent, build_system_prompt
from mcp_xena_marketing_agent.viewer_gen import generate_html_viewer


# ── Demo Scenarios ──────────────────────────────────────────────────────────

DEMOS = {
    "saas_launch": {
        "title": "B2B SaaS Product Launch — FlowDesk",
        "prompt": (
            "Create a full social media marketing campaign for FlowDesk — "
            "an AI-powered project management platform for engineering teams. "
            "Research the current landscape of AI project management tools, "
            "then generate compelling social media posts that position FlowDesk "
            "as the intelligent alternative to legacy tools like Jira and Asana. "
            "Generate at least one hero image with text overlay."
        ),
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
        "channels": ["social_media", "landing_page", "blog_post"],
    },
    "wellness_brand": {
        "title": "D2C Wellness Brand — Solara",
        "prompt": (
            "Create a social media marketing campaign for Solara — a premium "
            "adaptogen supplement line. Research the current wellness supplement "
            "market trends and competitor positioning, then generate social media "
            "posts and a hero image that position Solara as the sophisticated "
            "choice for high-performers."
        ),
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
                "Health-conscious professionals aged 28-45 who value premium "
                "quality and scientific backing"
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
        "channels": ["social_media", "landing_page", "ad_copy"],
    },
    "dev_tool": {
        "title": "Developer Tool Launch — VectorForge",
        "prompt": (
            "Create a social media marketing campaign for VectorForge — an "
            "open-source vector database optimized for AI agent memory. "
            "Research current vector database benchmarks and trends, then "
            "generate developer-focused social media posts with performance "
            "data and a hero image showcasing the tech."
        ),
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
        "channels": ["social_media", "blog_post", "landing_page"],
    },
}


# ── MCP Server Discovery ───────────────────────────────────────────────────


async def discover_tools() -> dict:
    """Connect to the MCP server and list all discovered tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_script = str(_this_dir / "mcp_server.py")
    server_params = StdioServerParameters(
        command=sys.executable, args=[server_script]
    )

    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│         MCP Xena Marketing Agent — Tool Discovery          │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

            tools_info = []
            total_bytes = 0
            for tool in result.tools:
                schema_json = json.dumps(tool.inputSchema, indent=2)
                schema_bytes = len(schema_json)
                total_bytes += schema_bytes
                desc = (tool.description or "").split("\n")[0]
                params = list(tool.inputSchema.get("properties", {}).keys())

                info = {
                    "name": tool.name,
                    "description": desc[:100],
                    "parameters": params,
                    "schema_bytes": schema_bytes,
                    "schema_tokens": schema_bytes // 4,
                }
                tools_info.append(info)

                print(f"  ✓ {tool.name}")
                print(f"    {desc[:80]}")
                print(f"    Parameters: {', '.join(params)}")
                print(f"    Schema: {schema_bytes} bytes (~{schema_bytes // 4} tokens)")
                print()

            print(f"  Total: {len(tools_info)} tools, "
                  f"{total_bytes} bytes (~{total_bytes // 4} tokens)")
            print(f"  Server: {server_script}")
            print()

            return {
                "tools": tools_info,
                "total_schema_bytes": total_bytes,
                "total_schema_tokens": total_bytes // 4,
                "server_script": server_script,
            }


# ── Dry Run ─────────────────────────────────────────────────────────────────


def dry_run(config: Config, prompt: str) -> None:
    """Show configuration and system prompt without calling any APIs."""
    system_prompt = build_system_prompt(config)

    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│         MCP Xena Marketing Agent — Dry Run                 │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Product:      {config.product.name}")
    print(f"  Tagline:      {config.product.tagline}")
    print(f"  Brand Voice:  {config.brand_voice_preset}")
    print(f"  Channels:     {', '.join(config.channels)}")
    print(f"  Model:        {config.model}")
    print(f"  Thinking:     {config.thinking_preset} "
          f"({config.resolved_thinking_budget} tokens)")
    print(f"  Images:       {'enabled' if config.generate_images else 'disabled'}")
    print(f"  Output:       {config.output_dir}")
    print(f"  MCP Server:   {config.mcp_server_script}")
    print()

    warnings = config.validate()
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    ⚠  {w}")
        print()

    print("  ── User Prompt ──")
    print()
    for line in prompt.strip().splitlines():
        print(f"    {line}")
    print()
    print(f"  ── System Prompt ({len(system_prompt)} chars) ──")
    print()
    for line in system_prompt.strip().splitlines()[:40]:
        print(f"    {line}")
    if len(system_prompt.strip().splitlines()) > 40:
        print(f"    ... ({len(system_prompt.strip().splitlines()) - 40} more lines)")
    print()


# ── Main Workflow Runner ───────────────────────────────────────────────────


async def run_workflow(
    prompt: str,
    product: ProductInfo | None = None,
    voice: str = "professional",
    channels: list[str] | None = None,
    output_dir: str | None = None,
    thinking: str = "balanced",
    no_images: bool = False,
    html_output: bool = False,
) -> dict:
    """Run the complete end-to-end marketing workflow."""

    overrides: dict = {}
    if product:
        overrides["product"] = product
    if voice:
        overrides["brand_voice_preset"] = voice
    if channels:
        overrides["channels"] = channels
    if output_dir:
        overrides["output_dir"] = output_dir
    if thinking:
        overrides["thinking_preset"] = thinking
    if no_images:
        overrides["generate_images"] = False

    config = Config.from_env(**overrides)
    warnings = config.validate()

    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│         MCP Xena Marketing Agent — Workflow Runner         │")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Product:    {config.product.name}")
    print(f"  Voice:      {config.brand_voice_preset}")
    print(f"  Channels:   {', '.join(config.channels)}")
    print(f"  Model:      {config.model}")
    print(f"  Thinking:   {config.thinking_preset} "
          f"({config.resolved_thinking_budget} tokens)")
    print(f"  Images:     {'enabled' if config.generate_images else 'disabled'}")
    print(f"  Output:     {config.output_dir}")
    print()

    for w in warnings:
        print(f"  ⚠  {w}", file=sys.stderr)

    agent = MCPXenaMarketingAgent(config=config)
    result = await agent.run(prompt)

    # Build result JSON
    result_json = {
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

    # Save JSON result
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "workflow_result.json"
    json_path.write_text(json.dumps(result_json, indent=2, default=str))
    print(f"\n  Result JSON: {json_path.resolve()}")

    # Generate HTML viewer
    if html_output:
        html_path = str(out_dir / "campaign_viewer.html")
        generated = generate_html_viewer(result_json, html_path)
        print(f"  HTML Viewer: {generated}")
        result_json["html_viewer_path"] = generated

    # Print summary
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    Workflow Complete                        │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"  Market searches:    {result.search_count}")
    print(f"  Pages fetched:      {result.fetch_count}")
    print(f"  Analysis phases:    {result.analyze_count}")
    print(f"  Content drafted:    {result.draft_count}")
    print(f"  Images generated:   {result.image_count}")
    print(f"  Total tool calls:   {len(result.tool_calls)}")
    print(f"  Tokens:             {result.input_tokens:,} in / "
          f"{result.output_tokens:,} out")
    print(f"  Duration:           {result.duration_seconds:.1f}s")

    if result.content_pieces:
        print()
        print("  Content:")
        for piece in result.content_pieces:
            print(f"    [{piece['channel']}] {piece['title']}")

    if result.images_generated:
        print()
        print("  Images:")
        for img in result.images_generated:
            label = img.get("platform_label") or img.get("channel", "")
            print(f"    [{label}] {img.get('file_path', 'N/A')}")

    if result.campaign_path:
        print(f"\n  Campaign file: {result.campaign_path}")

    print()
    return result_json


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_workflow",
        description=(
            "MCP Xena Marketing Agent — End-to-End Workflow Runner.\n\n"
            "Orchestrates the complete pipeline: research → copy → image → package."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_workflow.py --discover
              python run_workflow.py --demo saas_launch
              python run_workflow.py --demo dev_tool --html --thinking deep
              python run_workflow.py --demo wellness_brand --dry-run
              python run_workflow.py "Create a campaign for our AI coding assistant"

            Demos:          saas_launch, wellness_brand, dev_tool
            Brand voices:   professional, startup, luxury, technical
            Thinking modes: minimal (4K), balanced (10K), deep (20K)
        """),
    )

    parser.add_argument("prompt", nargs="?", help="Campaign topic or brief")
    parser.add_argument("--demo", choices=list(DEMOS.keys()), help="Run a demo")
    parser.add_argument("--list-demos", action="store_true", help="List demos")
    parser.add_argument("--discover", action="store_true",
                        help="Discover MCP tools (no API key needed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config and prompt without calling APIs")
    parser.add_argument("--voice", "-v",
                        choices=["professional", "startup", "luxury", "technical"])
    parser.add_argument("--channels", "-c", nargs="+")
    parser.add_argument("--thinking", "-t",
                        choices=["minimal", "balanced", "deep"], default="balanced")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--html", action="store_true",
                        help="Generate HTML campaign viewer")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Print result as JSON to stdout")

    args = parser.parse_args()

    # ── Discover ──
    if args.discover:
        asyncio.run(discover_tools())
        return

    # ── List demos ──
    if args.list_demos:
        print("\nAvailable demos:\n")
        for key, demo in DEMOS.items():
            print(f"  {key:20s} {demo['title']}")
        print()
        return

    # ── Resolve prompt and config ──
    product = None
    voice = args.voice or "professional"
    channels = args.channels

    if args.demo:
        demo = DEMOS[args.demo]
        prompt = demo["prompt"]
        product = demo.get("product")
        voice = args.voice or demo.get("voice", "professional")
        channels = args.channels or demo.get("channels")
        print(f"\n  Demo: {demo['title']}", file=sys.stderr)
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        print("\nError: provide a prompt, --demo, or --discover.", file=sys.stderr)
        sys.exit(1)

    # ── Dry run ──
    if args.dry_run:
        overrides: dict = {}
        if product:
            overrides["product"] = product
        if voice:
            overrides["brand_voice_preset"] = voice
        if channels:
            overrides["channels"] = channels
        if args.thinking:
            overrides["thinking_preset"] = args.thinking
        if args.no_images:
            overrides["generate_images"] = False
        config = Config.from_env(**overrides)
        dry_run(config, prompt)
        return

    # ── Full workflow ──
    result = asyncio.run(run_workflow(
        prompt=prompt,
        product=product,
        voice=voice,
        channels=channels,
        output_dir=args.output,
        thinking=args.thinking,
        no_images=args.no_images,
        html_output=args.html,
    ))

    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
