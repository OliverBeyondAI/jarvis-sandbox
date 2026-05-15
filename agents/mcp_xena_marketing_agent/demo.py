#!/usr/bin/env python3
"""
MCP Xena Marketing Agent — Demo Script

Interactive demo showcasing all major capabilities:
  1. Environment validation and API key checks
  2. MCP server tool discovery
  3. Configuration and brand voice preview
  4. Full campaign execution with error handling

Usage:
    # Check environment and discover tools (no API keys needed)
    python demo.py --check

    # Run a quick demo campaign (requires API keys)
    python demo.py

    # Run a specific demo scenario
    python demo.py --scenario saas_launch
    python demo.py --scenario wellness_brand
    python demo.py --scenario dev_tool

    # Custom campaign with options
    python demo.py --prompt "Launch campaign for our AI coding assistant" \\
                   --voice startup --thinking deep

    # Skip image generation (only needs ANTHROPIC_API_KEY + TAVILY_API_KEY)
    python demo.py --no-images

    # Generate HTML viewer alongside campaign output
    python demo.py --scenario dev_tool --html

Environment Variables:
    ANTHROPIC_API_KEY   (required) Claude API key for agent reasoning
    TAVILY_API_KEY      (recommended) Tavily API key for web search
    OPENAI_API_KEY      (optional) OpenAI API key for image generation
    XENA_OUTPUT_DIR     (optional) Output directory (default: ./marketing_output)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the package is importable when running this script directly
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
_agents_dir = _this_dir.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from mcp_xena_marketing_agent.config import Config, ProductInfo, BRAND_VOICE_PRESETS
from mcp_xena_marketing_agent.agent import MCPXenaMarketingAgent, build_system_prompt
from mcp_xena_marketing_agent.viewer_gen import generate_html_viewer


# ═══════════════════════════════════════════════════════════════════════════
# Demo Scenarios
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS: dict[str, dict] = {
    "saas_launch": {
        "title": "B2B SaaS Product Launch — FlowDesk",
        "description": "AI-powered project management platform for engineering teams",
        "prompt": textwrap.dedent("""\
            Create a full marketing campaign for FlowDesk — an AI-powered
            project management platform for engineering teams.

            Research the current landscape of AI project management tools,
            then generate compelling content that positions FlowDesk as the
            intelligent alternative to legacy tools like Jira and Asana.
            Emphasize AI-native workflow automation, natural language task
            creation, and predictive sprint planning.
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
        "title": "D2C Wellness Brand — Solara",
        "description": "Premium adaptogen supplement line for busy professionals",
        "prompt": textwrap.dedent("""\
            Create a marketing campaign for Solara — a premium adaptogen
            supplement line targeting busy professionals who want sustained
            energy without caffeine crashes.

            Research current wellness supplement market trends and competitor
            positioning, then generate content that feels elevated and
            aspirational while being grounded in science.
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
        "channels": ["landing_page", "email_sequence", "social_media", "ad_copy"],
    },
    "dev_tool": {
        "title": "Developer Tool Launch — VectorForge",
        "description": "Open-source vector database for AI agent memory",
        "prompt": textwrap.dedent("""\
            Create a marketing campaign for VectorForge — an open-source
            vector database optimized for AI agent memory and retrieval.

            Research current vector database benchmarks and trends, then
            generate developer-focused content with performance data.
            Target AI/ML engineers building LLM-powered applications.
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


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _header(title: str) -> None:
    """Print a boxed header."""
    width = max(len(title) + 4, 60)
    print()
    print(f"┌{'─' * (width - 2)}┐")
    print(f"│{title:^{width - 2}}│")
    print(f"└{'─' * (width - 2)}┘")
    print()


def _status(label: str, value: str, ok: bool = True) -> None:
    """Print a status line with a check/cross indicator."""
    icon = "OK" if ok else "MISSING"
    print(f"  [{icon:>7}]  {label:<24} {value}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Environment Check
# ═══════════════════════════════════════════════════════════════════════════

def check_environment() -> dict[str, bool]:
    """Validate environment variables and print status."""
    _header("Environment Check")

    keys = {
        "ANTHROPIC_API_KEY": {
            "required": True,
            "purpose": "Claude API (agent reasoning)",
        },
        "TAVILY_API_KEY": {
            "required": False,
            "purpose": "Tavily (web search & research)",
        },
        "OPENAI_API_KEY": {
            "required": False,
            "purpose": "OpenAI (image generation)",
        },
    }

    results = {}
    for key, info in keys.items():
        value = os.environ.get(key, "")
        is_set = bool(value)
        results[key] = is_set

        if is_set:
            masked = value[:8] + "..." + value[-4:] if len(value) > 16 else "***"
            _status(key, masked, ok=True)
        else:
            tag = "REQUIRED" if info["required"] else "optional"
            _status(key, f"not set ({tag})", ok=not info["required"])

    print()
    print("  Purpose:")
    for key, info in keys.items():
        req = "*" if info["required"] else " "
        print(f"    {req} {key:<24} {info['purpose']}")
    print()
    print("  * = required")
    print()

    # Optional env vars
    output_dir = os.environ.get("XENA_OUTPUT_DIR", "./marketing_output")
    print(f"  Output directory: {output_dir}")
    print()

    # Summary
    all_required_set = all(
        results[k] for k, v in keys.items() if v["required"]
    )
    if all_required_set:
        print("  Ready to run campaigns.")
    else:
        print("  WARNING: Required API keys are missing.")
        print("  Set them in your shell before running:")
        print()
        print("    export ANTHROPIC_API_KEY='sk-ant-...'")
        print("    export TAVILY_API_KEY='tvly-...'")
        print("    export OPENAI_API_KEY='sk-...'")
        print()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 2. MCP Tool Discovery
# ═══════════════════════════════════════════════════════════════════════════

async def discover_mcp_tools() -> list[dict]:
    """Connect to the MCP server and list all available tools."""
    _header("MCP Tool Discovery")

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("  ERROR: 'mcp' package not installed.")
        print("  Install with: pip install 'mcp>=1.9.0'")
        return []

    server_script = str(_this_dir / "mcp_server.py")
    server_params = StdioServerParameters(
        command=sys.executable, args=[server_script]
    )

    print(f"  Server: {server_script}")
    print(f"  Connecting...")
    print()

    try:
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

                    tools_info.append({
                        "name": tool.name,
                        "description": desc[:100],
                        "parameters": params,
                        "schema_bytes": schema_bytes,
                    })

                    print(f"  {tool.name}")
                    print(f"    {desc[:80]}")
                    print(f"    Parameters: {', '.join(params)}")
                    print(f"    Schema: {schema_bytes} bytes (~{schema_bytes // 4} tokens)")
                    print()

                print(f"  Total: {len(tools_info)} tools, "
                      f"{total_bytes:,} bytes (~{total_bytes // 4:,} tokens)")
                print()
                return tools_info

    except Exception as exc:
        print(f"  ERROR: Failed to connect to MCP server")
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("  Troubleshooting:")
        print("    1. Ensure dependencies are installed: pip install -e .")
        print("    2. Verify the server runs: python mcp_server.py")
        print()
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 3. Configuration Preview
# ═══════════════════════════════════════════════════════════════════════════

def preview_config(config: Config, prompt: str) -> None:
    """Display the resolved configuration and system prompt preview."""
    _header("Configuration Preview")

    print(f"  Product:      {config.product.name}")
    if config.product.tagline:
        print(f"  Tagline:      {config.product.tagline}")
    print(f"  Brand Voice:  {config.brand_voice_preset}")
    print(f"  Channels:     {', '.join(config.channels)}")
    print(f"  Model:        {config.model}")
    print(f"  Thinking:     {config.thinking_preset} "
          f"({config.resolved_thinking_budget:,} tokens)")
    print(f"  Images:       {'enabled' if config.generate_images else 'disabled'}")
    print(f"  Output:       {config.output_dir}")
    print()

    # Warnings
    warnings = config.validate()
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")
        print()

    # Brand voice details
    voice = config.brand_voice
    print(f"  Brand Voice Details ({config.brand_voice_preset}):")
    print(f"    Tone:        {voice.get('tone', '')}")
    print(f"    Personality: {voice.get('personality', '')}")
    print()

    # Prompt preview
    print("  Campaign Brief:")
    for line in prompt.strip().splitlines():
        print(f"    {line.strip()}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Campaign Runner with Error Handling
# ═══════════════════════════════════════════════════════════════════════════

async def run_campaign(
    prompt: str,
    product: ProductInfo | None = None,
    voice: str = "professional",
    channels: list[str] | None = None,
    thinking: str = "balanced",
    no_images: bool = False,
    output_dir: str | None = None,
    html_output: bool = False,
) -> dict | None:
    """Run a full campaign with comprehensive error handling."""

    # ── Build config ──
    overrides: dict = {}
    if product:
        overrides["product"] = product
    if voice:
        overrides["brand_voice_preset"] = voice
    if channels:
        overrides["channels"] = channels
    if thinking:
        overrides["thinking_preset"] = thinking
    if no_images:
        overrides["generate_images"] = False
    if output_dir:
        overrides["output_dir"] = output_dir

    config = Config.from_env(**overrides)

    # ── Validate ──
    warnings = config.validate()
    if not config.anthropic_api_key:
        _header("Error: Missing ANTHROPIC_API_KEY")
        print("  The Anthropic API key is required to run campaigns.")
        print("  Set it in your environment:")
        print()
        print("    export ANTHROPIC_API_KEY='sk-ant-api03-...'")
        print()
        print("  Get a key at: https://console.anthropic.com/settings/keys")
        print()
        return None

    for w in warnings:
        print(f"  [warning] {w}", file=sys.stderr)

    # ── Preview config ──
    preview_config(config, prompt)

    # ── Run ──
    _header("Running Campaign")

    try:
        agent = MCPXenaMarketingAgent(config=config)
        result = await agent.run(prompt)
    except KeyboardInterrupt:
        print("\n  Campaign interrupted by user.")
        return None
    except Exception as exc:
        _header("Campaign Failed")
        print(f"  Error: {type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        print()
        print("  Common issues:")
        print("    - Invalid API key: check ANTHROPIC_API_KEY is correct")
        print("    - Rate limit: wait a moment and try again")
        print("    - Network error: check your internet connection")
        print("    - MCP server: ensure dependencies are installed")
        print()
        return None

    # ── Build result dict ──
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

    # ── Save result JSON ──
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "demo_result.json"
    json_path.write_text(json.dumps(result_dict, indent=2, default=str))

    # ── Generate HTML viewer ──
    html_path_str = None
    if html_output:
        html_path = str(out_dir / "campaign_viewer.html")
        html_path_str = generate_html_viewer(result_dict, html_path)

    # ── Print summary ──
    _header("Campaign Complete")

    print(f"  Product:          {config.product.name}")
    print(f"  Brand Voice:      {config.brand_voice_preset}")
    print(f"  Duration:         {result.duration_seconds:.1f}s")
    print(f"  Tokens:           {result.input_tokens:,} in / {result.output_tokens:,} out")
    print()
    print(f"  Market searches:  {result.search_count}")
    print(f"  Pages fetched:    {result.fetch_count}")
    print(f"  Analysis phases:  {result.analyze_count}")
    print(f"  Content drafted:  {result.draft_count}")
    print(f"  Images generated: {result.image_count}")
    print(f"  Total tool calls: {len(result.tool_calls)}")

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

    print()
    print("  Output files:")
    if result.campaign_path:
        print(f"    Campaign:  {result.campaign_path}")
    print(f"    JSON:      {json_path.resolve()}")
    if html_path_str:
        print(f"    Viewer:    {html_path_str}")
    print()

    return result_dict


# ═══════════════════════════════════════════════════════════════════════════
# 5. Programmatic Usage Example
# ═══════════════════════════════════════════════════════════════════════════

async def programmatic_example() -> None:
    """Show how to use the agent programmatically from Python code."""
    _header("Programmatic Usage Example")

    print("  The following Python code shows how to use the agent as a library:")
    print()
    print(textwrap.indent(textwrap.dedent("""\
        import asyncio
        from mcp_xena_marketing_agent import MCPXenaMarketingAgent, Config, ProductInfo

        async def main():
            # Configure the agent
            config = Config.from_env(
                product=ProductInfo(
                    name="MyProduct",
                    tagline="AI for everyone",
                    description="An AI-powered productivity tool",
                    target_audience="Knowledge workers",
                    key_features=["Smart automation", "Natural language"],
                    competitors=["Notion", "Coda"],
                ),
                brand_voice_preset="startup",
                channels=["landing_page", "social_media"],
                thinking_preset="balanced",
                generate_images=False,  # set True if OPENAI_API_KEY is set
            )

            # Validate configuration
            warnings = config.validate()
            for w in warnings:
                print(f"Warning: {w}")

            # Run the agent
            agent = MCPXenaMarketingAgent(config=config)
            result = await agent.run("Create a launch campaign for MyProduct")

            # Access results
            print(f"Campaign saved to: {result.campaign_path}")
            print(f"Content pieces: {result.draft_count}")
            print(f"Duration: {result.duration_seconds:.1f}s")

            for piece in result.content_pieces:
                print(f"  [{piece['channel']}] {piece['title']}")

        asyncio.run(main())
    """), "    "))
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="demo",
        description="MCP Xena Marketing Agent — Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python demo.py --check                          # Validate environment
              python demo.py                                  # Run default demo (saas_launch)
              python demo.py --scenario wellness_brand        # Run a specific scenario
              python demo.py --scenario dev_tool --html       # With HTML viewer
              python demo.py --prompt "Campaign for X" \\
                             --voice startup --no-images      # Custom campaign
              python demo.py --example                        # Show programmatic usage

            Scenarios:     saas_launch, wellness_brand, dev_tool
            Brand voices:  professional, startup, luxury, technical
            Thinking:      minimal (4K), balanced (10K), deep (20K)

            Environment Variables:
              ANTHROPIC_API_KEY   (required)  Claude API key
              TAVILY_API_KEY      (optional)  Tavily web search key
              OPENAI_API_KEY      (optional)  OpenAI image generation key
              XENA_OUTPUT_DIR     (optional)  Output directory
        """),
    )

    parser.add_argument(
        "--check", action="store_true",
        help="Check environment and discover MCP tools (no API keys needed)",
    )
    parser.add_argument(
        "--scenario", "-s", choices=list(SCENARIOS.keys()),
        help="Run a pre-built demo scenario",
    )
    parser.add_argument(
        "--prompt", "-p", type=str,
        help="Custom campaign brief",
    )
    parser.add_argument(
        "--voice", "-v",
        choices=["professional", "startup", "luxury", "technical"],
        help="Brand voice preset (default: from scenario or 'professional')",
    )
    parser.add_argument(
        "--channels", "-c", nargs="+",
        help="Marketing channels to target",
    )
    parser.add_argument(
        "--thinking", "-t",
        choices=["minimal", "balanced", "deep"], default="balanced",
        help="Extended thinking budget (default: balanced)",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Disable image generation (skip OPENAI_API_KEY requirement)",
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Generate an HTML campaign viewer",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output directory (default: ./marketing_output)",
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Print programmatic usage example and exit",
    )

    args = parser.parse_args()

    # ── Programmatic example ──
    if args.example:
        asyncio.run(programmatic_example())
        return

    # ── Environment check + tool discovery ──
    if args.check:
        check_environment()
        asyncio.run(discover_mcp_tools())
        return

    # ── Resolve scenario / prompt ──
    if args.scenario:
        scenario = SCENARIOS[args.scenario]
        prompt = scenario["prompt"]
        product = scenario.get("product")
        voice = args.voice or scenario.get("voice", "professional")
        channels = args.channels or scenario.get("channels")
        _header(f"Demo: {scenario['title']}")
        print(f"  {scenario['description']}")
        print()
    elif args.prompt:
        prompt = args.prompt
        product = None
        voice = args.voice or "professional"
        channels = args.channels
    else:
        # Default: run saas_launch
        scenario = SCENARIOS["saas_launch"]
        prompt = scenario["prompt"]
        product = scenario.get("product")
        voice = args.voice or scenario.get("voice", "professional")
        channels = args.channels or scenario.get("channels")
        _header(f"Demo: {scenario['title']} (default)")
        print(f"  {scenario['description']}")
        print(f"  Tip: use --scenario <name> to pick a different demo")
        print(f"  Tip: use --check to validate your environment first")
        print()

    # ── Run campaign ──
    result = asyncio.run(run_campaign(
        prompt=prompt,
        product=product,
        voice=voice,
        channels=channels,
        thinking=args.thinking,
        no_images=args.no_images,
        output_dir=args.output,
        html_output=args.html,
    ))

    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
