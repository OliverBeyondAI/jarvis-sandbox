"""
MCP Xena Marketing Agent — Core agent powered by Claude Agent SDK + MCP.

Combines three integration patterns:
  1. MCP client → connects to the marketing tools MCP server for tool discovery
     and execution (market research, content drafting, image generation)
  2. Claude Agent SDK → managed-agents API for server-managed agentic loop
     with automatic tool dispatch
  3. Local fallback → Messages API tool-use loop when managed-agents is unavailable

Workflow phases:
  1. Market Research     — Web search for trends, competitors, audience insights
  2. Competitor Analysis — Deep-dive into competitor positioning and messaging
  3. Messaging Strategy  — Develop key messages, value props, and positioning
  4. Content Generation  — Draft content across multiple marketing channels
  5. Image Generation    — Create AI visuals for key content pieces
  6. Campaign Assembly   — Compile and save the complete campaign document
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import Config
from .tools import ALL_TOOLS


# ---------------------------------------------------------------------------
# System Prompt Builder
# ---------------------------------------------------------------------------


def build_system_prompt(config: Config) -> str:
    """Build the full system prompt with brand voice and product context."""

    brand_block = config.brand_voice_prompt_block()
    product_block = config.product.to_prompt_block()
    channels_str = ", ".join(config.channels)

    image_guidance = ""
    if config.generate_images:
        image_guidance = textwrap.dedent("""\
            6. **generate_image** — Create social media-ready marketing images
               with embedded text and branding. Provide:
               - `concept`: the visual scene/mood (e.g. "gradient abstract background")
               - `headline`: primary text rendered on the image (3-8 words)
               - `tagline`: secondary text below the headline
               - `brand_name`: product name for corner branding
               - `platform`: target platform for correct dimensions
                 (instagram_post, instagram_story, twitter_post, linkedin_post,
                  facebook_ad, blog_header, landing_hero)
               - `color_palette`: brand hex colors (e.g. "#3B82F6, #1B2A4A")
               - `style`: natural, vivid, minimal, or editorial
        """)

    return textwrap.dedent(f"""\
        You are **Xena** — an expert AI marketing strategist and content creator.
        You autonomously research markets, analyze competitors, craft brand-aligned
        messaging, and generate polished marketing content across multiple channels.

        ## Your Mission

        Given a product/service and brand voice guidelines, you autonomously:
        1. Research the market landscape, competitors, and target audience
        2. Analyze competitor positioning and identify messaging opportunities
        3. Develop a messaging strategy with key value propositions
        4. Generate compelling marketing content for each requested channel
        5. Generate AI images for key marketing visuals
        6. Compile everything into a polished campaign document

        ## Brand Voice Guidelines

        {brand_block}

        ## Product / Service

        {product_block}

        ## Target Channels

        Generate content for: {channels_str}

        ## Tools Available

        1. **market_research** — Search the web for market intelligence, trends,
           competitor activity, and audience insights. Run MULTIPLE searches
           (minimum 3) to cover different angles.

        2. **fetch_url** — Read the full content of a web page. Use this to
           deep-dive into competitor websites, industry reports, or product pages.

        3. **analyze_market** — Record structured market analysis findings after
           each research phase. This tracks your insights and identifies gaps.

        4. **draft_content** — Record a drafted content piece for a specific
           channel. Use this for each piece of content you generate.

        5. **save_campaign** — Save the complete campaign document as markdown.
           Call this exactly ONCE when all content is finalized.

        {image_guidance}

        ## Marketing Protocol

        Follow this multi-phase process strictly:

        ### Phase 1: Market Research (minimum 3 searches)
        - Search for market trends in the product's category
        - Research the target audience — pain points, motivations, language
        - Investigate competitor positioning and messaging
        - Use analyze_market to record findings after searches

        ### Phase 2: Competitor Deep-Dive (minimum 1 full-page read)
        - Identify 1-2 key competitor websites or product pages
        - Use fetch_url to read their full content
        - Analyze their messaging: headlines, value props, CTAs, tone
        - Use analyze_market to record competitor intelligence

        ### Phase 3: Messaging Strategy
        - Synthesize research into a clear positioning statement
        - Define 3-5 key messages that differentiate the product
        - Identify the primary value proposition for each audience segment
        - Determine the emotional and rational appeals to use

        ### Phase 4: Content Generation
        For EACH requested channel, generate content that:
        - Adheres strictly to the brand voice guidelines
        - Incorporates insights from market research
        - Differentiates from competitor messaging
        - Includes a clear, compelling call-to-action
        - Is tailored to the channel's format and audience expectations

        Use draft_content for each piece of content you create.

        ### Phase 5: Visual Assets
        Generate 1-2 key marketing images using generate_image:
        - A hero image for the landing page (platform="landing_hero") with:
          - A punchy headline (3-8 words) from the messaging strategy
          - The product tagline as secondary text
          - Brand name for corner branding
          - Brand color palette as hex codes
        - A social media visual (platform="instagram_post" or "twitter_post") with:
          - An attention-grabbing headline
          - A CTA or value prop as the tagline
          - Style that matches the brand voice (minimal for professional, vivid for startup)
        Always provide headline, brand_name, and concept for every image.

        ### Phase 6: Campaign Assembly
        - Compile all content into a single, polished campaign document
        - Include the messaging strategy summary at the top
        - Organize by channel with clear section headers
        - Save using save_campaign with the complete markdown

        ## Rules

        - Execute AT LEAST 3 market research searches
        - Deep-dive into AT LEAST 1 competitor page using fetch_url
        - Use analyze_market at least twice
        - Use draft_content for EVERY content piece before compiling
        - ALWAYS call save_campaign as the final step
        - Do NOT ask the user questions — you have full autonomy
        - Maintain the brand voice consistently across all content
    """)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Structured output from an agent run."""

    text: str = ""
    campaign_path: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    insights: list[dict[str, Any]] = field(default_factory=list)
    content_pieces: list[dict[str, Any]] = field(default_factory=list)
    images_generated: list[dict[str, Any]] = field(default_factory=list)
    search_count: int = 0
    fetch_count: int = 0
    analyze_count: int = 0
    draft_count: int = 0
    image_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    mcp_tools_discovered: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# MCP Xena Marketing Agent
# ---------------------------------------------------------------------------


class MCPXenaMarketingAgent:
    """
    Autonomous marketing content agent powered by Claude + MCP.

    Supports two execution modes:
      1. MCP mode (default) — connects to the marketing tools MCP server,
         discovers tools dynamically, and executes them via MCP protocol
      2. Managed mode — uses the Anthropic managed-agents API with tools
         defined as custom tool schemas

    Both modes use the same system prompt and produce the same output.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """Lazy-load async Anthropic client."""
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self.config.anthropic_api_key:
                kwargs["api_key"] = self.config.anthropic_api_key
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    # -- Public API --------------------------------------------------------

    async def run(self, prompt: str) -> AgentResult:
        """
        Execute a full autonomous marketing campaign generation cycle.

        Connects to the MCP server, discovers tools, then runs the agentic
        loop until the campaign is complete.

        Args:
            prompt: The marketing brief or campaign request.

        Returns:
            AgentResult with campaign content, file path, and metadata.
        """
        start_time = time.time()
        result = AgentResult()
        system_prompt = build_system_prompt(self.config)

        self._log_header(prompt)

        env = dict(os.environ)
        env["XENA_OUTPUT_DIR"] = self.config.output_dir

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.config.mcp_server_script],
            env=env,
        )

        self._log("Connecting to MCP server...")
        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._log("  MCP server connected and initialized.")

                    # Discover tools from MCP server
                    tools_result = await session.list_tools()
                    mcp_tools = []
                    for tool in tools_result.tools:
                        mcp_tools.append({
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": tool.inputSchema,
                        })
                        result.mcp_tools_discovered.append(tool.name)

                    self._log(f"  Discovered {len(mcp_tools)} tools: "
                              f"{', '.join(result.mcp_tools_discovered)}")

                    # Run the agentic loop with MCP tool execution
                    await self._agentic_loop(
                        session, system_prompt, prompt, mcp_tools, result
                    )

        except Exception as exc:
            self._log(f"  MCP connection failed: {exc}")
            self._log("  Falling back to local agent loop...")
            await self._local_fallback_loop(system_prompt, prompt, result)

        result.duration_seconds = time.time() - start_time
        self._log_summary(result)
        return result

    # -- MCP agentic loop --------------------------------------------------

    async def _agentic_loop(
        self,
        session: ClientSession,
        system_prompt: str,
        prompt: str,
        tools: list[dict[str, Any]],
        result: AgentResult,
    ) -> None:
        """Run the Claude messages API loop with MCP tool execution."""

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        thinking_config = {
            "type": "enabled",
            "budget_tokens": self.config.resolved_thinking_budget,
        }

        self._log(f"\nStarting agentic loop (thinking budget: "
                  f"{self.config.resolved_thinking_budget} tokens)...")

        for turn in range(1, self.config.max_agent_turns + 1):
            self._log(f"\n  Turn {turn}/{self.config.max_agent_turns}")

            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                tools=tools,
                messages=messages,
                thinking=thinking_config,
            )

            # Track token usage
            if hasattr(response, "usage"):
                result.input_tokens += response.usage.input_tokens or 0
                result.output_tokens += response.usage.output_tokens or 0

            # Check for end_turn
            if response.stop_reason == "end_turn":
                result.text = self._extract_text(response)
                self._log("  Agent finished (end_turn).")
                return

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type == "thinking":
                    snippet = block.thinking[:100] if block.thinking else ""
                    self._log(f"    [thinking] {snippet}...")
                    continue

                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = dict(block.input)
                self._log(f"    -> {tool_name}({_summarize_input(tool_name, tool_input)})")

                # Track usage
                result.tool_calls.append({"name": tool_name, "input": tool_input})
                _track_tool_usage(result, tool_name, tool_input)

                # Execute via MCP
                mcp_result = await session.call_tool(tool_name, tool_input)
                result_text = ""
                for content_block in mcp_result.content:
                    if hasattr(content_block, "text"):
                        result_text += content_block.text

                # Collect metadata from results
                _collect_from_tool(result, tool_name, tool_input, result_text)

                self._log(f"       <- {result_text[:120]}...")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                result.text = self._extract_text(response)
                return

        self._log("  Agent reached maximum turns.")
        result.text = self._extract_text(response) or "Agent reached maximum turns."

    # -- Local fallback (no MCP) -------------------------------------------

    async def _local_fallback_loop(
        self,
        system_prompt: str,
        prompt: str,
        result: AgentResult,
    ) -> None:
        """Fallback agentic loop using local tool implementations."""
        # Import MCP server functions directly as a fallback
        from . import mcp_server

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        # Strip 'type' key from schemas for the messages API
        api_tools = [
            {k: v for k, v in tool.items() if k != "type"}
            for tool in ALL_TOOLS
        ]

        thinking_config = {
            "type": "enabled",
            "budget_tokens": self.config.resolved_thinking_budget,
        }

        self._log(f"\nStarting local fallback loop...")

        # Map tool names to MCP server functions
        tool_dispatch = {
            "market_research": mcp_server.market_research,
            "fetch_url": mcp_server.fetch_url,
            "analyze_market": mcp_server.analyze_market,
            "draft_content": mcp_server.draft_content,
            "generate_image": mcp_server.generate_image,
            "save_campaign": mcp_server.save_campaign,
        }

        for turn in range(1, self.config.max_agent_turns + 1):
            self._log(f"\n  Turn {turn}/{self.config.max_agent_turns}")

            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                tools=api_tools,
                messages=messages,
                thinking=thinking_config,
            )

            if hasattr(response, "usage"):
                result.input_tokens += response.usage.input_tokens or 0
                result.output_tokens += response.usage.output_tokens or 0

            if response.stop_reason == "end_turn":
                result.text = self._extract_text(response)
                self._log("  Agent finished.")
                return

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = dict(block.input)
                self._log(f"    -> {tool_name}({_summarize_input(tool_name, tool_input)})")

                result.tool_calls.append({"name": tool_name, "input": tool_input})
                _track_tool_usage(result, tool_name, tool_input)

                handler = tool_dispatch.get(tool_name)
                if handler:
                    result_text = handler(**tool_input)
                else:
                    result_text = json.dumps({"error": f"Unknown tool: {tool_name}"})

                _collect_from_tool(result, tool_name, tool_input, result_text)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                result.text = self._extract_text(response)
                return

        self._log("  Agent reached maximum turns.")

    # -- Logging -----------------------------------------------------------

    def _log_header(self, prompt: str) -> None:
        _log(f"Starting campaign: {prompt[:120]}...")
        _log(f"Model: {self.config.model}")
        _log(f"Thinking: {self.config.thinking_preset} "
             f"({self.config.resolved_thinking_budget} tokens)")
        _log(f"Brand voice: {self.config.brand_voice_preset}")
        _log(f"Product: {self.config.product.name}")
        _log(f"Channels: {', '.join(self.config.channels)}")
        _log(f"Images: {'enabled' if self.config.generate_images else 'disabled'}")
        _log(f"MCP server: {self.config.mcp_server_script}")
        _log("")

    @staticmethod
    def _log_summary(result: AgentResult) -> None:
        _log("")
        _log("=" * 60)
        _log("CAMPAIGN COMPLETE")
        _log(f"  MCP tools discovered: {', '.join(result.mcp_tools_discovered) or 'N/A'}")
        _log(f"  Market searches:      {result.search_count}")
        _log(f"  Pages fetched:        {result.fetch_count}")
        _log(f"  Analysis phases:      {result.analyze_count}")
        _log(f"  Content drafted:      {result.draft_count}")
        _log(f"  Images generated:     {result.image_count}")
        _log(f"  Total tool calls:     {len(result.tool_calls)}")
        _log(f"  Tokens:               {result.input_tokens:,} in / "
             f"{result.output_tokens:,} out")
        _log(f"  Duration:             {result.duration_seconds:.1f}s")
        if result.campaign_path:
            _log(f"  Campaign saved:       {result.campaign_path}")
        _log("=" * 60)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract all text content from a Claude response."""
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track_tool_usage(result: AgentResult, tool_name: str, tool_input: dict) -> None:
    """Track tool usage counts."""
    if tool_name == "market_research":
        result.search_count += 1
    elif tool_name == "fetch_url":
        result.fetch_count += 1
    elif tool_name == "analyze_market":
        result.analyze_count += 1
    elif tool_name == "draft_content":
        result.draft_count += 1
    elif tool_name == "generate_image":
        result.image_count += 1


def _collect_from_tool(
    result: AgentResult,
    tool_name: str,
    tool_input: dict[str, Any],
    result_str: str,
) -> None:
    """Collect insights, content pieces, images, and campaign path from tool results."""
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return

    if tool_name == "analyze_market":
        for insight in data.get("insights", []):
            result.insights.append(insight)

    elif tool_name == "draft_content":
        result.content_pieces.append({
            "channel": tool_input.get("channel", ""),
            "title": tool_input.get("title", ""),
            "cta": tool_input.get("cta", ""),
        })

    elif tool_name == "generate_image":
        if data.get("status") == "generated":
            result.images_generated.append({
                "channel": data.get("channel", "") or data.get("platform", ""),
                "file_path": data.get("file_path", ""),
                "concept": tool_input.get("concept", "")[:100],
                "headline": tool_input.get("headline", ""),
                "platform": data.get("platform", ""),
                "platform_label": data.get("platform_label", ""),
            })

    elif tool_name == "save_campaign":
        if data.get("saved"):
            result.campaign_path = data.get("path", "")


def _summarize_input(tool_name: str, input_dict: dict[str, Any]) -> str:
    """Create a short log summary of tool input."""
    if tool_name == "market_research":
        q = input_dict.get("query", "")
        return f'"{q[:60]}..."' if len(q) > 60 else f'"{q}"'
    if tool_name == "fetch_url":
        url = input_dict.get("url", "")
        return f'"{url[:60]}..."' if len(url) > 60 else f'"{url}"'
    if tool_name == "save_campaign":
        fn = input_dict.get("filename", "")
        size = len(input_dict.get("content", ""))
        return f'"{fn}" ({size:,} chars)'
    if tool_name == "draft_content":
        ch = input_dict.get("channel", "?")
        title = input_dict.get("title", "?")
        return f'{ch}: "{title[:40]}"'
    if tool_name == "analyze_market":
        phase = input_dict.get("phase", "?")
        count = len(input_dict.get("insights", []))
        return f'phase="{phase}", {count} insights'
    if tool_name == "generate_image":
        concept = input_dict.get("concept", "")
        headline = input_dict.get("headline", "")
        platform = input_dict.get("platform", "")
        parts = []
        if platform:
            parts.append(f"platform={platform}")
        if headline:
            parts.append(f'headline="{headline[:30]}"')
        parts.append(f'concept="{concept[:40]}"')
        return ", ".join(parts)
    return json.dumps(input_dict, default=str)[:60]


def _log(msg: str) -> None:
    """Print progress to stderr."""
    print(f"[mcp-xena] {msg}", file=sys.stderr)
