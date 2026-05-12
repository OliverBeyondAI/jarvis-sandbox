"""
Xena Marketing Agent — Core agent powered by the Anthropic Managed Agents API.

Uses the beta managed-agents API (client.beta.agents / client.beta.sessions)
for a server-managed agentic loop with custom tool execution, replacing the
manual Messages API tool-use loop.

Workflow phases:
  1. Market Research     — Web search for trends, competitors, audience insights
  2. Competitor Analysis — Deep-dive into competitor positioning and messaging
  3. Messaging Strategy  — Develop key messages, value props, and positioning
  4. Content Generation  — Draft content across multiple marketing channels
  5. Campaign Assembly   — Compile and save the complete campaign document
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic

from .config import Config
from .tools import ALL_TOOLS, execute_tool, set_output_dir


# ---------------------------------------------------------------------------
# System Prompt Builder
# ---------------------------------------------------------------------------


def build_system_prompt(config: Config) -> str:
    """Build the full system prompt with brand voice and product context."""

    brand_block = config.brand_voice_prompt_block()
    product_block = config.product.to_prompt_block()
    channels_str = ", ".join(config.channels)

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
        5. Compile everything into a polished campaign document

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

        Channel-specific guidance:
        - **Landing Page**: Hero headline + subhead + 3 value prop sections + CTA
        - **Email Sequence**: 3-email nurture sequence (intro, value, convert)
        - **Social Media**: 5 posts across platforms (LinkedIn, Twitter/X, etc.)
        - **Blog Post**: 800-1200 word thought leadership article
        - **Ad Copy**: 3 variations (headline + body + CTA) for paid channels
        - **Press Release**: Standard press release format
        - **Product Description**: Concise, benefit-led product copy

        ### Phase 5: Campaign Assembly
        - Compile all content into a single, polished campaign document
        - Include the messaging strategy summary at the top
        - Organize by channel with clear section headers
        - Save using save_campaign with the complete markdown

        ## Quality Standards

        - **Brand consistency**: Every piece must match the brand voice guidelines
        - **Research-backed**: Messaging should reflect actual market insights
        - **Differentiated**: Content must stand apart from competitor messaging
        - **Actionable**: Every piece includes a clear call-to-action
        - **Channel-native**: Content feels natural for its target channel
        - **Audience-aware**: Language and framing match the target audience

        ## Rules

        - Execute AT LEAST 3 market research searches
        - Deep-dive into AT LEAST 1 competitor page using fetch_url
        - Use analyze_market at least twice (after research and after competitor analysis)
        - Use draft_content for EVERY content piece before compiling
        - ALWAYS call save_campaign with the complete markdown as the final step
        - Do NOT ask the user questions — you have full autonomy
        - Maintain the brand voice consistently across all content
        - Include specific, concrete details from your research in the content
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
    search_count: int = 0
    fetch_count: int = 0
    analyze_count: int = 0
    draft_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Managed Agent Runner
# ---------------------------------------------------------------------------


def _run_managed_agent_loop(
    client: anthropic.Anthropic,
    session_id: str,
    result: AgentResult,
    config: Config,
) -> None:
    """Execute the managed-agent event loop: stream events, handle custom tool calls."""

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
                            result.text += block.text + "\n"
                            snippet = block.text[:200]
                            if len(block.text) > 200:
                                snippet += "..."
                            _log(f"  [message] {snippet}")

                elif event.type == "agent.custom_tool_use":
                    tool_name = event.name
                    tool_input = dict(event.input)
                    tool_use_id = event.id
                    input_summary = _summarize_input(tool_name, tool_input)
                    _log(f"  -> {tool_name}({input_summary})")

                    result.tool_calls.append(
                        {"name": tool_name, "input": tool_input}
                    )
                    _track_tool_usage(result, tool_name, tool_input)

                    pending_tool_calls.append({
                        "id": tool_use_id,
                        "name": tool_name,
                        "input": tool_input,
                    })

                elif event.type == "span.model_request_end":
                    if hasattr(event, "model_usage") and event.model_usage:
                        usage = event.model_usage
                        if hasattr(usage, "input_tokens"):
                            result.input_tokens += usage.input_tokens or 0
                        if hasattr(usage, "output_tokens"):
                            result.output_tokens += usage.output_tokens or 0

                elif event.type == "session.status_idle":
                    stop_type = getattr(event.stop_reason, "type", "unknown")
                    if stop_type == "end_turn":
                        _log("Agent finished (end_turn).")
                        return
                    elif stop_type == "requires_action":
                        break  # exit stream, handle tool calls below
                    else:
                        _log(f"Agent stopped: {stop_type}")
                        return

                elif event.type == "session.error":
                    error_msg = getattr(event, "error", "Unknown error")
                    _log(f"  [error] {error_msg}")
                    return

                elif event.type == "session.status_terminated":
                    _log("Session terminated.")
                    return

        # Execute pending tool calls and send results back
        if not pending_tool_calls:
            return

        tool_result_events = []
        for tc in pending_tool_calls:
            result_str = execute_tool(tc["name"], tc["input"])

            # Collect insights and content from results
            _collect_from_tool(result, tc["name"], tc["input"], result_str)

            tool_result_events.append({
                "type": "user.custom_tool_result",
                "custom_tool_use_id": tc["id"],
                "content": [{"type": "text", "text": result_str}],
            })

        client.beta.sessions.events.send(
            session_id=session_id,
            events=tool_result_events,
        )


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


def _collect_from_tool(
    result: AgentResult,
    tool_name: str,
    tool_input: dict[str, Any],
    result_str: str,
) -> None:
    """Collect insights, content pieces, and campaign path from tool results."""
    if tool_name == "analyze_market":
        try:
            data = json.loads(result_str)
            for insight in data.get("insights", []):
                result.insights.append(insight)
        except (json.JSONDecodeError, KeyError):
            pass

    elif tool_name == "draft_content":
        result.content_pieces.append({
            "channel": tool_input.get("channel", ""),
            "title": tool_input.get("title", ""),
            "cta": tool_input.get("cta", ""),
        })

    elif tool_name == "save_campaign":
        try:
            save_result = json.loads(result_str)
            if save_result.get("saved"):
                result.campaign_path = save_result.get("path", "")
        except json.JSONDecodeError:
            pass


# ---------------------------------------------------------------------------
# Xena Marketing Agent
# ---------------------------------------------------------------------------


class XenaMarketingAgent:
    """
    Autonomous marketing content agent powered by Claude Opus 4.7.

    Uses the Anthropic beta managed-agents API for server-managed agentic
    execution with custom tool dispatch.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy-load Anthropic client."""
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self.config.anthropic_api_key:
                kwargs["api_key"] = self.config.anthropic_api_key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def run(self, prompt: str) -> AgentResult:
        """
        Execute a full autonomous marketing campaign generation cycle.

        Creates a managed agent with custom tools, opens a session, sends
        the marketing brief, and processes events until the agent completes.

        Args:
            prompt: The marketing brief or campaign request.

        Returns:
            AgentResult with campaign content, file path, and metadata.
        """
        start_time = time.time()
        set_output_dir(self.config.output_dir)

        system_prompt = build_system_prompt(self.config)
        result = AgentResult()

        self._log_header(prompt)

        # Step 1: Create managed agent with custom tools
        _log("Creating managed agent...")
        agent = self.client.beta.agents.create(
            model=self.config.model,
            name="xena-marketing-agent",
            description="Autonomous marketing campaign generator",
            system=system_prompt,
            tools=ALL_TOOLS,
        )
        _log(f"  Agent created: {agent.id}")

        # Step 2: Create environment
        environment = self.client.beta.environments.create(
            name="xena-marketing-env",
        )
        _log(f"  Environment created: {environment.id}")

        # Step 3: Create session
        session = self.client.beta.sessions.create(
            agent={"agent_id": agent.id},
            environment_id=environment.id,
        )
        _log(f"  Session created: {session.id}")

        # Step 4: Send user message to kick off the agent
        _log("Sending marketing brief...\n")
        self.client.beta.sessions.events.send(
            session_id=session.id,
            events=[{
                "type": "user.message",
                "content": [{"type": "text", "text": prompt}],
            }],
        )

        # Step 5: Run the event loop — stream events, handle tool calls
        _run_managed_agent_loop(self.client, session.id, result, self.config)

        result.duration_seconds = time.time() - start_time

        # Clean up
        try:
            self.client.beta.sessions.delete(session_id=session.id)
            self.client.beta.environments.delete(environment_id=environment.id)
            self.client.beta.agents.archive(agent_id=agent.id)
        except Exception:
            pass  # best-effort cleanup

        self._log_summary(result)
        return result

    def _log_header(self, prompt: str) -> None:
        _log(f"Starting campaign: {prompt[:120]}...")
        _log(f"Model: {self.config.model}")
        _log(f"Brand voice: {self.config.brand_voice_preset}")
        _log(f"Product: {self.config.product.name}")
        _log(f"Channels: {', '.join(self.config.channels)}")
        _log(f"API: Managed Agents (beta)")
        _log("")

    @staticmethod
    def _log_summary(result: AgentResult) -> None:
        _log("")
        _log("=" * 60)
        _log("CAMPAIGN COMPLETE")
        _log(f"  Market searches:   {result.search_count}")
        _log(f"  Pages fetched:     {result.fetch_count}")
        _log(f"  Analysis phases:   {result.analyze_count}")
        _log(f"  Content drafted:   {result.draft_count}")
        _log(f"  Total tool calls:  {len(result.tool_calls)}")
        _log(f"  Tokens:            {result.input_tokens:,} in / {result.output_tokens:,} out")
        _log(f"  Duration:          {result.duration_seconds:.1f}s")
        if result.campaign_path:
            _log(f"  Campaign saved:    {result.campaign_path}")
        _log("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    return json.dumps(input_dict, default=str)[:60]


def _log(msg: str) -> None:
    """Print progress to stderr."""
    print(f"[xena] {msg}", file=sys.stderr)
