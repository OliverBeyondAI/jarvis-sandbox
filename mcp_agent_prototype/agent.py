#!/usr/bin/env python3
"""
MCP Agent — demonstrates deferred tool loading via MCP to save tokens.

Deferred tool loading pattern:
  1. On startup, connect to the MCP server and list all available tools.
  2. Pass only lightweight tool *summaries* (name + description, no schema) to
     Claude in the system prompt — no tool schemas in the API request.
  3. Claude responds with which tools it intends to use (a "planning" turn).
  4. Load only those tools' full schemas and re-issue the request so Claude can
     generate proper tool_use blocks with correct arguments.
  5. Execute the tool calls via MCP and feed results back to Claude.

This means the agent only pays the token cost of full tool schemas for tools
that are actually invoked, not for every tool the server exposes.

Usage:
    python agent.py                         # interactive mode
    python agent.py "find enterprise customers in tech"
    python agent.py --list-tools            # just show discovered tools
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7-20250501"
MAX_TOKENS = 4096
MCP_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolSummary:
    """Lightweight tool metadata — name + description, no schema."""
    name: str
    description: str

    def as_prompt_line(self) -> str:
        desc = self.description.split("\n")[0].strip()
        return f"- **{self.name}**: {desc}"


@dataclass
class ToolDetail:
    """Full tool definition suitable for the Claude messages API."""
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_claude_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    @property
    def schema_bytes(self) -> int:
        return len(json.dumps(self.input_schema))

    @property
    def schema_tokens_estimate(self) -> int:
        return self.schema_bytes // 4


@dataclass
class AgentResult:
    """Structured output from an agent run."""
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tools_discovered: list[str] = field(default_factory=list)
    tools_loaded: list[str] = field(default_factory=list)
    total_schema_tokens: int = 0
    tokens_saved_estimate: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# MCP Agent with deferred tool loading
# ---------------------------------------------------------------------------

class MCPAgent:
    """
    Agent that connects to an MCP server, discovers tools, and uses deferred
    tool loading to minimize token usage.

    The key insight: MCP servers can expose many tools, each with large JSON
    schemas. Sending all schemas to Claude on every request wastes tokens.
    Instead, we send a brief summary of available tools in the system prompt,
    and only attach the full schema for tools Claude actually requests.
    """

    def __init__(
        self,
        server_script: str = MCP_SERVER_SCRIPT,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        verbose: bool = True,
    ):
        self.server_script = server_script
        self.model = model
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.client = anthropic.AsyncAnthropic()

        # Populated during discovery
        self._summaries: dict[str, ToolSummary] = {}
        self._details: dict[str, ToolDetail] = {}

    # -- MCP connection & discovery ----------------------------------------

    async def run(self, prompt: str) -> AgentResult:
        """
        Full agent lifecycle: connect to MCP server, discover tools, run
        the agentic loop with deferred loading, and return the result.
        """
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
        )

        self._log("Connecting to MCP server...")
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._log("  Connected and initialized.")

                # Step 1: Discover tools (lightweight)
                await self._discover_tools(session)

                # Step 2: Run the agentic loop with deferred loading
                return await self._agentic_loop(session, prompt)

    async def _discover_tools(self, session: ClientSession) -> None:
        """List all tools from the MCP server and store summaries + details."""
        result = await session.list_tools()
        self._summaries.clear()
        self._details.clear()

        for tool in result.tools:
            summary = ToolSummary(
                name=tool.name,
                description=tool.description or "(no description)",
            )
            self._summaries[tool.name] = summary

            detail = ToolDetail(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            self._details[tool.name] = detail

        self._log(f"  Discovered {len(self._summaries)} tools:")
        for name in self._summaries:
            d = self._details[name]
            self._log(f"    • {name} (schema: {d.schema_bytes} bytes — deferred)")

    # -- System prompt with lightweight tool summaries ---------------------

    def _build_system_prompt(self, *, planning: bool) -> str:
        """
        Build the system prompt.

        When planning=True (first turn, no schemas loaded yet), instruct Claude
        to declare which tools it plans to use so we can load only those schemas.

        When planning=False (schemas loaded), let Claude proceed normally.
        """
        tool_lines = "\n".join(s.as_prompt_line() for s in self._summaries.values())

        if planning:
            return textwrap.dedent(f"""\
                You are a helpful AI agent connected to an enterprise MCP server.
                You have access to the following tools via MCP:

                {tool_lines}

                IMPORTANT: You do not yet have the full tool schemas loaded. Before
                using any tools, you MUST first respond with a brief plan stating
                which tools you intend to use and why. Format your plan like this:

                TOOLS_NEEDED: tool_name1, tool_name2

                Include the TOOLS_NEEDED line with comma-separated tool names, then
                briefly explain your approach. The system will then load the required
                schemas so you can make proper tool calls on the next turn.
            """)
        else:
            return textwrap.dedent(f"""\
                You are a helpful AI agent connected to an enterprise MCP server.
                You have access to the following tools via MCP:

                {tool_lines}

                When you need to use a tool, call it by name with appropriate arguments.
                The system will handle execution automatically.

                Always explain what you're doing and present results clearly.
            """)

    # -- Agentic loop with deferred tool loading ---------------------------

    async def _agentic_loop(
        self, session: ClientSession, prompt: str
    ) -> AgentResult:
        """
        Run the Claude messages API loop with true deferred tool loading.

        Phase 1 (planning): Send the user prompt with tool summaries in the
        system prompt but NO tool schemas in the API tools parameter. Claude
        responds with which tools it plans to use.

        Phase 2 (schema loading): Parse Claude's response for tool names,
        load only those schemas.

        Phase 3 (execution): Re-issue the request with loaded schemas so
        Claude can generate proper tool_use blocks. Execute tools via MCP
        and feed results back in the standard agentic loop.
        """
        # Calculate total schema size across all tools
        total_schema_tokens = sum(
            d.schema_tokens_estimate for d in self._details.values()
        )

        result = AgentResult(
            tools_discovered=list(self._summaries.keys()),
            total_schema_tokens=total_schema_tokens,
        )

        loaded_tools: set[str] = set()
        max_turns = 15

        self._log(f"\nStarting agentic loop (deferred tool loading)...")
        self._log(f"  Total schema tokens if all loaded: ~{total_schema_tokens}")

        # --- Phase 1: Planning turn (no tool schemas sent) ----------------
        self._log(f"\n  Phase 1: Planning (no tool schemas sent)")

        planning_response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._build_system_prompt(planning=True),
            messages=[{"role": "user", "content": prompt}],
        )

        planning_text = self._extract_text(planning_response)
        self._log(f"    Claude's plan: {planning_text[:200]}...")

        # Parse which tools Claude wants
        requested_tools = self._parse_tool_requests(planning_text)
        if not requested_tools:
            # Claude didn't request any tools — return the text response as-is
            self._log(f"    No tools requested — returning text response.")
            result.text = planning_text
            result.tokens_saved_estimate = total_schema_tokens
            return result

        # --- Phase 2: Load only the requested schemas --------------------
        self._log(f"\n  Phase 2: Loading schemas for requested tools only")
        for tool_name in requested_tools:
            if tool_name in self._details:
                loaded_tools.add(tool_name)
                d = self._details[tool_name]
                self._log(f"    ↓ Loaded schema for '{tool_name}' ({d.schema_bytes} bytes)")
            else:
                self._log(f"    ✗ Unknown tool requested: '{tool_name}' — skipping")

        loaded_tokens = sum(
            self._details[name].schema_tokens_estimate
            for name in loaded_tools
        )
        saved_tokens = total_schema_tokens - loaded_tokens
        self._log(f"    Loaded {len(loaded_tools)}/{len(self._details)} tool schemas")
        self._log(f"    Tokens spent on schemas: ~{loaded_tokens}")
        self._log(f"    Tokens saved by deferral: ~{saved_tokens}")

        # --- Phase 3: Execution loop (with loaded schemas) ----------------
        self._log(f"\n  Phase 3: Execution loop")

        claude_tools = [
            self._details[name].as_claude_tool()
            for name in loaded_tools
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]
        system_prompt = self._build_system_prompt(planning=False)

        for turn in range(1, max_turns + 1):
            self._log(f"    Turn {turn}/{max_turns}")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=claude_tools,
                messages=messages,
            )

            # Check for end_turn
            if response.stop_reason == "end_turn":
                result.text = self._extract_text(response)
                result.tools_loaded = list(loaded_tools)
                result.tokens_saved_estimate = saved_tokens
                not_loaded = set(self._summaries.keys()) - loaded_tools
                if not_loaded:
                    self._log(f"\n    Tools never loaded (schemas saved): {', '.join(not_loaded)}")
                self._log("    Agent finished.\n")
                return result

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = dict(block.input)

                if tool_name not in self._details:
                    self._log(f"      ✗ Unknown tool: {tool_name}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    })
                    continue

                # If Claude requests a tool we didn't pre-load, load it now
                if tool_name not in loaded_tools:
                    d = self._details[tool_name]
                    self._log(f"      ↓ Late-loading schema for '{tool_name}' ({d.schema_bytes} bytes)")
                    loaded_tools.add(tool_name)
                    claude_tools.append(d.as_claude_tool())
                    loaded_tokens += d.schema_tokens_estimate
                    saved_tokens = total_schema_tokens - loaded_tokens

                self._log(f"      → Calling {tool_name}({self._summarize_input(tool_input)})")

                # Execute via MCP
                mcp_result = await session.call_tool(tool_name, tool_input)
                result_text = ""
                for content_block in mcp_result.content:
                    if hasattr(content_block, "text"):
                        result_text += content_block.text

                result.tool_calls.append({
                    "name": tool_name,
                    "input": tool_input,
                    "result_preview": result_text[:200],
                })

                self._log(f"        ← Result: {result_text[:120]}...")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                result.text = self._extract_text(response)
                result.tools_loaded = list(loaded_tools)
                result.tokens_saved_estimate = saved_tokens
                return result

        self._log("    Reached maximum turns.")
        result.text = self._extract_text(response) or "Agent reached maximum turns."
        result.tools_loaded = list(loaded_tools)
        result.tokens_saved_estimate = saved_tokens
        return result

    def _parse_tool_requests(self, text: str) -> list[str]:
        """
        Parse Claude's planning response for tool names it wants to use.

        Looks for a TOOLS_NEEDED: line, then falls back to matching known tool
        names mentioned anywhere in the text.
        """
        # Primary: look for explicit TOOLS_NEEDED line
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("TOOLS_NEEDED:"):
                names_part = stripped.split(":", 1)[1].strip()
                names = [n.strip() for n in names_part.split(",") if n.strip()]
                # Validate against known tools
                valid = [n for n in names if n in self._details]
                if valid:
                    self._log(f"    Parsed TOOLS_NEEDED: {', '.join(valid)}")
                    return valid

        # Fallback: look for known tool names mentioned in text
        found = []
        text_lower = text.lower()
        for name in self._details:
            if name.lower() in text_lower:
                found.append(name)
        if found:
            self._log(f"    Inferred tools from text: {', '.join(found)}")
        return found

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _summarize_input(input_dict: dict[str, Any]) -> str:
        if "query" in input_dict:
            q = input_dict["query"]
            return f'query="{q[:50]}..."' if len(q) > 50 else f'query="{q}"'
        return json.dumps(input_dict, default=str)[:80]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

async def demo_list_tools() -> None:
    """Connect to the MCP server and list discovered tools."""
    agent = MCPAgent(verbose=True)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[agent.server_script],
    )

    print("=" * 60)
    print("MCP Tool Discovery")
    print("=" * 60)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await agent._discover_tools(session)

            print(f"\nDiscovered {len(agent._summaries)} tools:\n")
            for name, detail in agent._details.items():
                schema_json = json.dumps(detail.input_schema, indent=2)
                print(f"  Tool: {name}")
                print(f"  Description: {detail.description.split(chr(10))[0]}")
                print(f"  Schema size: {detail.schema_bytes} bytes (~{detail.schema_tokens_estimate} tokens)")
                print(f"  Schema:\n{textwrap.indent(schema_json, '    ')}")
                print()

            total_bytes = sum(d.schema_bytes for d in agent._details.values())
            total_tokens = sum(d.schema_tokens_estimate for d in agent._details.values())
            print(f"Total schema payload: {total_bytes} bytes (~{total_tokens} tokens)")
            print(f"With deferred loading, these tokens are only spent when a tool is actually used.")


async def demo_agent(prompt: str) -> None:
    """Run the agent with a prompt and display results."""
    print("=" * 60)
    print("MCP Agent with Deferred Tool Loading")
    print("=" * 60)
    print(f"\nPrompt: {prompt}\n")

    agent = MCPAgent(verbose=True)
    result = await agent.run(prompt)

    print("=" * 60)
    print("Result")
    print("=" * 60)
    print(f"\n{result.text}\n")
    print("-" * 60)
    print(f"Tools discovered: {', '.join(result.tools_discovered)}")
    print(f"Tools loaded:     {', '.join(result.tools_loaded) or '(none)'}")
    print(f"Tool calls made:  {len(result.tool_calls)}")
    print(f"Tokens saved (est): ~{result.tokens_saved_estimate} (schemas not loaded)")
    print(f"Total schema tokens: ~{result.total_schema_tokens} (if all loaded)")

    if result.tool_calls:
        print(f"\nTool call log:")
        for i, tc in enumerate(result.tool_calls, 1):
            print(f"  {i}. {tc['name']}({json.dumps(tc['input'], default=str)[:80]})")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if "--list-tools" in sys.argv:
        asyncio.run(demo_list_tools())
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        prompt = " ".join(sys.argv[1:])
        asyncio.run(demo_agent(prompt))
    else:
        # Interactive mode
        print("MCP Agent (deferred tool loading) — type 'quit' to exit\n")
        print("Example prompts:")
        print('  "Find all enterprise customers"')
        print('  "Search for technology companies and email a summary to boss@company.com"')
        print('  "What products are available under $200?"')
        print()

        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not prompt or prompt.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            asyncio.run(demo_agent(prompt))
            print()


if __name__ == "__main__":
    main()
