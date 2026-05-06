#!/usr/bin/env python3
"""
Research Agent — Autonomous research and synthesis with Claude Agent SDK.

A specialized agent that uses Tavily web search for discovery and a file system
tool for persisting research notes to the data/ directory. Built on Opus 4.7
with both managed-agents and local fallback runners.

Usage:
    python -m agents.research_agent "What are the latest advances in quantum computing?"
    python -m agents.research_agent --topic "AI regulation" --depth deep
    python -m agents.research_agent --topic "mRNA vaccines" --output ./data/mrna_research.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anthropic
import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7-20250501"
MAX_TOKENS = 8192
MAX_TURNS = 30

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autonomous research agent powered by Claude Opus 4.7. Your role
    is to conduct thorough, multi-step research on any topic and produce a
    well-structured synthesis.

    ## Tools Available

    1. **tavily_search** — Search the web for current information. Use multiple
       queries to cover different angles of the topic.
    2. **fetch_url** — Retrieve full content from a specific URL for deeper reading.
    3. **save_notes** — Save research notes, findings, or the final synthesis to
       the local file system (data/ directory). Use this to persist intermediate
       findings and the final report.

    ## Research Protocol

    Follow this structured approach:

    1. **Discovery Phase**: Issue 3-5 diverse search queries to map the landscape.
       Vary query phrasing to capture different perspectives and sources.

    2. **Deep Dive Phase**: Select the most promising sources and fetch their full
       content. Look for primary sources, expert opinions, and recent developments.

    3. **Synthesis Phase**: Organize findings into a coherent narrative. Identify
       key themes, areas of consensus, open questions, and emerging trends.

    4. **Output Phase**: Save the final synthesis using save_notes with a clear
       filename. The synthesis should include:
       - Executive summary (2-3 sentences)
       - Key findings (bulleted)
       - Detailed analysis (organized by theme)
       - Sources cited
       - Open questions / areas for further research

    ## Guidelines

    - Always cite sources with URLs.
    - Prefer recent sources (last 12 months when relevant).
    - Acknowledge uncertainty and conflicting information.
    - Save intermediate notes as you go (e.g., "notes_[topic]_discovery.md").
    - Save the final synthesis as "research_[topic]_[date].md".
    - Be thorough but concise — aim for actionable insights over exhaustive detail.
""")

BRIEF_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a research agent with web search, URL fetching, and file saving tools.
    Conduct focused research on the given topic: search for key information, read
    important sources, and save a concise summary to the data/ directory.
    Always cite sources. Save your output using save_notes.
""")


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic custom tool format)
# ---------------------------------------------------------------------------

TAVILY_SEARCH_TOOL: dict[str, Any] = {
    "name": "tavily_search",
    "type": "custom",
    "description": (
        "Search the web using Tavily for current information on any topic. "
        "Returns search results with titles, URLs, and content snippets. "
        "Use varied queries to get comprehensive coverage."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (1-10, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

FETCH_URL_TOOL: dict[str, Any] = {
    "name": "fetch_url",
    "type": "custom",
    "description": (
        "Fetch the text content of a web page at the given URL. "
        "Returns extracted text with HTML stripped. "
        "Use this to read full articles or documents found via search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch content from.",
            },
        },
        "required": ["url"],
    },
}

SAVE_NOTES_TOOL: dict[str, Any] = {
    "name": "save_notes",
    "type": "custom",
    "description": (
        "Save research notes or synthesis to the local file system (data/ directory). "
        "Use this to persist intermediate findings and the final research report. "
        "Files are saved as markdown in the project's data/ directory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": (
                    "Name of the file to save (e.g., 'research_quantum_2026-05-06.md'). "
                    "Will be saved inside the data/ directory."
                ),
            },
            "content": {
                "type": "string",
                "description": "The markdown content to write to the file.",
            },
            "append": {
                "type": "boolean",
                "description": "If true, append to existing file instead of overwriting.",
                "default": False,
            },
        },
        "required": ["filename", "content"],
    },
}

ALL_TOOLS: list[dict[str, Any]] = [TAVILY_SEARCH_TOOL, FETCH_URL_TOOL, SAVE_NOTES_TOOL]


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

async def tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web using the Tavily API."""
    def _sync_search() -> dict[str, Any]:
        try:
            from tavily import TavilyClient
            client = TavilyClient()  # Uses TAVILY_API_KEY env var
            results = client.search(query=query, max_results=min(max_results, 10))
            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0),
                    }
                    for r in results.get("results", [])
                ],
            }
        except ImportError:
            return {"query": query, "error": "tavily-python not installed. Run: pip install tavily-python"}
        except Exception as e:
            return {"query": query, "error": f"Search failed: {e}"}

    return await asyncio.to_thread(_sync_search)


async def fetch_url(url: str) -> dict[str, Any]:
    """Fetch and extract text content from a URL."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0,
            headers={
                "User-Agent": "ResearchAgent/1.0",
                "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            text = response.text
            if "text/html" in content_type:
                text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
                text = re.sub(r"\s+", " ", text).strip()

            max_chars = 50_000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Content truncated at 50,000 characters]"

            return {
                "url": url,
                "status": response.status_code,
                "content_type": content_type.split(";")[0].strip(),
                "content": text,
                "length": len(text),
            }
    except httpx.HTTPStatusError as e:
        return {"url": url, "error": f"HTTP {e.response.status_code}: {e}"}
    except httpx.TimeoutException:
        return {"url": url, "error": "Request timed out after 30 seconds"}
    except Exception as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}


async def save_notes(filename: str, content: str, append: bool = False) -> dict[str, Any]:
    """Save research notes to the data/ directory."""
    try:
        # Sanitize filename — prevent path traversal
        safe_name = Path(filename).name
        if not safe_name:
            return {"error": "Invalid filename"}

        filepath = DATA_DIR / safe_name
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        prefix = "\n\n---\n\n" if append and filepath.exists() else ""

        def _write():
            with open(filepath, mode, encoding="utf-8") as f:
                f.write(prefix + content)

        await asyncio.to_thread(_write)

        return {
            "saved": True,
            "path": str(filepath),
            "filename": safe_name,
            "size_bytes": filepath.stat().st_size,
            "mode": "appended" if append else "created",
        }
    except Exception as e:
        return {"error": f"Failed to save: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(name: str, input_dict: dict[str, Any]) -> str:
    """Execute a tool by name and return JSON string result."""
    try:
        if name == "tavily_search":
            result = await tavily_search(**input_dict)
        elif name == "fetch_url":
            result = await fetch_url(**input_dict)
        elif name == "save_notes":
            result = await save_notes(**input_dict)
        else:
            result = {"error": f"Unknown tool: {name}"}
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ResearchResult:
    """Structured output from a research agent run."""
    synthesis: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    files_saved: list[str] = field(default_factory=list)
    sources_consulted: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def summary(self) -> str:
        """One-line summary of the research run."""
        return (
            f"Research complete: {self.sources_consulted} sources consulted, "
            f"{len(self.files_saved)} file(s) saved, "
            f"{len(self.tool_calls)} tool calls made."
        )


# ---------------------------------------------------------------------------
# Managed Agent Runner (Claude Agent SDK)
# ---------------------------------------------------------------------------

class ManagedResearchRunner:
    """Runs the research agent via Claude Agent SDK managed-agents API."""

    def __init__(self, model: str = MODEL, system_prompt: str = SYSTEM_PROMPT, verbose: bool = True):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt
        self.verbose = verbose

    async def run(self, prompt: str) -> ResearchResult:
        """Execute a full research session via managed-agents."""
        agent = None
        environment = None
        result = ResearchResult()

        try:
            self._log("Creating managed research agent...")
            agent = self.client.beta.agents.create(
                model=self.model,
                name="research-agent",
                description="Autonomous research agent with web search and file system tools.",
                system=self.system_prompt,
                tools=ALL_TOOLS,
            )
            self._log(f"  Agent: {agent.id}")

            environment = self.client.beta.environments.create(name="research-env")
            self._log(f"  Environment: {environment.id}")

            session = self.client.beta.sessions.create(
                agent=agent.id,
                environment_id=environment.id,
            )
            self._log(f"  Session: {session.id}")

            self._log("Sending research prompt...")
            self.client.beta.sessions.events.send(
                session_id=session.id,
                events=[{"type": "user.message", "content": [{"type": "text", "text": prompt}]}],
            )

            final_text = ""
            for round_num in range(1, MAX_TURNS + 1):
                self._log(f"  Round {round_num}...")
                pending_tool_results = []

                events = self.client.beta.sessions.events.list(session_id=session.id, order="asc")
                for event in events.data:
                    if event.type == "agent.message":
                        for block in event.content:
                            if hasattr(block, "text"):
                                final_text = block.text

                    elif event.type == "agent.custom_tool_use":
                        tool_name = event.name
                        tool_input = dict(event.input)
                        self._log(f"    [{tool_name}] {_summarize(tool_input)}")

                        result.tool_calls.append({"name": tool_name, "input": tool_input})
                        tool_result = await execute_tool(tool_name, tool_input)

                        # Track files saved and sources
                        if tool_name == "save_notes":
                            parsed = json.loads(tool_result)
                            if parsed.get("saved"):
                                result.files_saved.append(parsed["path"])
                        elif tool_name in ("tavily_search", "fetch_url"):
                            result.sources_consulted += 1

                        pending_tool_results.append({
                            "type": "user.custom_tool_result",
                            "custom_tool_use_id": event.id,
                            "content": [{"type": "text", "text": tool_result}],
                        })

                    elif event.type == "session.status_idle":
                        stop = event.stop_reason
                        if hasattr(stop, "type") and stop.type == "end_turn":
                            self._log("  Research complete.")
                            result.synthesis = final_text
                            return result

                if pending_tool_results:
                    self._log(f"    Sending {len(pending_tool_results)} result(s)...")
                    self.client.beta.sessions.events.send(
                        session_id=session.id,
                        events=pending_tool_results,
                    )
                else:
                    if final_text:
                        result.synthesis = final_text
                        return result
                    break

            result.synthesis = final_text or "Research agent reached maximum rounds."
            return result

        finally:
            if agent:
                try:
                    self.client.beta.agents.archive(agent_id=agent.id)
                except Exception:
                    pass
            if environment:
                try:
                    self.client.beta.environments.delete(environment_id=environment.id)
                except Exception:
                    pass

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[research] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Local Agent Runner (Fallback)
# ---------------------------------------------------------------------------

class LocalResearchRunner:
    """Runs the research agent locally using AsyncAnthropic with tool loop."""

    def __init__(self, model: str = MODEL, system_prompt: str = SYSTEM_PROMPT, verbose: bool = True):
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.system_prompt = system_prompt
        self.verbose = verbose

    def _messages_api_tools(self) -> list[dict[str, Any]]:
        """Convert tool schemas for the messages API (strip 'type' key)."""
        return [{k: v for k, v in tool.items() if k != "type"} for tool in ALL_TOOLS]

    async def run(self, prompt: str) -> ResearchResult:
        """Run the local agentic loop."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        result = ResearchResult()

        self._log("Starting local research loop...")

        for turn in range(1, MAX_TURNS + 1):
            self._log(f"  Turn {turn}/{MAX_TURNS}")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=self._messages_api_tools(),
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                result.synthesis = _extract_text(response)
                self._log("  Research complete.")
                return result

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for block in assistant_content:
                if block.type == "tool_use":
                    self._log(f"    [{block.name}] {_summarize(block.input)}")
                    result.tool_calls.append({"name": block.name, "input": block.input})

                    tool_result = await execute_tool(block.name, block.input)

                    # Track saved files and sources
                    if block.name == "save_notes":
                        parsed = json.loads(tool_result)
                        if parsed.get("saved"):
                            result.files_saved.append(parsed["path"])
                    elif block.name in ("tavily_search", "fetch_url"):
                        result.sources_consulted += 1

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                result.synthesis = _extract_text(response)
                return result

        result.synthesis = "Research agent reached maximum turns."
        return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[research] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public API — ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Autonomous research agent with web search and file system tools.

    Uses Claude Opus 4.7 via the Claude Agent SDK (managed-agents API) with
    automatic fallback to a local async runner. Conducts multi-step research
    and saves structured synthesis to the data/ directory.

    Example:
        agent = ResearchAgent()
        result = await agent.research("What are the latest advances in quantum computing?")
        print(result.synthesis)
        print(result.files_saved)
    """

    def __init__(
        self,
        model: str = MODEL,
        depth: str = "deep",
        verbose: bool = True,
        use_managed: bool = True,
        output_dir: Optional[Path] = None,
    ):
        self.model = model
        self.depth = depth
        self.verbose = verbose
        self.use_managed = use_managed

        if output_dir:
            global DATA_DIR
            DATA_DIR = Path(output_dir)

        self.system_prompt = SYSTEM_PROMPT if depth == "deep" else BRIEF_SYSTEM_PROMPT

    async def research(self, topic: str) -> ResearchResult:
        """
        Conduct autonomous research on the given topic.

        Args:
            topic: The research question or topic to investigate.

        Returns:
            ResearchResult with synthesis text, files saved, and metadata.
        """
        prompt = self._build_prompt(topic)

        if self.use_managed:
            try:
                runner = ManagedResearchRunner(
                    model=self.model,
                    system_prompt=self.system_prompt,
                    verbose=self.verbose,
                )
                return await runner.run(prompt)
            except (anthropic.APIError, anthropic.APIConnectionError) as exc:
                if self.verbose:
                    print(
                        f"[research] Managed API unavailable ({type(exc).__name__}), "
                        "falling back to local runner...",
                        file=sys.stderr,
                    )

        runner = LocalResearchRunner(
            model=self.model,
            system_prompt=self.system_prompt,
            verbose=self.verbose,
        )
        return await runner.run(prompt)

    def _build_prompt(self, topic: str) -> str:
        """Build the research prompt with topic and date context."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.depth == "deep":
            return textwrap.dedent(f"""\
                Research Topic: {topic}

                Today's date: {today}

                Please conduct thorough research on this topic following your research
                protocol. Search from multiple angles, read key sources in depth, and
                produce a comprehensive synthesis. Save your findings to the data/
                directory as you go, with the final synthesis as the last file saved.
            """)
        else:
            return textwrap.dedent(f"""\
                Research Topic: {topic}
                Date: {today}

                Conduct focused research on this topic. Search for the most important
                recent information, summarize key findings, and save to data/.
            """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(response: Any) -> str:
    """Extract all text content blocks from a Claude response."""
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


def _summarize(input_dict: dict[str, Any]) -> str:
    """Create a short summary of tool input for logging."""
    if "query" in input_dict:
        q = input_dict["query"]
        return f'query="{q[:50]}..."' if len(q) > 50 else f'query="{q}"'
    if "url" in input_dict:
        u = input_dict["url"]
        return f'url="{u[:60]}..."' if len(u) > 60 else f'url="{u}"'
    if "filename" in input_dict:
        return f'file="{input_dict["filename"]}"'
    return json.dumps(input_dict, default=str)[:60]


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """CLI interface for the research agent."""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent — Claude Opus 4.7 + Tavily Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m agents.research_agent "quantum computing breakthroughs 2026"
              python -m agents.research_agent --topic "AI regulation EU" --depth brief
              python -m agents.research_agent --topic "mRNA therapeutics" --local
        """),
    )
    parser.add_argument("query", nargs="?", help="Research topic (positional)")
    parser.add_argument("--topic", "-t", help="Research topic (named)")
    parser.add_argument(
        "--depth", "-d", choices=["deep", "brief"], default="deep",
        help="Research depth: 'deep' (multi-phase) or 'brief' (focused). Default: deep",
    )
    parser.add_argument("--local", "-l", action="store_true", help="Force local runner (skip managed API)")
    parser.add_argument("--model", "-m", default=MODEL, help=f"Model to use. Default: {MODEL}")
    parser.add_argument("--output", "-o", help="Output directory (default: ./data/)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress logging")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()
    topic = args.query or args.topic

    if not topic:
        parser.error("Please provide a research topic via positional arg or --topic")

    output_dir = Path(args.output) if args.output else None

    agent = ResearchAgent(
        model=args.model,
        depth=args.depth,
        verbose=not args.quiet,
        use_managed=not args.local,
        output_dir=output_dir,
    )

    result = asyncio.run(agent.research(topic))

    if args.json:
        output = {
            "synthesis": result.synthesis,
            "files_saved": result.files_saved,
            "sources_consulted": result.sources_consulted,
            "tool_calls_count": len(result.tool_calls),
            "timestamp": result.timestamp,
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 72)
        print("RESEARCH SYNTHESIS")
        print("=" * 72 + "\n")
        print(result.synthesis)
        print("\n" + "-" * 72)
        print(result.summary)
        if result.files_saved:
            print("\nFiles saved:")
            for f in result.files_saved:
                print(f"  - {f}")
        print()


if __name__ == "__main__":
    main()
