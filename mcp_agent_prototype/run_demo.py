#!/usr/bin/env python3
"""
MCP Agent Demo — End-to-end demonstration of MCP tool discovery, deferred
tool loading, and agent invocation.

This script starts the MCP server and agent together, runs a sequence of
example queries that showcase different capabilities, and prints a summary.

Usage:
    python run_demo.py              # run full demo (all example queries)
    python run_demo.py --discover   # tool discovery only (no API calls)
    python run_demo.py --query "your custom query here"

Requires:
    - ANTHROPIC_API_KEY environment variable set
    - Python 3.11+
    - pip install mcp anthropic
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path

# Ensure the package directory is importable
sys.path.insert(0, str(Path(__file__).parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent import MCPAgent, ToolSummary

MCP_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")

# ---------------------------------------------------------------------------
# Color / formatting helpers (no dependencies)
# ---------------------------------------------------------------------------

_SUPPORTS_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _SUPPORTS_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def cyan(t: str) -> str:
    return _c("36", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def banner(title: str, width: int = 64) -> str:
    line = "=" * width
    pad = (width - len(title)) // 2
    return f"\n{line}\n{' ' * pad}{bold(title)}\n{line}"


def section(title: str, width: int = 64) -> str:
    return f"\n{dim('-' * width)}\n  {cyan(title)}\n{dim('-' * width)}"


# ---------------------------------------------------------------------------
# Demo: tool discovery (no API key needed)
# ---------------------------------------------------------------------------

async def demo_discover() -> None:
    """Connect to the MCP server and display all discovered tools."""
    print(banner("MCP Tool Discovery"))
    print()
    print(f"  Server script: {dim(MCP_SERVER_SCRIPT)}")
    print()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

            print(f"  {green('Connected!')} Found {bold(str(len(result.tools)))} tools:\n")

            total_schema_bytes = 0
            for tool in result.tools:
                schema_json = json.dumps(tool.inputSchema, indent=2)
                schema_bytes = len(schema_json)
                total_schema_bytes += schema_bytes

                desc_first_line = (tool.description or "").split("\n")[0].strip()
                print(f"  {bold(tool.name)}")
                print(f"    {desc_first_line}")
                print(f"    Schema: {schema_bytes} bytes (~{schema_bytes // 4} tokens)")

                # Show parameter names
                props = tool.inputSchema.get("properties", {})
                required = set(tool.inputSchema.get("required", []))
                params = []
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "any")
                    marker = " *" if pname in required else ""
                    params.append(f"{pname}: {ptype}{marker}")
                print(f"    Params: {', '.join(params)}")
                print()

            print(f"  Total schema payload: {bold(str(total_schema_bytes))} bytes (~{total_schema_bytes // 4} tokens)")
            print(f"  With deferred loading, schemas are only sent when a tool is invoked.")
            print()


# ---------------------------------------------------------------------------
# Demo: run agent queries
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    {
        "label": "Database Search",
        "query": "Find all enterprise customers in the technology industry",
        "description": "Tests search_database with filtered customer lookup",
    },
    {
        "label": "Product Catalog",
        "query": "What products are available for under $200?",
        "description": "Tests search_database against the products table with price filtering",
    },
    {
        "label": "Multi-Tool Workflow",
        "query": (
            "Search for enterprise customers, then send an email to reports@company.com "
            "with a summary of what you found. Use subject 'Enterprise Customer Report'."
        ),
        "description": "Tests chaining search_database -> send_email in a single agent run",
    },
]


async def demo_agent_queries(queries: list[dict]) -> None:
    """Run one or more queries through the agent and display results."""
    print(banner("MCP Agent Demo"))
    print()
    print(f"  Model:   {dim(MCPAgent().model)}")
    print(f"  Server:  {dim(MCP_SERVER_SCRIPT)}")
    print(f"  Queries: {len(queries)}")
    print()

    for i, q in enumerate(queries, 1):
        label = q.get("label", f"Query {i}")
        desc = q.get("description", "")
        prompt = q["query"]

        print(section(f"[{i}/{len(queries)}] {label}"))
        if desc:
            print(f"  {dim(desc)}")
        print(f"\n  {bold('Prompt:')} {prompt}\n")

        agent = MCPAgent(verbose=True)

        t0 = time.time()
        try:
            result = await agent.run(prompt)
        except Exception as e:
            print(f"\n  {red('Error:')} {e}\n")
            continue
        elapsed = time.time() - t0

        # Display result
        print(f"\n  {bold('Agent Response:')}")
        for line in result.text.strip().splitlines():
            print(f"    {line}")

        print(f"\n  {bold('Stats:')}")
        print(f"    Tools discovered : {', '.join(result.tools_discovered)}")
        print(f"    Tools loaded     : {', '.join(result.tools_loaded) or '(none)'}")
        not_loaded = set(result.tools_discovered) - set(result.tools_loaded)
        if not_loaded:
            print(f"    Tools deferred   : {', '.join(not_loaded)} {dim('(schemas never sent)')}")
        print(f"    Tool calls       : {len(result.tool_calls)}")
        print(f"    Tokens saved     : ~{result.tokens_saved_estimate} {dim('(schemas never loaded)')}")
        print(f"    Total schema tok : ~{result.total_schema_tokens} {dim('(if all loaded)')}")
        print(f"    Elapsed          : {elapsed:.1f}s")

        if result.tool_calls:
            print(f"\n  {bold('Tool Call Log:')}")
            for j, tc in enumerate(result.tool_calls, 1):
                input_str = json.dumps(tc["input"], default=str)
                if len(input_str) > 100:
                    input_str = input_str[:97] + "..."
                print(f"    {j}. {green(tc['name'])}({dim(input_str)})")
                preview = tc.get("result_preview", "")
                if preview:
                    short = preview[:120].replace("\n", " ")
                    print(f"       -> {dim(short)}...")
        print()

    print(banner("Demo Complete"))
    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _check_api_key() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    print(f"\n  {red('Error:')} ANTHROPIC_API_KEY environment variable is not set.")
    print(f"  Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")
    print(f"  You can still run {bold('--discover')} without an API key.\n")
    return False


def main() -> None:
    args = sys.argv[1:]

    # --discover: tool discovery only
    if "--discover" in args:
        asyncio.run(demo_discover())
        return

    # --query "...": single custom query
    if "--query" in args:
        idx = args.index("--query")
        if idx + 1 >= len(args):
            print("Usage: run_demo.py --query \"your query here\"")
            sys.exit(1)
        query_text = args[idx + 1]
        if not _check_api_key():
            sys.exit(1)
        asyncio.run(demo_agent_queries([{"label": "Custom Query", "query": query_text}]))
        return

    # --help
    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    # Default: full demo
    print(banner("MCP Agent Prototype — Full Demo"))
    print()
    print("  This demo will:")
    print(f"    1. {cyan('Discover')} tools from the MCP server (no API key needed)")
    print(f"    2. {cyan('Run 3 example queries')} through the Claude agent")
    print(f"       showing tool discovery, deferred loading, and multi-tool chains")
    print()

    # Phase 1: Discovery (always works)
    asyncio.run(demo_discover())

    # Phase 2: Agent queries (needs API key)
    if not _check_api_key():
        print("  Skipping agent queries. Set ANTHROPIC_API_KEY to run the full demo.\n")
        return

    asyncio.run(demo_agent_queries(EXAMPLE_QUERIES))


if __name__ == "__main__":
    main()
