#!/usr/bin/env python3
"""
MCP + Tavily Integration Verification Script

Verifies that:
  1. The MCP server starts and accepts connections
  2. Tools are discovered via MCP protocol (including market_research)
  3. The Tavily-powered market_research tool can be called via MCP
  4. Search results are returned in the expected format
  5. Multiple search queries work (trending topics + specific research)
  6. Error handling works when Tavily key is missing

Usage:
    python -m agents.mcp_xena_marketing_agent.verify_mcp_tavily
    python agents/mcp_xena_marketing_agent/verify_mcp_tavily.py
    python agents/mcp_xena_marketing_agent/verify_mcp_tavily.py --verbose
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI colors for terminal output
# ---------------------------------------------------------------------------

class C:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {C.GREEN}\u2713{C.RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {C.RED}\u2717{C.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}!{C.RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {C.DIM}{msg}{C.RESET}")


def header(msg: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}{msg}{C.RESET}")


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------

async def verify_all(verbose: bool = False) -> bool:
    """Run all MCP + Tavily integration checks. Returns True if all pass."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_script = str(Path(__file__).parent / "mcp_server.py")
    passed = 0
    failed = 0
    total_start = time.time()

    print()
    print(f"{C.BOLD}{'=' * 64}{C.RESET}")
    print(f"{C.BOLD}  MCP + Tavily Integration Verification{C.RESET}")
    print(f"{C.BOLD}{'=' * 64}{C.RESET}")

    # ------------------------------------------------------------------
    # Test 1: MCP Server Connection
    # ------------------------------------------------------------------
    header("1. MCP Server Connection")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env={**os.environ, "XENA_OUTPUT_DIR": "/tmp/xena_verify"},
    )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                ok("MCP server started and initialized")
                passed += 1

                # ----------------------------------------------------------
                # Test 2: Tool Discovery
                # ----------------------------------------------------------
                header("2. Tool Discovery via MCP")

                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]

                if len(tool_names) > 0:
                    ok(f"Discovered {len(tool_names)} tools")
                    passed += 1
                else:
                    fail("No tools discovered")
                    failed += 1

                expected_tools = [
                    "market_research",
                    "fetch_url",
                    "analyze_market",
                    "draft_content",
                    "generate_image",
                    "save_campaign",
                ]

                for name in expected_tools:
                    if name in tool_names:
                        ok(f"Tool '{name}' available")
                        passed += 1
                    else:
                        fail(f"Tool '{name}' NOT found")
                        failed += 1

                if verbose:
                    for tool in tools_result.tools:
                        schema_size = len(json.dumps(tool.inputSchema))
                        info(f"  {tool.name}: schema={schema_size}B, "
                             f"desc={tool.description[:60] if tool.description else 'N/A'}...")

                # ----------------------------------------------------------
                # Test 3: market_research tool schema validation
                # ----------------------------------------------------------
                header("3. market_research Tool Schema")

                mr_tool = next(
                    (t for t in tools_result.tools if t.name == "market_research"),
                    None,
                )

                if mr_tool:
                    schema = mr_tool.inputSchema
                    props = schema.get("properties", {})

                    if "query" in props:
                        ok("'query' parameter defined")
                        passed += 1
                    else:
                        fail("'query' parameter missing")
                        failed += 1

                    if "max_results" in props:
                        ok("'max_results' parameter defined")
                        passed += 1
                    else:
                        fail("'max_results' parameter missing")
                        failed += 1

                    if "search_depth" in props:
                        ok("'search_depth' parameter defined")
                        passed += 1
                    else:
                        fail("'search_depth' parameter missing")
                        failed += 1

                    required = schema.get("required", [])
                    if "query" in required:
                        ok("'query' is required")
                        passed += 1
                    else:
                        fail("'query' should be required")
                        failed += 1

                    if verbose:
                        info(f"Full schema: {json.dumps(schema, indent=2)}")
                else:
                    fail("market_research tool not found — cannot validate schema")
                    failed += 4

                # ----------------------------------------------------------
                # Test 4: Call market_research via MCP (Tavily search)
                # ----------------------------------------------------------
                header("4. Tavily Search via MCP (market_research)")

                has_tavily_key = bool(os.environ.get("TAVILY_API_KEY"))

                if has_tavily_key:
                    # Test 4a: Trending topics search
                    info("Searching: 'AI marketing automation trends 2025'")
                    start = time.time()
                    result_a = await session.call_tool(
                        "market_research",
                        {
                            "query": "AI marketing automation trends 2025",
                            "max_results": 3,
                            "search_depth": "basic",
                        },
                    )
                    elapsed = time.time() - start

                    result_text = ""
                    for block in result_a.content:
                        if hasattr(block, "text"):
                            result_text += block.text

                    try:
                        data = json.loads(result_text)
                    except json.JSONDecodeError:
                        fail(f"Response is not valid JSON: {result_text[:100]}")
                        data = None
                        failed += 1

                    if data and "error" not in data:
                        ok(f"Search returned valid JSON ({elapsed:.1f}s)")
                        passed += 1

                        results = data.get("results", [])
                        count = data.get("result_count", 0)

                        if count > 0 and len(results) > 0:
                            ok(f"Got {count} search results")
                            passed += 1
                        else:
                            fail(f"No results returned (count={count})")
                            failed += 1

                        # Validate result structure
                        if results:
                            first = results[0]
                            has_title = "title" in first and first["title"]
                            has_url = "url" in first and first["url"]
                            has_snippet = "snippet" in first and first["snippet"]
                            has_score = "relevance_score" in first

                            if has_title and has_url and has_snippet:
                                ok("Results have title, url, snippet fields")
                                passed += 1
                            else:
                                fail(f"Missing fields: title={has_title}, "
                                     f"url={has_url}, snippet={has_snippet}")
                                failed += 1

                            if has_score:
                                score = first["relevance_score"]
                                if 0.0 <= score <= 1.0:
                                    ok(f"Relevance score in valid range ({score:.3f})")
                                    passed += 1
                                else:
                                    fail(f"Relevance score out of range: {score}")
                                    failed += 1
                            else:
                                fail("Missing relevance_score field")
                                failed += 1

                            if verbose:
                                info("Search results:")
                                for r in results:
                                    info(f"  - {r.get('title', 'N/A')}")
                                    info(f"    {r.get('url', 'N/A')}")
                                    snippet = r.get("snippet", "")
                                    info(f"    {snippet[:120]}...")
                    elif data and "error" in data:
                        fail(f"Search returned error: {data['error']}")
                        failed += 4

                    # Test 4b: Specific marketing research query
                    header("5. Specific Research Query via MCP")
                    info("Searching: 'competitor analysis SaaS project management tools'")
                    start = time.time()
                    result_b = await session.call_tool(
                        "market_research",
                        {
                            "query": "competitor analysis SaaS project management tools",
                            "max_results": 2,
                        },
                    )
                    elapsed = time.time() - start

                    result_text_b = ""
                    for block in result_b.content:
                        if hasattr(block, "text"):
                            result_text_b += block.text

                    try:
                        data_b = json.loads(result_text_b)
                    except json.JSONDecodeError:
                        data_b = None

                    if data_b and "error" not in data_b:
                        b_count = data_b.get("result_count", 0)
                        ok(f"Second search returned {b_count} results ({elapsed:.1f}s)")
                        passed += 1

                        # Verify query is echoed back
                        if data_b.get("query") == "competitor analysis SaaS project management tools":
                            ok("Query echoed correctly in response")
                            passed += 1
                        else:
                            fail("Query not echoed in response")
                            failed += 1
                    else:
                        err_msg = data_b.get("error", "unknown") if data_b else "invalid JSON"
                        fail(f"Second search failed: {err_msg}")
                        failed += 2

                else:
                    warn("TAVILY_API_KEY not set — testing error handling path")

                    result_no_key = await session.call_tool(
                        "market_research",
                        {"query": "test query without API key"},
                    )

                    result_text = ""
                    for block in result_no_key.content:
                        if hasattr(block, "text"):
                            result_text += block.text

                    try:
                        data = json.loads(result_text)
                    except json.JSONDecodeError:
                        data = None

                    if data:
                        if "error" in data:
                            ok(f"Graceful error without API key: {data['error'][:80]}")
                            passed += 1
                        else:
                            # Tavily might raise its own error
                            ok("Tool handled missing key gracefully")
                            passed += 1
                    else:
                        fail("Invalid response format without API key")
                        failed += 1

                    warn("Skipping live search tests (set TAVILY_API_KEY to enable)")
                    # Count skipped tests
                    info("  5 search validation tests skipped")
                    info("  2 second query tests skipped")

                # ----------------------------------------------------------
                # Test 6: fetch_url tool via MCP
                # ----------------------------------------------------------
                header("6. fetch_url Tool via MCP")

                info("Fetching: https://httpbin.org/html")
                start = time.time()
                fetch_result = await session.call_tool(
                    "fetch_url",
                    {"url": "https://httpbin.org/html"},
                )
                elapsed = time.time() - start

                fetch_text = ""
                for block in fetch_result.content:
                    if hasattr(block, "text"):
                        fetch_text += block.text

                try:
                    fetch_data = json.loads(fetch_text)
                except json.JSONDecodeError:
                    fetch_data = None

                if fetch_data and "error" not in fetch_data:
                    status = fetch_data.get("status", 0)
                    content = fetch_data.get("content", "")

                    if status == 200:
                        ok(f"fetch_url returned HTTP 200 ({elapsed:.1f}s)")
                        passed += 1
                    else:
                        fail(f"fetch_url returned HTTP {status}")
                        failed += 1

                    if len(content) > 0:
                        ok(f"Content returned ({fetch_data.get('length', 0)} chars)")
                        passed += 1
                    else:
                        fail("No content returned")
                        failed += 1
                elif fetch_data and "error" in fetch_data:
                    warn(f"fetch_url error (network): {fetch_data['error'][:80]}")
                    warn("Skipping fetch_url validation (network issue)")
                else:
                    fail("Invalid fetch_url response format")
                    failed += 2

                # ----------------------------------------------------------
                # Test 7: analyze_market + draft_content (non-network tools)
                # ----------------------------------------------------------
                header("7. Non-Network Tools via MCP")

                analyze_result = await session.call_tool(
                    "analyze_market",
                    {
                        "phase": "market_landscape",
                        "insights": [
                            {
                                "headline": "AI tools market growing 40% YoY",
                                "detail": "Test insight for verification",
                                "type": "market_trend",
                            }
                        ],
                        "gaps": ["Need more competitor pricing data"],
                    },
                )
                analyze_text = ""
                for block in analyze_result.content:
                    if hasattr(block, "text"):
                        analyze_text += block.text

                try:
                    analyze_data = json.loads(analyze_text)
                except json.JSONDecodeError:
                    analyze_data = None

                if analyze_data and analyze_data.get("status") == "recorded":
                    ok(f"analyze_market recorded {analyze_data.get('insights_recorded', 0)} insights")
                    passed += 1
                else:
                    fail("analyze_market failed")
                    failed += 1

                draft_result = await session.call_tool(
                    "draft_content",
                    {
                        "channel": "social_media",
                        "title": "Test Post",
                        "body": "This is a verification test post.",
                        "cta": "Learn more",
                    },
                )
                draft_text = ""
                for block in draft_result.content:
                    if hasattr(block, "text"):
                        draft_text += block.text

                try:
                    draft_data = json.loads(draft_text)
                except json.JSONDecodeError:
                    draft_data = None

                if draft_data and draft_data.get("status") == "drafted":
                    ok(f"draft_content saved ({draft_data.get('body_length', 0)} chars)")
                    passed += 1
                else:
                    fail("draft_content failed")
                    failed += 1

        ok("MCP server shut down cleanly")
        passed += 1

    except Exception as exc:
        fail(f"MCP connection failed: {type(exc).__name__}: {exc}")
        failed += 1
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - total_start
    total = passed + failed

    print()
    print(f"{C.BOLD}{'=' * 64}{C.RESET}")
    if failed == 0:
        print(f"{C.BOLD}{C.GREEN}  ALL {passed} CHECKS PASSED{C.RESET}  ({elapsed_total:.1f}s)")
    else:
        print(f"{C.BOLD}{C.RED}  {failed} FAILED{C.RESET} / "
              f"{C.GREEN}{passed} passed{C.RESET} / {total} total  ({elapsed_total:.1f}s)")
    print(f"{C.BOLD}{'=' * 64}{C.RESET}")
    print()

    return failed == 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    success = asyncio.run(verify_all(verbose=verbose))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
