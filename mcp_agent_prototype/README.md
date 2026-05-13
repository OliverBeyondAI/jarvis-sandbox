# MCP Agent Prototype

A prototype demonstrating **Model Context Protocol (MCP)** integration with the **Anthropic Python SDK**, featuring deferred tool loading to minimize token usage.

> **Note on SDKs:** The Claude Agent SDK (`@anthropic-ai/claude-agent-sdk`) is a
> TypeScript-only package that wraps Claude Code as a subprocess. For Python, the
> official approach is the `anthropic` package (>=0.97.0) with the Messages API
> tool-use loop, which is what this prototype uses. The deferred tool loading
> pattern demonstrated here is SDK-agnostic and applies to both.

## What It Does

This prototype implements an end-to-end MCP agent that:

1. **Starts an MCP server** exposing mock enterprise tools (database search, email sending) via stdio transport
2. **Discovers tools dynamically** — the agent connects to the server and lists all available tools at runtime
3. **Defers schema loading** — on the first turn, only tool summaries (name + description) are sent to Claude in the system prompt with **no tool schemas in the API request**. Claude responds with a plan stating which tools it needs. Only those tools' schemas are then loaded for execution, saving tokens for every tool not invoked.
4. **Runs an agentic loop** — Claude reasons about which tools to use, the agent resolves schemas, executes tools via MCP, and feeds results back

### Architecture

```
┌─────────────┐    stdio     ┌──────────────────┐
│  MCP Server  │◄───────────►│    MCP Agent      │
│  (FastMCP)   │             │  (Anthropic SDK + │
│              │             │   deferred tools) │
│  Tools:      │             │                   │
│  - search_db │    tool     │  1. Discover tools│
│  - send_email│◄── calls ──►│  2. Plan (no API  │
└─────────────┘             │     tool schemas) │
                             │  3. Load needed   │
                             │     schemas only  │
                             │  4. Execute loop  │
                             └──────────────────┘
```

### Deferred Tool Loading — How It Works

Standard approach: send all tool schemas to Claude on every request (~N tools x schema size tokens).

This prototype's approach:
1. **Planning turn**: Send tool summaries in the system prompt, with **zero** tool schemas in the API `tools` parameter. Claude responds with a `TOOLS_NEEDED: tool1, tool2` declaration.
2. **Schema loading**: Parse the plan, load full schemas for only the requested tools.
3. **Execution loop**: Re-issue the user's request with only the needed schemas attached. Claude generates proper `tool_use` blocks and tools are executed via MCP.

Tools that are never invoked never cost schema tokens. The `tokens_saved_estimate` in the result reflects the actual savings (total schema tokens minus loaded schema tokens).

## Setup

### Requirements

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
cd mcp_agent_prototype

# Option A: pip install dependencies directly
pip install "mcp>=1.9.0" "anthropic>=0.97.0"

# Option B: install as a package (editable)
pip install -e .
```

### Set API Key

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

## Usage

### Full Demo (recommended)

Runs tool discovery + 3 example queries showing different capabilities:

```bash
python run_demo.py
```

This will:
1. Connect to the MCP server and list all discovered tools
2. Run a **database search** query (filtered customer lookup)
3. Run a **product catalog** query (price-filtered product search)
4. Run a **multi-tool workflow** (search -> email chain)

### Tool Discovery Only (no API key needed)

See what tools the MCP server exposes, with parameter details and schema sizes:

```bash
python run_demo.py --discover
```

### Custom Query

```bash
python run_demo.py --query "Find churned customers and email a win-back summary to sales@company.com"
```

### Interactive Mode

```bash
python agent.py
```

### Direct Agent CLI

```bash
python agent.py "Find enterprise customers in tech"
python agent.py --list-tools
```

### MCP Server Standalone

```bash
# stdio transport (default, used by MCP clients)
python mcp_server.py

# SSE transport (HTTP on port 8000)
python mcp_server.py --transport sse
```

## Files

| File | Description |
|------|-------------|
| `run_demo.py` | End-to-end demo script — starts server + agent, runs example queries |
| `agent.py` | MCP agent with deferred tool loading and agentic loop |
| `mcp_server.py` | FastMCP server with mock `search_database` and `send_email` tools |
| `pyproject.toml` | Package metadata and dependencies |
| `.gitignore` | Excludes `__pycache__/`, build artifacts |

## Mock Tools

The MCP server exposes two tools with realistic schemas:

**`search_database`** — Query a simulated CRM/product database
- Tables: `customers` (8 records), `products` (6 records)
- Supports natural-language queries and structured JSON filters
- Parameters: `query`, `table`, `filters`, `limit`

**`send_email`** — Send through a simulated SMTP gateway
- Emails are logged internally (not actually transmitted)
- Parameters: `to`, `subject`, `body`, `cc`, `bcc`, `priority`, `reply_to`

## Example Output

```
================================================================
    MCP Tool Discovery
================================================================

  Connected! Found 2 tools:

  search_database
    Search the company database for records matching a query.
    Schema: 832 bytes (~208 tokens)
    Params: query: string *, table: string, filters: string, limit: integer

  send_email
    Send an email through the company's simulated SMTP gateway.
    Schema: 614 bytes (~153 tokens)
    Params: to: string *, subject: string *, body: string *, cc: string, ...

  Total schema payload: 1446 bytes (~361 tokens)
  With deferred loading, schemas are only sent when a tool is invoked.
```
