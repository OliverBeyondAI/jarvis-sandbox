# Weather Thinking Agent

Prototype demonstrating **Claude Opus 4.7 extended thinking** combined with **MCP tool use**. The agent connects to a mock weather MCP server, reasons through multi-city weather comparisons using chain-of-thought, and produces travel recommendations — with every step of its thinking process visible.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     run_e2e_demo.py                         │
│              (top-level orchestration script)                │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌─────────────────────┐     ┌────────────────────────┐
│  mock_weather_mcp/  │     │ weather_thinking_agent/ │
│   mcp_server.py     │◄────│      agent.py           │
│                     │stdio│      viewer.py          │
│  3 weather tools:   │     │      run_demo.py        │
│  • current weather  │     │                         │
│  • forecast         │     │  Claude Opus 4.7        │
│  • alerts           │     │  + extended thinking    │
└─────────────────────┘     └────────────────────────┘
```

**Data flow:**
1. Agent starts MCP server as a subprocess (stdio transport)
2. Agent discovers available tools via MCP protocol
3. User query goes to Claude with tool definitions
4. Claude *thinks* (extended thinking) about what data it needs
5. Claude calls MCP tools to fetch weather data
6. Claude *thinks* again, analyzing the results
7. Claude produces a final recommendation
8. Full trace (thinking + tool calls + response) is captured

## Quick start

### Prerequisites

```bash
# Install dependencies (from repo root)
pip install mcp anthropic
```

### Run without API key (phases 1-2)

```bash
# From the repo root — discovers tools and makes direct MCP calls
python run_e2e_demo.py
```

This verifies the MCP server boots correctly, lists its 3 tools with schemas, and makes sample tool calls directly (no LLM involved).

### Run with extended thinking (phases 1-4)

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Full end-to-end demo with HTML trace viewer
python run_e2e_demo.py --agent --html
```

This runs all 4 phases:

| Phase | What happens | API key needed? |
|-------|-------------|-----------------|
| 1. Discovery | Boot MCP server, list tools + schemas | No |
| 2. Direct calls | Call each tool directly via MCP | No |
| 3. Agent run | Claude reasons + calls tools + responds | Yes |
| 4. HTML viewer | Generate self-contained trace visualization | Yes |

### Custom queries

```bash
python run_e2e_demo.py --agent --query "Is Denver safe for hiking this week?"
python run_e2e_demo.py --agent --html --query "Compare SF and Chicago weather"
```

### Using the agent directly

```bash
cd weather_thinking_agent

# Default travel query
python agent.py

# Custom query with thinking budget
python agent.py --budget 8000 "Compare SF and Miami weather"

# Save trace and generate HTML
python agent.py --trace trace.json --html trace_viewer.html
```

## What you'll see

### Phase 1: Tool Discovery
```
  > Starting MCP server: mock_weather_mcp/mcp_server.py
  ✓ Server connected (42ms)
  ✓ Discovered 3 tools

  get_current_weather
    Get current weather conditions for a city.
    Params: city: string (required), units: string
    Schema: 312 bytes (~78 tokens)
  ...
```

### Phase 3: Extended Thinking
The agent's reasoning process is printed in real-time:
```
  ── Turn 1 ──
  Thinking: I need to check the weather for all three cities...
  Tool call: get_current_weather({"city": "New York"})
    Result: {"city": "New York", "current": {"temperature": 72...}} (3ms)
  Tool call: get_forecast({"city": "New York", "days": 5})
    Result: {"city": "New York", "forecast": [...]} (2ms)
  ...

  ── Turn 2 ──
  Thinking: Now I have data for all three cities. Let me compare...
  Text: Based on my analysis of the weather data...
```

### HTML Trace Viewer
After running with `--html`, open `weather_thinking_agent/trace_viewer.html` in a browser. The viewer shows:
- Execution timeline (thinking blocks vs tool calls)
- Expandable thinking blocks with full chain-of-thought
- Tool call details with input/output JSON
- Agent stats (tokens, turns, wall time)
- Final response

## File reference

```
weather_thinking_agent/
├── agent.py          # Core agent: MCP connection + extended thinking loop
├── run_demo.py       # Standalone demo runner (discovery, direct calls, agent)
├── viewer.py         # HTML trace viewer generator (self-contained output)
├── pyproject.toml    # Dependencies: mcp>=1.9.0, anthropic>=0.97.0
├── trace.json        # Generated: full agent trace (after running agent)
├── trace_viewer.html # Generated: HTML visualization (after --html)
└── README.md         # This file

mock_weather_mcp/
├── mcp_server.py     # FastMCP server with 3 weather tools + mock data
├── run_demo.py       # Server-side demo runner
├── pyproject.toml    # Dependencies
└── README.md         # Server documentation

run_e2e_demo.py       # Top-level end-to-end demo orchestrating both components
```

## Key concepts

### Extended thinking
```python
response = await client.messages.create(
    model="claude-opus-4-6-20250514",
    thinking={"type": "enabled", "budget_tokens": 10_000},
    tools=claude_tools,
    messages=messages,
)

# Each response can contain thinking blocks alongside text and tool_use
for block in response.content:
    if block.type == "thinking":
        print(block.thinking)  # Claude's chain-of-thought reasoning
    elif block.type == "tool_use":
        result = await session.call_tool(block.name, block.input)
    elif block.type == "text":
        print(block.text)      # Claude's visible response
```

The `budget_tokens` parameter controls how many tokens Claude can spend on internal reasoning. Higher budgets allow deeper analysis but cost more tokens.

### MCP tool flow
1. Start MCP server as subprocess via `StdioServerParameters`
2. Initialize `ClientSession` and call `list_tools()` to discover available tools
3. Convert MCP tool schemas to Claude's `tools` format
4. Run agentic loop: Claude reasons, emits `tool_use` blocks, agent executes via MCP, feeds results back
5. Loop ends when Claude returns `end_turn` with a text response

### Trace capture
Every interaction is recorded in a structured `AgentTrace`:
- `thinking_blocks` — all chain-of-thought reasoning (text, turn, budget)
- `tool_calls` — every MCP tool invocation (name, input, result, duration)
- `final_response` — Claude's synthesized answer
- Token usage and wall time for performance analysis
