# Claude Agent SDK — Research & Implementation Guide

Research findings for building autonomous AI agents, covering the official SDK API surface, patterns from existing prototypes in this repo, and best practices for implementation.

---

## 1. SDK Identity & Architecture

### Two Distinct SDK Surfaces

| SDK | Language | Package | Approach |
|-----|----------|---------|----------|
| **Claude Agent SDK** (TypeScript) | Node.js 18+ | `@anthropic-ai/claude-agent-sdk` v0.2.138 | Spawns Claude Code as subprocess; full tool suite built-in |
| **Anthropic Python SDK** | Python 3.9+ | `anthropic` >=0.97.0 | Direct API calls; Messages API + beta managed-agents API |

**Key distinction:** The TypeScript SDK wraps the entire Claude Code CLI (Bash, Read, Edit, Glob, Grep, WebFetch, etc.) and manages it as a subprocess. The Python SDK provides raw API access — you build your own tool loop and tool implementations.

### TypeScript SDK Architecture

```
Your App → query() → Claude Code subprocess → Anthropic API
                ↕ JSON stdin/stdout protocol
           Tool execution, file ops, MCP servers
```

### Python SDK Architecture

```
Your App → anthropic.Anthropic() → Anthropic Messages API
                ↕ tool_use / tool_result loop
           Custom tool implementations (your code)
```

---

## 2. TypeScript SDK — Core API

### `query()` — Primary Entry Point

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

for await (const message of query({ prompt: "Analyze this codebase" })) {
  if (message.type === 'assistant') console.log(message.message);
  if (message.type === 'result') console.log(message.result);
}
```

Returns an `AsyncGenerator<SDKMessage, void>` with streaming control methods:
- `interrupt()` — stop execution
- `setModel(model)` — switch model mid-session
- `readFile(path)` — read from session filesystem
- `rewindFiles(messageId)` — undo file changes to checkpoint
- `setMcpServers(servers)` — add/remove MCP tool servers dynamically
- `close()` — terminate

### `startup()` — Pre-warmed Queries (Zero Latency)

```typescript
const warm = await startup({ options: { model: 'claude-sonnet-4-6' } });
const q = warm.query("Analyze this codebase");
for await (const msg of q) { /* ... */ }
warm.close();
```

### Custom Tools via MCP

```typescript
import { query, tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod/v4';

const weatherTool = tool(
  'get_weather',
  'Get current weather for a city',
  { city: z.string() },
  async (args) => ({
    content: [{ type: 'text', text: `Sunny in ${args.city}` }]
  }),
  { annotations: { readOnly: true }, alwaysLoad: true }
);

const server = createSdkMcpServer({
  name: 'my-tools',
  version: '1.0.0',
  tools: [weatherTool],
});

for await (const msg of query({
  prompt: "Weather in Tokyo?",
  options: { mcpServers: { 'my-tools': server } }
})) { /* ... */ }
```

### Subagent Definitions

```typescript
const options = {
  agents: {
    'test-runner': {
      description: 'Runs tests and reports results',
      prompt: 'You are a test runner...',
      tools: ['Bash', 'Read', 'Grep'],
      model: 'sonnet',
      maxTurns: 10,
      background: false,
    },
  },
};
```

### Key Configuration Options

| Option | Values | Purpose |
|--------|--------|---------|
| `model` | `'claude-sonnet-4-6'`, `'claude-opus-4-7'` | Model selection |
| `thinking` | `{ type: 'adaptive' }`, `{ type: 'enabled', budgetTokens: N }` | Extended thinking |
| `effort` | `'low'` / `'medium'` / `'high'` / `'xhigh'` / `'max'` | Reasoning effort |
| `maxTurns` | number | Turn limit |
| `maxBudgetUsd` | number | Cost cap |
| `tools` | `['Bash', 'Read']` or `{ type: 'preset', preset: 'claude_code' }` | Tool access |
| `outputFormat` | `{ type: 'json_schema', schema: ... }` | Structured output |
| `sandbox` | `{ enabled: true }` | Sandboxed execution |
| `permissionMode` | `'default'` / `'acceptEdits'` / `'bypassPermissions'` | Permission level |
| `betas` | `['context-1m-2025-08-07']` | Beta features (1M context) |

### Session Management

```typescript
// Resume a session
query({ prompt: "continue", options: { resume: sessionId } })

// Fork on resume
query({ prompt: "try different approach", options: { resume: id, forkSession: true } })
```

Session stores for production: implement `SessionStore` interface for Redis/Postgres/S3.

### Hook System (28 Events)

Lifecycle interception at every stage: `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `SubagentStart`, `PermissionRequest`, `FileChanged`, etc.

---

## 3. Python SDK — Core API

### Messages API with Tool-Use Loop (Primary Pattern)

```python
import anthropic

client = anthropic.AsyncAnthropic()

async def run_agent(prompt: str, tools: list, system: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    for _ in range(25):  # max turns
        response = await client.messages.create(
            model="claude-opus-4-7-20250501",
            max_tokens=8192,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Check for tool use
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return response.content[0].text  # done

        # Execute tools and continue
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tu in tool_uses:
            result = await execute_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})
```

### Beta Managed-Agents API

```python
# Create agent with managed state
agent = client.beta.agents.create(
    model="claude-opus-4-7-20250501",
    name="research-agent",
    system=system_prompt,
    tools=tools,
)

# Create environment & session
environment = client.beta.environments.create(name="agent-env")
session = client.beta.sessions.create(agent=agent.id, environment_id=environment.id)

# Event-driven interaction
client.beta.sessions.events.send(session_id=session.id, events=[...])
events = client.beta.sessions.events.list(session_id=session.id, order="asc")

# Event types: "agent.message", "agent.custom_tool_use", "session.status_idle"
```

### Tool Definition (MCP-Compatible)

```python
TOOLS = [
    {
        "name": "web_search",
        "type": "custom",           # required for managed-agents API
        "description": "Search the web",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

# Tool dispatcher
async def execute_tool(name: str, input_dict: dict) -> str:
    if name == "web_search":
        return json.dumps(await web_search(**input_dict))
    raise ValueError(f"Unknown tool: {name}")
```

**Note:** For the Messages API, strip the `"type"` field from tool schemas. The managed-agents API requires it.

---

## 4. Existing Prototypes in This Repo

### Inventory (9 Agent Projects)

| Project | Pattern | Key Feature |
|---------|---------|-------------|
| `agents/` | 3-stage pipeline + orchestrator | ManagedAgent → LocalAgent fallback |
| `agents/trend_research/` | Stage 1: web research | Tavily search + URL fetching |
| `agents/synthesis/` | Stage 2: trend analysis | Structured application mapping |
| `agents/memo_generation/` | Stage 3: memo output | Executive-ready artifacts |
| `chief_of_staff_agent/` | Multi-phase research | Structured ResearchBrief model |
| `dispatch_agent/` | Parallel sub-agents | FanOutChannel async coordination |
| `ophthoflow_pa_agent/` | Domain-specific (healthcare) | Prior auth workflow |
| `ophthoflow-prior-auth-agent/` | Domain-specific (advanced) | AWS Bedrock support |
| `research_summarizer/` | URL summarization | Single-purpose tool agent |

### Proven Patterns from Prototypes

**1. Frozen Dataclass Config**
```python
@dataclass(frozen=True)
class Config:
    model: str = "claude-opus-4-7-20250501"
    max_tokens: int = 8192
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))

    def validate(self) -> list[str]:
        return ["API key not set"] if not self.api_key else []
```

**2. Pydantic Models for Structured Output**
```python
class ResearchBrief(BaseModel):
    title: str
    summary: str
    sources: list[Source]
    confidence: float
```

**3. Tool Registry Pattern**
```python
ALL_TOOLS = {"fetch_url": fetch_url, "tavily_search": tavily_search}

async def execute_tool(name: str, input_dict: dict) -> str:
    handler = ALL_TOOLS[name]
    return json.dumps(await handler(**input_dict), default=str)
```

**4. Dual-Mode Storage (Local/S3)**
```python
if os.environ.get("USE_LOCAL_STORAGE"):
    save_local(f"./reports/{domain}/{timestamp}/report.json", data)
else:
    save_s3(bucket, f"{prefix}/{domain}/{timestamp}/report.json", data)
```

**5. Graceful Degradation**
```
ManagedAgent API (beta) → catch error → LocalAgent fallback (stable)
```

---

## 5. Implementation Decision Matrix

### When to Use Which Approach

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Full-stack app with file ops | TS SDK `query()` | Built-in tools (Bash, Edit, Read) |
| Custom research pipeline | Python Messages API | Full control over tool loop |
| Production with state mgmt | TS SDK + SessionStore | Persistence, resume, fork |
| Quick prototype | Python Messages API | Simplest setup, most portable |
| Parallel sub-agents | TS SDK agents config | Native subagent support |
| Parallel sub-agents (Python) | `dispatch_agent/` pattern | FanOutChannel + asyncio |
| Enterprise/AWS | Python + Bedrock client | Direct Bedrock integration |
| Browser-based agent | TS SDK `/browser` export | WebSocket transport |

### Model Selection

| Model | Use Case | Cost |
|-------|----------|------|
| `claude-opus-4-7-20250501` | Complex reasoning, multi-step research | Highest |
| `claude-opus-4-6` | General agent tasks | High |
| `claude-sonnet-4-6` | Fast tasks, sub-agents, high-volume | Medium |
| `claude-haiku-4-5-20251001` | Classification, routing, simple tools | Low |

---

## 6. Best Practices

### Architecture
1. **Messages API tool-use loop as primary pattern** — more reliable and portable than the beta managed-agents API
2. **Pydantic models for all inter-agent data** — enforces output quality and makes pipelines predictable
3. **Tool registry (dict + dispatcher)** — adding capabilities is trivial, schemas stay in sync
4. **Frozen dataclass configs** — immutable, validated, env-driven

### Safety & Control
5. **Restrict tools per agent** — each agent gets only the tools it needs
6. **Set maxTurns and budget caps** — prevent runaway loops and costs
7. **Sandbox untrusted execution** — filesystem/network isolation for user-provided code
8. **Use permission callbacks** — fine-grained control over dangerous operations

### Performance
9. **Pre-warm with `startup()`** — zero-latency dispatch for latency-sensitive paths (TS SDK)
10. **Use `effort` levels** — `'low'` for routing, `'high'` for complex reasoning
11. **Parallel sub-agents** — fan-out independent research queries
12. **1M context beta** — `betas: ['context-1m-2025-08-07']` for large codebases

### Production
13. **Session stores** — Redis/Postgres/S3 for distributed persistence and resume
14. **Dual-mode storage** — local dev / cloud prod controlled by env vars
15. **Structured output** — JSON schema for programmatic consumption
16. **Hook system** — intercept and customize at 28 lifecycle points

---

## 7. Quick Start Templates

### Minimal Python Agent

```python
import anthropic, json, asyncio

TOOLS = [{
    "name": "calculator",
    "description": "Evaluate a math expression",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"]
    }
}]

async def main():
    client = anthropic.AsyncAnthropic()
    messages = [{"role": "user", "content": "What is 42 * 17?"}]

    for _ in range(10):
        resp = await client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            tools=TOOLS, messages=messages,
        )
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            print(resp.content[0].text)
            break
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tu.id,
             "content": str(eval(tu.input["expression"]))}
            for tu in tool_uses
        ]})

asyncio.run(main())
```

### Minimal TypeScript Agent

```typescript
import { query, tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod/v4';

const calc = tool('calculator', 'Evaluate math', { expr: z.string() },
  async ({ expr }) => ({ content: [{ type: 'text', text: String(eval(expr)) }] }));

const server = createSdkMcpServer({ name: 'tools', version: '1.0.0', tools: [calc] });

for await (const msg of query({
  prompt: 'What is 42 * 17?',
  options: { mcpServers: { tools: server }, model: 'claude-sonnet-4-6' },
})) {
  if (msg.type === 'result') console.log(msg.result);
}
```

---

*Research compiled 2026-05-10. SDK versions: `@anthropic-ai/claude-agent-sdk` v0.2.138, `anthropic` (Python) >=0.97.0.*
