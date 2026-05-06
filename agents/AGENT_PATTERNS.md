# Agent Patterns & Conventions Reference

Research summary of patterns used in `jarvis-sandbox/agents/`.

---

## Architecture Overview

**Sequential 3-stage pipeline:**
```
Topic → [Trend Research] → ResearchReport → [Synthesis] → SynthesisReport → [Memo Generation] → ArtifactBundle
```

Each agent is independent (can run standalone) and composable via the orchestrator.

---

## SDK & Import Conventions

```python
import anthropic

# Sync client (most uses)
client = anthropic.Anthropic()

# Async client (tool loops)
client = anthropic.AsyncAnthropic()

# AWS Bedrock variant
client = anthropic.AsyncAnthropicBedrock()

# Managed-agents API (beta)
client.beta.agents.create()
client.beta.environments.create()
client.beta.sessions.create()

# Messages API (fallback)
client.messages.create(model=..., max_tokens=..., system=..., tools=..., messages=...)
```

**Default model:** `"claude-opus-4-7-20250501"`

---

## Tool Definition Pattern

Tools use MCP-compatible custom schema:

```python
TOOLS = [
    {
        "name": "tool_name",
        "type": "custom",
        "description": "What the tool does",
        "input_schema": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "..."}
            },
            "required": ["param"]
        }
    }
]
```

**Dispatcher pattern:**
```python
async def execute_tool(name: str, input_dict: dict) -> str:
    if name == "fetch_url":
        result = await fetch_url(**input_dict)
    elif name == "tavily_search":
        result = await tavily_search(**input_dict)
    return json.dumps(result, default=str)
```

All tool implementations are async. Sync libraries wrapped with `asyncio.to_thread()`.

---

## Agent Runner Pattern (agent.py)

Two runners with automatic failover:

1. **ManagedAgentRunner** — Uses `client.beta.agents` API (managed sessions, streaming events)
2. **LocalAgentRunner** — Standard agentic loop via messages API (up to 25 turns)

Public **Agent** facade selects runner and handles fallback.

---

## Configuration Pattern

Frozen dataclass with env var defaults:

```python
@dataclass(frozen=True)
class Config:
    model: str = "claude-opus-4-7-20250501"
    max_tokens: int = 8192
    tavily_api_key: str = field(
        default_factory=lambda: os.environ.get("TAVILY_API_KEY", "")
    )

    @classmethod
    def from_env(cls) -> Config:
        return cls()

    def validate(self) -> list[str]:
        warnings = []
        if not self.tavily_api_key:
            warnings.append("TAVILY_API_KEY not set")
        return warnings
```

**Key env vars:** `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `MEETING_PREP_USE_BEDROCK`

---

## Data Models (Pydantic)

All inter-agent data uses Pydantic BaseModel with enums for categorical fields:

- `ResearchReport` — trends, analyses, sources, methodology
- `SynthesisReport` — applications, themes, priorities, risks
- `InternalMemo` / `ArtifactBundle` — rendered outputs + metadata

Enums: `TrendCategory`, `Platform`, `FitLevel`, `EffortLevel`, `ImpactLevel`, `MemoAudience`

---

## Storage Pattern

Dual-mode (local dev / S3 prod) controlled by env vars:

```python
# Local: ./trend_reports/{domain}/{timestamp}/report.json
# S3:    s3://bucket/prefix/{domain}/{timestamp}/report.json
```

---

## Orchestrator Pattern

```python
class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        # Composes research + synthesis + memo configs
        # Validates all upfront

    async def run(self, topic: str) -> PipelineResult:
        # Stage 1: research
        # Stage 2: synthesis
        # Stage 3: memo generation
        # Returns PipelineResult with timing + summaries
```

---

## CLI Entry Points

```bash
python -m agents "topic"              # General agent
python -m agents --local --json       # Local runner, JSON output
python -m agents.orchestrator.main --topic "AI" --output ./out --demo
```

---

## Key Design Principles

1. Modularity — each agent standalone + composable
2. Async-first — all tools async, sync wrapped
3. Structured data — Pydantic models throughout
4. Graceful degradation — managed API → local fallback; S3 → filesystem
5. Config as frozen dataclass — immutable, validated, env-driven
6. Tools as JSON schemas — dispatched by name, return JSON strings
7. Verbose stderr logging — tagged `[agent]`, `[orchestrator]`, etc.
