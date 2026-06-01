# Claude Sonnet 4.6 Model ID References — Audit Checklist

Audit of all Jarvis source files containing Claude Sonnet 4.6 model IDs.
Excludes `node_modules/` (SDK type definitions) and `.jarvis-task.md`.

All references have been upgraded from Sonnet 4.6 → Opus 4.7 (`claude-opus-4-7-20250501`).

---

## Bedrock Format (`anthropic.claude-sonnet-4-*`)

| # | File | Line | Current Value | Status |
|---|------|------|---------------|--------|
| 1 | `jarvis-transcriber/transcribe_endpoint.py` | 37 | `us.anthropic.claude-opus-4-7-20250501-v1:0` | [x] |

## Direct API Format (`claude-sonnet-4-*`)

| # | File | Line | Current Value | Status |
|---|------|------|---------------|--------|
| 2 | `agents/agent.py` | 29 | `claude-opus-4-7-20250501` | [x] |
| 3 | `agents/prior_auth_agent.py` | 43 | `claude-opus-4-7-20250501` | [x] |
| 4 | `agents/main.py` | 98 | `claude-opus-4-7-20250501` (in help text) | [x] |
| 5 | `autonomous_research_agent/config.py` | 16 | `claude-opus-4-7-20250501` | [x] |
| 6 | `autonomous_research_agent/main.py` | 231 | `claude-opus-4-7-20250501` (in help text) | [x] |
| 7 | `autonomous_research_agent/tests/test_config.py` | 16 | `claude-opus-4-7-20250501` | [x] |
| 8 | `autonomous_research_agent/tests/test_agent.py` | 26 | `claude-opus-4-7-20250501` | [x] |
| 9 | `autonomous_research_agent/tests/conftest.py` | 38 | `claude-opus-4-7-20250501` | [x] |
| 10 | `dispatch_agent/main.py` | 172 | `claude-opus-4-7-20250501` (in help text) | [x] |
| 11 | `mcp_agent_prototype/agent.py` | 43 | `claude-opus-4-7-20250501` | [x] |
| 12 | `mock_weather_mcp/run_demo.py` | 206 | `claude-opus-4-7-20250501` | [x] |
| 13 | `social_content_agent/feedback_loop.py` | 34 | `claude-opus-4-7-20250501` | [x] |
| 14 | `social_content_agent/agent.py` | 253 | `claude-opus-4-7-20250501` | [x] |
| 15 | `social_content_agent/main.py` | 201 | `claude-opus-4-7-20250501` | [x] |
| 16 | `social_content_agent/main.py` | 257 | `claude-opus-4-7-20250501` (default arg) | [x] |
| 17 | `social_content_agent/main.py` | 258 | `claude-opus-4-7-20250501` (in help text) | [x] |
| 18 | `xena_image_gen_prototype/evaluate.py` | 328 | `claude-opus-4-7-20250501` | [x] |

## Documentation References (`claude-sonnet-4-6` shorthand)

| # | File | Line | Current Value | Status |
|---|------|------|---------------|--------|
| 19 | `AGENT_SDK_RESEARCH.md` | 60 | `claude-opus-4-7` | [x] |
| 20 | `AGENT_SDK_RESEARCH.md` | 115 | `claude-opus-4-7` | [x] |
| 21 | `AGENT_SDK_RESEARCH.md` | 318 | `claude-opus-4-7` | [x] |
| 22 | `AGENT_SDK_RESEARCH.md` | 374 | `claude-opus-4-7` | [x] |
| 23 | `AGENT_SDK_RESEARCH.md` | 404 | `claude-opus-4-7` | [x] |

---

## Summary

- **Total references**: 23 (across 14 unique files)
- **All 23 references**: upgraded to Opus 4.7 ✓
- **Bedrock format**: upgraded to cross-region inference profile (`us.anthropic.claude-opus-4-7-20250501-v1:0`)
- **Direct API format**: upgraded to `claude-opus-4-7-20250501`
- **Documentation shorthand**: upgraded to `claude-opus-4-7`
- **Excluded**: `node_modules/@anthropic-ai/sdk/` type definitions (managed by SDK version)
