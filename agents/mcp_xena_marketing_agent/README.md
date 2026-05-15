# MCP Xena Marketing Agent

Autonomous marketing campaign generator powered by Claude Agent SDK, MCP tool servers, Tavily web search, and OpenAI image generation.

Xena researches markets, analyzes competitors, develops messaging strategy, generates multi-channel marketing content, and creates branded visuals — all autonomously through a 6-phase workflow.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Claude Agent SDK (Opus 4.7)               │
│              Extended thinking + agentic loop                │
├─────────────────────────────────────────────────────────────┤
│                      MCP Client Layer                       │
│            Tool discovery + schema negotiation              │
├──────────┬──────────┬───────────┬──────────┬───────────────┤
│  market  │ fetch    │ analyze   │  draft   │  generate     │
│ research │  _url    │ _market   │ _content │  _image       │
│ (Tavily) │ (httpx)  │ (struct)  │ (struct) │ (OpenAI)      │
└──────────┴──────────┴───────────┴──────────┴───────────────┘
```

**Workflow Phases:**
1. Market Research — web search for trends, competitors, audience insights
2. Competitor Analysis — deep-dive into competitor positioning
3. Messaging Strategy — key messages, value props, and positioning
4. Content Generation — draft content across marketing channels
5. Image Generation — AI visuals with embedded text and branding
6. Campaign Assembly — compile and save the complete campaign

## Setup

### Prerequisites

- Python 3.11+
- API keys (see below)

### Install Dependencies

```bash
cd agents/mcp_xena_marketing_agent
pip install -e .
```

Or install directly:

```bash
pip install anthropic>=0.97.0 mcp>=1.9.0 openai>=1.30.0 tavily-python>=0.5.0 httpx>=0.27.0 pydantic>=2.0.0 beautifulsoup4>=4.12.0
```

### API Keys

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API — powers the agent's reasoning |
| `TAVILY_API_KEY` | Recommended | Tavily — web search and market research |
| `OPENAI_API_KEY` | Optional | OpenAI — image generation (gpt-image-1) |

Set them in your shell:

```bash
export ANTHROPIC_API_KEY='sk-ant-api03-...'
export TAVILY_API_KEY='tvly-...'
export OPENAI_API_KEY='sk-...'
```

**Behavior when keys are missing:**
- `ANTHROPIC_API_KEY` — agent will not run (required)
- `TAVILY_API_KEY` — market research returns mock data instead of live results
- `OPENAI_API_KEY` — image generation is skipped; text content still generated

### Optional Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `XENA_OUTPUT_DIR` | `./marketing_output` | Directory for campaign output files |

## Usage

### Quick Start — Demo Script

The demo script validates your environment and runs pre-built campaign scenarios:

```bash
cd agents/mcp_xena_marketing_agent

# 1. Check environment and discover MCP tools (no API key needed)
python demo.py --check

# 2. Run the default demo (SaaS product launch)
python demo.py

# 3. Run a specific scenario
python demo.py --scenario saas_launch
python demo.py --scenario wellness_brand
python demo.py --scenario dev_tool

# 4. Customize the run
python demo.py --scenario dev_tool --thinking deep --html
python demo.py --scenario wellness_brand --no-images

# 5. Custom campaign brief
python demo.py --prompt "Launch campaign for our AI coding assistant" \
               --voice startup --no-images

# 6. Show programmatic usage example
python demo.py --example
```

### CLI Entry Point

```bash
# As a module (from repo root)
python -m agents.mcp_xena_marketing_agent --demo saas_launch
python -m agents.mcp_xena_marketing_agent --demo wellness_brand --voice luxury
python -m agents.mcp_xena_marketing_agent --demo dev_tool --thinking deep

# Custom campaign
python -m agents.mcp_xena_marketing_agent "Launch campaign for our AI writing tool"

# MCP tool discovery (no API key needed)
python -m agents.mcp_xena_marketing_agent --discover

# List available demos
python -m agents.mcp_xena_marketing_agent --list-demos
```

### Workflow Runner

```bash
cd agents/mcp_xena_marketing_agent

# Discover MCP tools
python run_workflow.py --discover

# Run a demo with HTML viewer
python run_workflow.py --demo saas_launch --html

# Dry-run (show config, no API calls)
python run_workflow.py --demo wellness_brand --dry-run
```

### Programmatic (Python)

```python
import asyncio
from mcp_xena_marketing_agent import MCPXenaMarketingAgent, Config, ProductInfo

async def main():
    config = Config.from_env(
        product=ProductInfo(
            name="MyProduct",
            tagline="AI for everyone",
            description="An AI-powered productivity tool",
            target_audience="Knowledge workers",
            key_features=["Smart automation", "Natural language"],
            competitors=["Notion", "Coda"],
        ),
        brand_voice_preset="startup",
        channels=["landing_page", "social_media"],
        thinking_preset="balanced",
        generate_images=False,
    )

    # Validate before running
    warnings = config.validate()
    for w in warnings:
        print(f"Warning: {w}")

    agent = MCPXenaMarketingAgent(config=config)
    result = await agent.run("Create a launch campaign for MyProduct")

    print(f"Campaign saved to: {result.campaign_path}")
    print(f"Content pieces: {result.draft_count}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    for piece in result.content_pieces:
        print(f"  [{piece['channel']}] {piece['title']}")

asyncio.run(main())
```

## Demo Scenarios

| Scenario | Product | Voice | Channels |
|---|---|---|---|
| `saas_launch` | FlowDesk — AI project management | startup | landing page, email, social, blog |
| `wellness_brand` | Solara — adaptogen supplements | luxury | landing page, email, social, ads |
| `dev_tool` | VectorForge — vector database | technical | landing page, blog, social, product |

## Configuration

### Brand Voice Presets

| Preset | Tone | Personality |
|---|---|---|
| `professional` | Authoritative, polished, confident | Trusted advisor |
| `startup` | Energetic, bold, conversational | Ambitious builder |
| `luxury` | Refined, aspirational, exclusive | Curator of excellence |
| `technical` | Precise, informative, credible | Expert engineer |

### Extended Thinking Presets

| Preset | Budget | Best For |
|---|---|---|
| `minimal` | 4,000 tokens | Quick iterations, simple campaigns |
| `balanced` | 10,000 tokens | Standard campaigns (default) |
| `deep` | 20,000 tokens | Complex multi-channel strategies |

### CLI Flags

| Flag | Description |
|---|---|
| `--demo <name>` | Run a pre-built demo scenario |
| `--voice <preset>` | Brand voice: professional, startup, luxury, technical |
| `--channels <list>` | Marketing channels to generate |
| `--thinking <preset>` | Thinking budget: minimal, balanced, deep |
| `--no-images` | Disable image generation |
| `--html` | Generate HTML campaign viewer |
| `--output <dir>` | Output directory |
| `--json` | Output result as JSON |
| `--discover` | List MCP tools (no API key needed) |

## MCP Tools

The agent connects to its MCP server and discovers 6 tools:

| Tool | Purpose |
|---|---|
| `market_research` | Web search via Tavily (trends, competitors, audience) |
| `fetch_url` | Read and extract content from web pages |
| `analyze_market` | Record structured analysis findings by phase |
| `draft_content` | Draft content for a specific marketing channel |
| `generate_image` | Create branded images with text overlay (OpenAI) |
| `save_campaign` | Save the compiled campaign as markdown |

## Output

Campaign runs produce:

```
marketing_output/
  campaign_flowdesk_20260515_143022.md   # Full campaign document
  demo_result.json                        # Structured result data
  campaign_viewer.html                    # Interactive HTML viewer (--html)
  images/
    social_media_20260515_143045.png      # Generated marketing images
    landing_hero_20260515_143052.png
```

## Project Structure

```
agents/mcp_xena_marketing_agent/
  __init__.py          # Package exports
  __main__.py          # Module entry point
  agent.py             # Core agent (Claude SDK + MCP client)
  config.py            # Configuration, brand voices, product info
  models.py            # Pydantic data models (Campaign, Content, etc.)
  tools.py             # Tool schema definitions
  mcp_server.py        # MCP tool server (FastMCP)
  main.py              # CLI entry point
  run_workflow.py       # Standalone workflow runner
  demo.py              # Interactive demo script
  viewer_gen.py        # HTML viewer generator
  viewer.html          # HTML viewer template
  verify_mcp_tavily.py # MCP + Tavily verification script
  pyproject.toml       # Package configuration
  README.md            # This file
```

## Troubleshooting

**"ANTHROPIC_API_KEY not set"**
Set the key: `export ANTHROPIC_API_KEY='sk-ant-...'`
Get one at [console.anthropic.com](https://console.anthropic.com/settings/keys).

**"MCP server connection failed"**
Ensure dependencies are installed (`pip install -e .`) and test the server directly: `python mcp_server.py`

**"Image generation skipped"**
Set `OPENAI_API_KEY` or use `--no-images` to suppress the warning.

**"Market research returned mock data"**
Set `TAVILY_API_KEY` for live web search results. Without it, the agent uses placeholder data.

**Rate limit errors**
Wait 30-60 seconds and retry. For sustained use, check your API plan limits.
