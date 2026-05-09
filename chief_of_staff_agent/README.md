# Chief of Staff Research Agent

Multi-step autonomous research agent that produces executive briefings using Claude Opus and Tavily web search.

## Architecture

The agent uses the Anthropic Messages API with a **manual tool-use loop** (rather than the higher-level Claude Agent SDK) for full control over the research pipeline:

- **Fine-grained tool dispatch** — custom routing, logging, and source tracking per tool call
- **Multi-phase research protocol** — planning, broad search, deep dives, gap analysis, and report generation
- **Structured data collection** — Pydantic models track sources and findings throughout the pipeline
- **Streaming progress** — real-time stderr logging of each agent turn and tool invocation

The Claude Agent SDK (`claude_agent_sdk`) provides managed agent loops, but at the time of implementation it is not available as a standalone installable package. The manual approach also offers more transparency into the agent's decision-making process, which is valuable for a research prototype.

## Setup

### Prerequisites

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Tavily API key](https://tavily.com/) for web search

### Install

```bash
cd chief_of_staff_agent
pip install -e .

# Or install with dev dependencies for testing:
pip install -e ".[dev]"
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |
| `TAVILY_API_KEY` | Yes | Tavily API key for web search |
| `COS_OUTPUT_DIR` | No | Output directory for reports (default: `./chief_of_staff_reports`) |

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export TAVILY_API_KEY="tvly-..."
```

## Usage

### Free-form research

```bash
python -m chief_of_staff_agent "What is the current state of AI regulation in the EU and US?"
```

### Structured brief

```bash
python -m chief_of_staff_agent --brief "Competitive landscape for enterprise AI agents"
```

### Demo scenarios

```bash
# List available demos
python -m chief_of_staff_agent --list-demos

# Run a demo
python -m chief_of_staff_agent --demo ai_regulation
```

### Options

```
--brief, -b    Research brief (topic + questions)
--demo         Run a pre-built demo scenario (ai_regulation, ai_agents, semiconductor_supply)
--list-demos   List available demo scenarios
--output, -o   Output directory for reports
--model, -m    Override Claude model
--json, -j     Output result metadata as JSON
```

## Project Structure

```
chief_of_staff_agent/
  __init__.py     # Package metadata
  __main__.py     # python -m entry point
  main.py         # CLI argument parsing and runner
  agent.py        # Core agent with tool-use loop
  tools.py        # Tool definitions and async implementations
  models.py       # Pydantic models for briefs, findings, sources, reports
  config.py       # Configuration with env var defaults
  pyproject.toml  # Dependencies and packaging
  tests/          # Test suite
```

## How It Works

1. **Planning** — The agent analyzes the research brief and identifies 3-5 research angles
2. **Broad Research** — Executes multiple Tavily web searches covering different angles
3. **Deep Dives** — Fetches full article text from the most relevant sources using BeautifulSoup for HTML extraction
4. **Gap Analysis** — Reviews findings and runs additional searches to fill gaps
5. **Report Generation** — Synthesizes everything into a structured markdown executive briefing and saves to disk

## Running Tests

```bash
pytest chief_of_staff_agent/tests/ -v
```
