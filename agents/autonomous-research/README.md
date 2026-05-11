# Autonomous Research Agent

An autonomous research agent powered by **Claude Opus 4.7** and **Tavily** web search. It iteratively searches the web, extracts content, and synthesizes findings into structured reports.

## Architecture

```
src/
├── index.ts            # Agent loop (messages API + tool-use) and entry point
├── types.ts            # TypeScript interfaces (ResearchReport, Finding, etc.)
├── demo.ts             # Demo script with custom query support
└── tools/
    ├── registry.ts     # Tool definitions + execute_tool dispatcher
    ├── search.ts       # Tavily search & extract tools
    └── synthesis.ts    # Report builder and markdown formatter
```

### How It Works

1. **Messages API loop** — the agent sends a research prompt to Claude Opus 4.7, which decides what to search, when to extract full content, and when to synthesize.
2. **Tavily Search** — real-time web search with relevance scoring (`basic` or `advanced` depth).
3. **Tavily Extract** — pulls full page content from the most relevant URLs.
4. **Iterative refinement** — each round of search results informs follow-up queries, building comprehensive coverage over multiple turns.
5. **Final synthesis** — after gathering sufficient information, the agent writes a structured markdown report.

## Setup

```bash
npm install
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
# Run default research topic
npm run dev

# Run with a custom topic
npm run demo -- "What is the state of open-source AI models in 2026?"

# Production build
npm run build && npm start
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude Opus 4.7 |
| `TAVILY_API_KEY` | Tavily API key for web search |
