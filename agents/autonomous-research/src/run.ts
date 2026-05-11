import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { runResearchAgent } from "./index.js";
import type { ResearchQuery } from "./types.js";

/**
 * Runnable entry point for the Autonomous Research Agent.
 *
 * Usage:
 *   npx tsx src/run.ts                              # Run default AI landscape research
 *   npx tsx src/run.ts "Topic" "Q1?" "Q2?" "Q3?"   # Custom topic + questions
 *   npx tsx src/run.ts --output report.md "Topic"   # Save report to file
 *
 * Environment:
 *   ANTHROPIC_API_KEY  — required
 *   TAVILY_API_KEY     — required
 */

interface CLIOptions {
  outputPath?: string;
  maxTurns: number;
  query: ResearchQuery;
}

function parseArgs(): CLIOptions {
  const args = process.argv.slice(2);
  let outputPath: string | undefined;
  let maxTurns = 20;
  const positional: string[] = [];

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--output" || args[i] === "-o") {
      outputPath = args[++i];
    } else if (args[i] === "--turns" || args[i] === "-t") {
      maxTurns = parseInt(args[++i], 10);
    } else if (args[i] === "--help" || args[i] === "-h") {
      printUsage();
      process.exit(0);
    } else {
      positional.push(args[i]);
    }
  }

  if (positional.length > 0) {
    const [topic, ...questions] = positional;
    return {
      outputPath,
      maxTurns,
      query: {
        topic,
        questions:
          questions.length > 0
            ? questions
            : [
                `What are the most important recent developments regarding: ${topic}?`,
                `Who are the key players and what are their strategies?`,
                `What are the technical and market implications?`,
                `What actionable opportunities or risks should decision-makers be aware of?`,
              ],
        maxDepth: 3,
      },
    };
  }

  // Default: AI landscape research
  return {
    outputPath,
    maxTurns,
    query: {
      topic: "The AI Landscape Pivots to Autonomous Systems",
      questions: [
        "What are the major AI labs shipping in terms of autonomous agent capabilities?",
        "How are enterprises adopting autonomous AI agents in production?",
        "What are the key technical breakthroughs enabling more autonomous AI systems?",
        "What are the safety and governance challenges of autonomous AI?",
        "What market signals indicate the shift from copilot to autonomous paradigms?",
      ],
      maxDepth: 3,
    },
  };
}

function printUsage(): void {
  console.log(`
  Autonomous Research Agent — Claude Opus 4.7 + Tavily Search

  USAGE
    npx tsx src/run.ts [options] [topic] [question1] [question2] ...

  OPTIONS
    -o, --output <path>   Save the report to a markdown file
    -t, --turns <n>       Max agent turns (default: 20)
    -h, --help            Show this help message

  EXAMPLES
    # Default AI landscape research
    npx tsx src/run.ts

    # Custom topic with auto-generated questions
    npx tsx src/run.ts "State of WebAssembly in 2026"

    # Custom topic with explicit questions, save to file
    npx tsx src/run.ts -o report.md "Rust in Production" \\
      "Which companies use Rust in production?" \\
      "What are the main adoption barriers?"

  ENVIRONMENT
    ANTHROPIC_API_KEY     Required — Claude API key
    TAVILY_API_KEY        Required — Tavily search API key
`);
}

function printBanner(query: ResearchQuery, maxTurns: number): void {
  const width = 60;
  console.log("");
  console.log("╔" + "═".repeat(width) + "╗");
  console.log(
    "║" + "  🔬 Autonomous Research Agent".padEnd(width) + "║",
  );
  console.log(
    "║" + "  Claude Opus 4.7 + Tavily Search".padEnd(width) + "║",
  );
  console.log("╠" + "═".repeat(width) + "╣");
  console.log(
    "║" + `  Topic: ${query.topic}`.substring(0, width).padEnd(width) + "║",
  );
  console.log(
    "║" +
      `  Questions: ${query.questions.length} | Max turns: ${maxTurns}`.padEnd(
        width,
      ) +
      "║",
  );
  console.log("╚" + "═".repeat(width) + "╝");
  console.log("");

  for (let i = 0; i < query.questions.length; i++) {
    console.log(`  ${i + 1}. ${query.questions[i]}`);
  }
  console.log("");
  console.log("─".repeat(width + 2));
}

async function main(): Promise<void> {
  // Validate env
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("Error: ANTHROPIC_API_KEY environment variable is required");
    process.exit(1);
  }
  if (!process.env.TAVILY_API_KEY) {
    console.error("Error: TAVILY_API_KEY environment variable is required");
    process.exit(1);
  }

  const { query, maxTurns, outputPath } = parseArgs();

  printBanner(query, maxTurns);

  const startTime = Date.now();

  const result = await runResearchAgent(query, {
    maxTurns,
    verbose: true,
  });

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  // Print results
  const width = 60;
  console.log("");
  console.log("═".repeat(width + 2));
  console.log("");
  console.log(result.markdown);
  console.log("");
  console.log("═".repeat(width + 2));
  console.log("");
  console.log(
    `  ✓ Completed in ${elapsed}s | ${result.turnsUsed} turns | ${result.structured ? "Structured" : "Unstructured"} output`,
  );

  // Save to file if requested
  if (outputPath) {
    const fullPath = resolve(outputPath);
    writeFileSync(fullPath, result.markdown, "utf-8");
    console.log(`  📁 Report saved to: ${fullPath}`);
  }

  console.log("");
}

main().catch((err) => {
  console.error("\n❌ Agent failed:", err.message ?? err);
  process.exit(1);
});
