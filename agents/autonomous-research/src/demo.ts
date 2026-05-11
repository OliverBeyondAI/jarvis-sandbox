import { runResearchAgent } from "./index.js";
import type { ResearchQuery } from "./types.js";

/**
 * Demo: Run the autonomous research agent with a custom query.
 *
 * Usage:
 *   npx tsx src/demo.ts
 *   npx tsx src/demo.ts "What is the state of open-source AI models in 2026?"
 */

const customTopic = process.argv[2];

const query: ResearchQuery = customTopic
  ? {
      topic: customTopic,
      questions: [
        `What are the most important recent developments regarding: ${customTopic}?`,
        `Who are the key players and what are their strategies?`,
        `What are the technical and market implications?`,
      ],
      maxDepth: 2,
    }
  : {
      topic: "The AI Landscape Pivots to Autonomous Systems",
      questions: [
        "What are the major AI labs shipping in terms of autonomous agent capabilities?",
        "How are enterprises adopting autonomous AI agents in production?",
        "What market signals indicate the shift from copilot to autonomous paradigms?",
      ],
      maxDepth: 3,
    };

async function main() {
  console.log(`\n🔬 Research Agent Demo\n`);
  console.log(`Topic: ${query.topic}`);
  console.log(`Questions: ${query.questions.length}`);
  console.log("─".repeat(60));

  const result = await runResearchAgent(query, { verbose: true });

  console.log("\n" + "─".repeat(60));
  console.log("\n📄 REPORT\n");
  console.log(result.markdown);
}

main().catch(console.error);
