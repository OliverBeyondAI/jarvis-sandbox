import Anthropic from "@anthropic-ai/sdk";
import { ALL_TOOLS, executeTool } from "./tools/registry.js";
import type { ResearchQuery } from "./types.js";

const MODEL = "claude-opus-4-7-20250715";
const MAX_TURNS = 20;

const anthropic = new Anthropic();

export interface ResearchResult {
  /** Formatted markdown report */
  markdown: string;
  /** Number of search iterations performed */
  turnsUsed: number;
  /** Whether the agent used the structured synthesis tool */
  structured: boolean;
}

/**
 * Autonomous Research Agent
 *
 * Uses Claude Opus 4.7 with a messages-API tool-use loop to autonomously
 * research topics via Tavily web search, iteratively gathering and
 * synthesizing information into a structured report with actionable insights.
 */
export async function runResearchAgent(
  query: ResearchQuery,
  options?: { maxTurns?: number; verbose?: boolean },
): Promise<ResearchResult> {
  const maxTurns = options?.maxTurns ?? MAX_TURNS;
  const verbose = options?.verbose ?? true;

  const systemPrompt = buildSystemPrompt();
  const userPrompt = buildResearchPrompt(query);

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userPrompt },
  ];

  let finalMarkdown = "";
  let structured = false;
  let turn = 0;

  for (; turn < maxTurns; turn++) {
    if (verbose) {
      process.stderr.write(`\n[Turn ${turn + 1}/${maxTurns}] `);
    }

    const response = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 16384,
      system: systemPrompt,
      tools: ALL_TOOLS,
      messages,
    });

    // Collect text and tool-use blocks from the response
    const toolUseBlocks: Anthropic.ContentBlockParam[] = [];
    const textParts: string[] = [];

    for (const block of response.content) {
      if (block.type === "text") {
        textParts.push(block.text);
        if (verbose) {
          const preview = block.text.substring(0, 120).replace(/\n/g, " ");
          process.stderr.write(`💭 ${preview}…\n`);
        }
      } else if (block.type === "tool_use") {
        toolUseBlocks.push(block);
        if (verbose) {
          const icon = block.name === "synthesize_report" ? "📄" : "🔧";
          process.stderr.write(
            `${icon} ${block.name}(${JSON.stringify(block.input).substring(0, 100)}…)\n`,
          );
        }
      }
    }

    // Append assistant message
    messages.push({ role: "assistant", content: response.content });

    // If the model stopped without tool use, we're done
    if (response.stop_reason === "end_turn" || toolUseBlocks.length === 0) {
      finalMarkdown = textParts.join("\n\n");
      if (verbose) {
        process.stderr.write("\n✓ Research complete\n");
      }
      break;
    }

    // Execute all tool calls and collect results
    const toolResults: Anthropic.ToolResultBlockParam[] = [];
    for (const block of toolUseBlocks) {
      if (block.type !== "tool_use") continue;

      const result = await executeTool(
        block.name,
        block.input as Record<string, unknown>,
      );

      // If the synthesis tool was called, capture its markdown output
      if (block.name === "synthesize_report") {
        finalMarkdown = result;
        structured = true;
        if (verbose) {
          process.stderr.write("📄 Report synthesized\n");
        }
      }

      toolResults.push({
        type: "tool_result",
        tool_use_id: block.id,
        content: result,
      });
    }

    messages.push({ role: "user", content: toolResults });

    // If we just synthesized, let the agent finish naturally on the next turn
  }

  return {
    markdown: finalMarkdown,
    turnsUsed: turn + 1,
    structured,
  };
}

function buildSystemPrompt(): string {
  return `You are an autonomous research agent. Your job is to thoroughly investigate a research topic by iteratively searching the web, extracting key information, and synthesizing your findings into a comprehensive report with actionable insights.

## Workflow

1. **Plan**: Break the research questions into specific search queries.
2. **Search**: Use tavily_search to find relevant, recent information. Use "advanced" search depth for important questions.
3. **Deep-dive**: Use tavily_extract on the most promising URLs to get full content.
4. **Iterate**: Based on what you learn, formulate follow-up searches to fill gaps or explore promising leads.
5. **Synthesize**: Once you have sufficient information, call the synthesize_report tool to produce the final structured report.

## Guidelines

- Focus on recent developments (2025-2026) and credible sources.
- Prioritize quantitative data, market signals, and concrete examples.
- Cross-reference claims across multiple sources for higher confidence.
- Search iteratively — each round of results should inform the next search.
- Aim for 3-5 search iterations before synthesizing.
- When you have gathered enough data, you MUST call the synthesize_report tool to produce the final output.
- Include 3-5 actionable insights: concrete recommendations with priority levels and timeframes.
- Each finding should have a confidence level (high/medium/low) based on source quality and corroboration.`;
}

function buildResearchPrompt(query: ResearchQuery): string {
  return `## Research Topic
${query.topic}

## Research Questions
${query.questions.map((q, i) => `${i + 1}. ${q}`).join("\n")}

## Depth
Perform up to ${query.maxDepth} rounds of iterative search to build a comprehensive understanding.

## Output
After completing your research, call the synthesize_report tool with your structured findings, conclusions, and actionable insights. Do NOT write the report as plain text — use the tool.

Please begin your research now.`;
}
