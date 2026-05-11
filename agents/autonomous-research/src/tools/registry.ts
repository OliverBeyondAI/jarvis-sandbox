import type Anthropic from "@anthropic-ai/sdk";
import {
  tavilySearchDef,
  tavilyExtractDef,
  executeTavilySearch,
  executeTavilyExtract,
} from "./search.js";
import {
  synthesizeReportDef,
  executeSynthesizeReport,
} from "./synthesis.js";

type ToolExecutor = (input: Record<string, unknown>) => Promise<string> | string;

const TOOL_EXECUTORS: Record<string, ToolExecutor> = {
  tavily_search: (input) =>
    executeTavilySearch(
      input as Parameters<typeof executeTavilySearch>[0],
    ),
  tavily_extract: (input) =>
    executeTavilyExtract(
      input as Parameters<typeof executeTavilyExtract>[0],
    ),
  synthesize_report: (input) => executeSynthesizeReport(input),
};

export const ALL_TOOLS: Anthropic.Tool[] = [
  tavilySearchDef,
  tavilyExtractDef,
  synthesizeReportDef,
];

export async function executeTool(
  name: string,
  input: Record<string, unknown>,
): Promise<string> {
  const executor = TOOL_EXECUTORS[name];
  if (!executor) {
    return JSON.stringify({ error: `Unknown tool: ${name}` });
  }

  try {
    return await executor(input);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return JSON.stringify({ error: message });
  }
}
