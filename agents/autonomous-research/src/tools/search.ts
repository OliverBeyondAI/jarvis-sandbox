import { tavily } from "@tavily/core";
import type Anthropic from "@anthropic-ai/sdk";

const client = tavily({ apiKey: process.env.TAVILY_API_KEY ?? "" });

export const tavilySearchDef: Anthropic.Tool = {
  name: "tavily_search",
  description:
    "Search the web for recent information on a topic. Returns relevant results with titles, URLs, and content snippets. Use 'advanced' search depth for thorough research on important questions.",
  input_schema: {
    type: "object" as const,
    properties: {
      query: { type: "string", description: "The search query" },
      search_depth: {
        type: "string",
        enum: ["basic", "advanced"],
        description: "basic for quick searches, advanced for thorough research",
      },
      topic: {
        type: "string",
        enum: ["general", "news"],
        description: "Topic type to optimize search results",
      },
      max_results: {
        type: "number",
        description: "Number of results to return (1-10, default 5)",
      },
    },
    required: ["query"],
  },
};

export const tavilyExtractDef: Anthropic.Tool = {
  name: "tavily_extract",
  description:
    "Extract the full content from one or more URLs. Use this to get detailed information from pages found via search.",
  input_schema: {
    type: "object" as const,
    properties: {
      urls: {
        type: "array",
        items: { type: "string" },
        description: "URLs to extract content from",
      },
    },
    required: ["urls"],
  },
};

export async function executeTavilySearch(input: {
  query: string;
  search_depth?: "basic" | "advanced";
  topic?: "general" | "news";
  max_results?: number;
}): Promise<string> {
  const response = await client.search(input.query, {
    searchDepth: input.search_depth ?? "basic",
    topic: input.topic ?? "general",
    maxResults: input.max_results ?? 5,
    includeAnswer: true,
  });

  return JSON.stringify(
    {
      answer: response.answer,
      results: response.results.map((r) => ({
        title: r.title,
        url: r.url,
        content: r.content,
        score: r.score,
      })),
    },
    null,
    2,
  );
}

export async function executeTavilyExtract(input: {
  urls: string[];
}): Promise<string> {
  const response = await client.extract(input.urls);

  return JSON.stringify(
    {
      results: response.results.map((r) => ({
        url: r.url,
        content: r.rawContent?.substring(0, 5000),
      })),
      failed: response.failedResults,
    },
    null,
    2,
  );
}
