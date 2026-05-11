import type Anthropic from "@anthropic-ai/sdk";
import type { ResearchReport } from "../types.js";

/**
 * Tool definition for synthesize_report — the agent calls this when it
 * has gathered enough information and is ready to produce a final report.
 */
export const synthesizeReportDef: Anthropic.Tool = {
  name: "synthesize_report",
  description:
    "Compile all research findings into a structured report with actionable insights. " +
    "Call this once you have completed your research and are ready to produce the final output. " +
    "The tool returns the formatted markdown report.",
  input_schema: {
    type: "object" as const,
    properties: {
      title: {
        type: "string",
        description: "Report title",
      },
      abstract: {
        type: "string",
        description:
          "2-3 sentence executive summary of the research findings",
      },
      sections: {
        type: "array",
        description: "Report sections, each covering a research question",
        items: {
          type: "object",
          properties: {
            title: { type: "string" },
            summary: {
              type: "string",
              description: "Brief summary of the section",
            },
            findings: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  claim: {
                    type: "string",
                    description: "The key finding or claim",
                  },
                  evidence: {
                    type: "string",
                    description:
                      "Supporting evidence, data points, or quotes",
                  },
                  confidence: {
                    type: "string",
                    enum: ["high", "medium", "low"],
                    description:
                      "Confidence level based on source quality and cross-referencing",
                  },
                  sourceUrls: {
                    type: "array",
                    items: { type: "string" },
                    description: "URLs backing this finding",
                  },
                },
                required: ["claim", "evidence", "confidence", "sourceUrls"],
              },
            },
          },
          required: ["title", "summary", "findings"],
        },
      },
      conclusions: {
        type: "array",
        items: { type: "string" },
        description: "3-5 high-level conclusions from the research",
      },
      actionableInsights: {
        type: "array",
        description:
          "Concrete, actionable recommendations derived from findings",
        items: {
          type: "object",
          properties: {
            recommendation: {
              type: "string",
              description: "What to do",
            },
            rationale: {
              type: "string",
              description: "Why — grounded in the research findings",
            },
            priority: {
              type: "string",
              enum: ["high", "medium", "low"],
            },
            timeframe: {
              type: "string",
              description:
                'When to act, e.g. "Immediate", "Next quarter", "6-12 months"',
            },
          },
          required: ["recommendation", "rationale", "priority", "timeframe"],
        },
      },
    },
    required: [
      "title",
      "abstract",
      "sections",
      "conclusions",
      "actionableInsights",
    ],
  },
};

/**
 * Execute the synthesis tool: builds a typed report and returns formatted markdown.
 */
export function executeSynthesizeReport(
  input: Record<string, unknown>,
): string {
  const report = buildReport(input);
  const markdown = reportToMarkdown(report);
  return markdown;
}

export function buildReport(args: Record<string, unknown>): ResearchReport {
  const raw = args as {
    title: string;
    abstract: string;
    sections: {
      title: string;
      summary: string;
      findings: {
        claim: string;
        evidence: string;
        confidence: "high" | "medium" | "low";
        sourceUrls: string[];
      }[];
    }[];
    conclusions: string[];
    actionableInsights: {
      recommendation: string;
      rationale: string;
      priority: "high" | "medium" | "low";
      timeframe: string;
    }[];
  };

  // Collect all unique sources across findings
  const allSourceUrls = new Set<string>();

  const sections = raw.sections.map((s) => ({
    title: s.title,
    summary: s.summary,
    findings: s.findings.map((f) => {
      f.sourceUrls.forEach((url) => allSourceUrls.add(url));
      return {
        claim: f.claim,
        evidence: f.evidence,
        confidence: f.confidence,
        sources: f.sourceUrls.map((url) => ({
          title: "",
          url,
          snippet: "",
          relevanceScore: 1,
        })),
      };
    }),
    sources: [],
  }));

  return {
    title: raw.title,
    abstract: raw.abstract,
    sections,
    conclusions: raw.conclusions,
    actionableInsights: raw.actionableInsights ?? [],
    sources: [...allSourceUrls].map((url) => ({
      title: "",
      url,
      snippet: "",
      relevanceScore: 1,
    })),
    generatedAt: new Date().toISOString(),
  };
}

export function reportToMarkdown(report: ResearchReport): string {
  const lines: string[] = [];

  // Header
  lines.push(`# ${report.title}`);
  lines.push("");
  lines.push(`> ${report.abstract}`);
  lines.push("");
  lines.push(`*Generated: ${new Date(report.generatedAt).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}*`);
  lines.push("");
  lines.push("---");
  lines.push("");

  // Table of Contents
  lines.push("## Table of Contents");
  lines.push("");
  report.sections.forEach((section, i) => {
    const anchor = section.title
      .toLowerCase()
      .replace(/[^a-z0-9 ]/g, "")
      .replace(/\s+/g, "-");
    lines.push(`${i + 1}. [${section.title}](#${anchor})`);
  });
  lines.push(`${report.sections.length + 1}. [Key Conclusions](#key-conclusions)`);
  lines.push(`${report.sections.length + 2}. [Actionable Insights](#actionable-insights)`);
  lines.push(`${report.sections.length + 3}. [Sources](#sources)`);
  lines.push("");
  lines.push("---");
  lines.push("");

  // Sections
  for (const section of report.sections) {
    lines.push(`## ${section.title}`);
    lines.push("");
    lines.push(section.summary);
    lines.push("");

    for (const finding of section.findings) {
      const badge =
        finding.confidence === "high"
          ? "🟢 High Confidence"
          : finding.confidence === "medium"
            ? "🟡 Medium Confidence"
            : "🔴 Low Confidence";

      lines.push(`### ${finding.claim}`);
      lines.push("");
      lines.push(`**${badge}**`);
      lines.push("");
      lines.push(finding.evidence);
      lines.push("");

      if (finding.sources.length > 0) {
        lines.push(
          `*Sources: ${finding.sources.map((s) => `[${new URL(s.url).hostname}](${s.url})`).join(", ")}*`,
        );
        lines.push("");
      }
    }
  }

  // Conclusions
  if (report.conclusions.length > 0) {
    lines.push("---");
    lines.push("");
    lines.push("## Key Conclusions");
    lines.push("");
    for (const conclusion of report.conclusions) {
      lines.push(`- ${conclusion}`);
    }
    lines.push("");
  }

  // Actionable Insights
  if (report.actionableInsights.length > 0) {
    lines.push("---");
    lines.push("");
    lines.push("## Actionable Insights");
    lines.push("");

    const priorityOrder = { high: 0, medium: 1, low: 2 };
    const sorted = [...report.actionableInsights].sort(
      (a, b) => priorityOrder[a.priority] - priorityOrder[b.priority],
    );

    for (const insight of sorted) {
      const icon =
        insight.priority === "high"
          ? "🔴"
          : insight.priority === "medium"
            ? "🟡"
            : "🟢";
      lines.push(
        `### ${icon} ${insight.recommendation}`,
      );
      lines.push("");
      lines.push(`**Priority:** ${insight.priority.charAt(0).toUpperCase() + insight.priority.slice(1)} | **Timeframe:** ${insight.timeframe}`);
      lines.push("");
      lines.push(insight.rationale);
      lines.push("");
    }
  }

  // Source index
  if (report.sources.length > 0) {
    lines.push("---");
    lines.push("");
    lines.push("## Sources");
    lines.push("");
    report.sources.forEach((src, i) => {
      lines.push(`${i + 1}. ${src.url}`);
    });
    lines.push("");
  }

  return lines.join("\n");
}
