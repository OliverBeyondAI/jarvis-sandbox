export interface ResearchQuery {
  topic: string;
  questions: string[];
  maxDepth: number;
}

export interface Source {
  title: string;
  url: string;
  snippet: string;
  relevanceScore: number;
  publishedDate?: string;
}

export interface Finding {
  claim: string;
  evidence: string;
  sources: Source[];
  confidence: "high" | "medium" | "low";
}

export interface ResearchSection {
  title: string;
  summary: string;
  findings: Finding[];
  sources: Source[];
}

export interface ActionableInsight {
  recommendation: string;
  rationale: string;
  priority: "high" | "medium" | "low";
  timeframe: string;
}

export interface ResearchReport {
  title: string;
  abstract: string;
  sections: ResearchSection[];
  conclusions: string[];
  actionableInsights: ActionableInsight[];
  sources: Source[];
  generatedAt: string;
}
