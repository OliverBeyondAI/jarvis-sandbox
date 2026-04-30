#!/usr/bin/env python3
"""
OphthoFlow PA Agent — Tavily-Powered Prior Authorization Research

A prototype Claude Opus 4.7 agent that, given a patient's ophthalmology
diagnosis and proposed treatment, uses Tavily API to research and summarize
the top 3 prior authorization requirements from common payers.

Outputs a structured JSON object containing the requirements.

Requirements:
    pip install anthropic tavily-python

Environment variables:
    ANTHROPIC_API_KEY  — Anthropic API key
    TAVILY_API_KEY     — Tavily API key

Usage:
    python ophthoflow_pa_agent.py
    python ophthoflow_pa_agent.py --diagnosis "Wet AMD" --treatment "Eylea injection"
    python ophthoflow_pa_agent.py --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from typing import Any

import anthropic

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-6"  # Latest Claude Opus model

# Common ophthalmology procedure code reference
PROCEDURE_CODES: dict[str, dict[str, str]] = {
    "eylea": {"code": "J0178", "generic": "aflibercept"},
    "aflibercept": {"code": "J0178", "generic": "aflibercept"},
    "lucentis": {"code": "J2778", "generic": "ranibizumab"},
    "ranibizumab": {"code": "J2778", "generic": "ranibizumab"},
    "avastin": {"code": "J9035", "generic": "bevacizumab"},
    "bevacizumab": {"code": "J9035", "generic": "bevacizumab"},
    "vabysmo": {"code": "J2503", "generic": "faricimab"},
    "faricimab": {"code": "J2503", "generic": "faricimab"},
    "cataract surgery": {"code": "66984", "generic": "phacoemulsification"},
    "vitrectomy": {"code": "67108", "generic": "vitrectomy"},
    "scleral buckle": {"code": "67101", "generic": "scleral buckle"},
    "prp laser": {"code": "67228", "generic": "panretinal photocoagulation"},
    "slt": {"code": "65855", "generic": "selective laser trabeculoplasty"},
    "membrane peel": {"code": "67040", "generic": "epiretinal membrane removal"},
}

# Common ophthalmology ICD-10 codes
DIAGNOSIS_CODES: dict[str, str] = {
    "wet amd": "H35.32",
    "wet age-related macular degeneration": "H35.32",
    "dry amd": "H35.31",
    "diabetic macular edema": "E11.311",
    "dme": "E11.311",
    "diabetic retinopathy": "E11.3211",
    "retinal detachment": "H33.001",
    "cataract": "H25.11",
    "glaucoma": "H40.11",
    "retinal vein occlusion": "H34.811",
    "rvo": "H34.811",
    "epiretinal membrane": "H35.371",
    "macular hole": "H35.341",
}

# Major payers to research
MAJOR_PAYERS = [
    "Aetna",
    "UnitedHealthcare",
    "Cigna",
    "Blue Cross Blue Shield",
    "Humana",
    "Medicare",
]


# ---------------------------------------------------------------------------
# Tavily search wrapper
# ---------------------------------------------------------------------------


def tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Search for prior authorization requirements using Tavily API.

    Args:
        query: Search query string focused on PA requirements.
        max_results: Maximum number of search results to return.

    Returns:
        Dictionary with search results including titles, URLs, and content.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "error": "TAVILY_API_KEY not set",
            "results": [],
            "fallback": True,
            "message": (
                "Tavily API key not configured. Using model knowledge for "
                "PA requirements. Set TAVILY_API_KEY for live web search."
            ),
        }

    if TavilyClient is None:
        return {
            "error": "tavily-python not installed",
            "results": [],
            "fallback": True,
            "message": (
                "tavily-python package not installed. Install with: "
                "pip install tavily-python"
            ),
        }

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
            })
        return {
            "answer": response.get("answer", ""),
            "results": results,
            "query": query,
            "fallback": False,
        }
    except Exception as e:
        return {
            "error": str(e),
            "results": [],
            "fallback": True,
            "message": f"Tavily search failed: {e}. Using model knowledge.",
        }


def lookup_procedure_code(treatment: str) -> dict[str, str]:
    """
    Look up the CPT/HCPCS procedure code for a given treatment.

    Args:
        treatment: Treatment name (e.g., "Eylea injection", "cataract surgery").

    Returns:
        Dictionary with code, generic name, and original treatment string.
    """
    treatment_lower = treatment.lower()
    for keyword, info in PROCEDURE_CODES.items():
        if keyword in treatment_lower:
            return {
                "treatment": treatment,
                "procedure_code": info["code"],
                "generic_name": info["generic"],
            }
    return {
        "treatment": treatment,
        "procedure_code": "unknown",
        "generic_name": treatment,
    }


def lookup_diagnosis_code(diagnosis: str) -> dict[str, str]:
    """
    Look up the ICD-10 diagnosis code for a given condition.

    Args:
        diagnosis: Diagnosis name (e.g., "Wet AMD", "diabetic macular edema").

    Returns:
        Dictionary with ICD-10 code and diagnosis description.
    """
    diagnosis_lower = diagnosis.lower()
    for keyword, code in DIAGNOSIS_CODES.items():
        if keyword in diagnosis_lower:
            return {
                "diagnosis": diagnosis,
                "icd10_code": code,
            }
    return {
        "diagnosis": diagnosis,
        "icd10_code": "unknown",
    }


# ---------------------------------------------------------------------------
# Tool definitions for Claude tool-use API
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "tavily_search",
        "description": (
            "Search the web using Tavily API for prior authorization requirements, "
            "payer policies, clinical criteria, and medical necessity guidelines "
            "related to ophthalmology procedures. Use this to research what each "
            "payer requires for a specific treatment. Craft specific search queries "
            "like 'Aetna prior authorization requirements aflibercept Eylea "
            "intravitreal injection ophthalmology 2025' for best results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Be specific: include payer name, drug/procedure "
                        "name, 'prior authorization', and 'ophthalmology'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum search results to return (default: 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_procedure_code",
        "description": (
            "Look up the CPT/HCPCS procedure code for an ophthalmology treatment. "
            "Supports anti-VEGF injections (Eylea, Lucentis, Avastin, Vabysmo), "
            "cataract surgery, vitrectomy, scleral buckle, PRP laser, SLT, and "
            "membrane peel."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "treatment": {
                    "type": "string",
                    "description": "Treatment or drug name (e.g., 'Eylea injection').",
                },
            },
            "required": ["treatment"],
        },
    },
    {
        "name": "lookup_diagnosis_code",
        "description": (
            "Look up the ICD-10 diagnosis code for an ophthalmology condition. "
            "Supports wet/dry AMD, diabetic macular edema, diabetic retinopathy, "
            "retinal detachment, cataract, glaucoma, RVO, epiretinal membrane, "
            "and macular hole."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "Diagnosis name (e.g., 'Wet AMD').",
                },
            },
            "required": ["diagnosis"],
        },
    },
]

# Tool dispatch table
TOOL_DISPATCH: dict[str, callable] = {
    "tavily_search": lambda **kwargs: tavily_search(**kwargs),
    "lookup_procedure_code": lambda **kwargs: lookup_procedure_code(**kwargs),
    "lookup_diagnosis_code": lambda **kwargs: lookup_diagnosis_code(**kwargs),
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are OphthoFlow, an expert ophthalmology prior authorization (PA) research
    agent. Your job is to research and compile prior authorization requirements
    for a given ophthalmology diagnosis and proposed treatment.

    WORKFLOW:
    1. Use lookup_procedure_code to identify the CPT/HCPCS code for the treatment.
    2. Use lookup_diagnosis_code to identify the ICD-10 code for the diagnosis.
    3. Use tavily_search to research PA requirements from the TOP 3 most relevant
       payers. Run separate, targeted searches for each payer. Example queries:
       - "Aetna prior authorization requirements aflibercept Eylea intravitreal
         injection wet AMD ophthalmology clinical policy"
       - "UnitedHealthcare prior auth criteria anti-VEGF injection macular
         degeneration medical necessity"
       - "Cigna precertification ophthalmology intravitreal injection policy"
    4. After gathering information from your searches, synthesize the findings into
       a structured JSON output.

    OUTPUT FORMAT:
    After completing your research, output a final JSON object with this structure:

    {
        "diagnosis": {
            "description": "Full diagnosis description",
            "icd10_code": "ICD-10 code"
        },
        "treatment": {
            "description": "Full treatment description",
            "procedure_code": "CPT/HCPCS code",
            "generic_name": "Generic drug/procedure name"
        },
        "payer_requirements": [
            {
                "payer_name": "Payer name",
                "requires_prior_auth": true/false,
                "key_requirements": [
                    "Requirement 1",
                    "Requirement 2"
                ],
                "step_therapy": "Description of step therapy if required, or null",
                "documentation_needed": [
                    "Document 1",
                    "Document 2"
                ],
                "clinical_criteria": "Summary of medical necessity criteria",
                "source_urls": ["URL references from search results"],
                "notes": "Any additional relevant information"
            }
        ],
        "summary": "Brief overall summary of PA landscape for this treatment",
        "researched_at": "ISO timestamp"
    }

    IMPORTANT GUIDELINES:
    - Always research at least 3 different payers
    - Be specific in search queries — include drug names, procedure types, and payer names
    - If Tavily search is unavailable, use your training knowledge to provide the best
      available information, but clearly note that results are based on model knowledge
      rather than live web search
    - Include source URLs when available from search results
    - Note any step therapy or fail-first requirements
    - Mention documentation requirements (OCT scans, VA records, prior treatment history)
    - Flag any time-sensitive requirements (auth expiration, review timelines)
""")


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class OphthoFlowPAResearchAgent:
    """
    OphthoFlow PA Research Agent — uses Claude + Tavily to research
    prior authorization requirements for ophthalmology procedures.
    """

    def __init__(self, model: str = MODEL, max_turns: int = 15):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_turns = max_turns

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call and return JSON-serialized result."""
        handler = TOOL_DISPATCH.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**tool_input)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def research(self, diagnosis: str, treatment: str) -> dict[str, Any]:
        """
        Research prior authorization requirements for a diagnosis/treatment pair.

        Args:
            diagnosis: Ophthalmology diagnosis (e.g., "Wet AMD").
            treatment: Proposed treatment (e.g., "Eylea intravitreal injection").

        Returns:
            Structured JSON object with PA requirements from top 3 payers.
        """
        user_prompt = (
            f"Research the prior authorization requirements for the following:\n\n"
            f"Diagnosis: {diagnosis}\n"
            f"Proposed Treatment: {treatment}\n\n"
            f"Find and summarize the top 3 payer PA requirements. "
            f"Output the final result as the specified JSON structure."
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt}
        ]

        print(f"\n{'='*70}")
        print(f"  OphthoFlow PA Research Agent")
        print(f"  Diagnosis:  {diagnosis}")
        print(f"  Treatment:  {treatment}")
        print(f"{'='*70}")
        print(f"\n  Researching PA requirements...\n")

        for turn in range(self.max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # If the model is done, extract the final response
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text

                # Try to extract JSON from the response
                result = self._extract_json(final_text)
                if result:
                    return result
                else:
                    return {
                        "raw_response": final_text,
                        "error": "Could not parse structured JSON from agent response.",
                    }

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_use_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    print(f"  [{block.name}] {self._summarize_input(block.name, block.input)}")
                    result_str = self._execute_tool(block.name, block.input)
                    tool_use_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            if tool_use_results:
                messages.append({"role": "user", "content": tool_use_results})

        return {"error": "Agent reached maximum turns without completing."}

    def _summarize_input(self, tool_name: str, tool_input: dict) -> str:
        """Create a short summary of a tool call for console output."""
        if tool_name == "tavily_search":
            return f"Searching: \"{tool_input.get('query', '')[:80]}...\""
        elif tool_name == "lookup_procedure_code":
            return f"Looking up code for: {tool_input.get('treatment', '')}"
        elif tool_name == "lookup_diagnosis_code":
            return f"Looking up ICD-10 for: {tool_input.get('diagnosis', '')}"
        return json.dumps(tool_input)[:100]

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract a JSON object from the agent's text response."""
        # Try to find JSON in code blocks first
        import re
        code_block_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find a JSON object directly
        brace_depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0 and start is not None:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        start = None

        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="OphthoFlow PA Research Agent — Research prior authorization requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python ophthoflow_pa_agent.py
              python ophthoflow_pa_agent.py --diagnosis "Wet AMD" --treatment "Eylea injection"
              python ophthoflow_pa_agent.py --diagnosis "Diabetic macular edema" --treatment "Avastin injection"
              python ophthoflow_pa_agent.py --interactive
        """),
    )
    parser.add_argument(
        "--diagnosis", "-d",
        type=str,
        help="Ophthalmology diagnosis (e.g., 'Wet AMD', 'Diabetic macular edema')",
    )
    parser.add_argument(
        "--treatment", "-t",
        type=str,
        help="Proposed treatment (e.g., 'Eylea injection', 'Avastin injection')",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Write JSON output to file",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    agent = OphthoFlowPAResearchAgent()

    if args.interactive:
        print("=" * 70)
        print("  OphthoFlow PA Research Agent — Interactive Mode")
        print("  Enter a diagnosis and treatment to research PA requirements.")
        print("  Type 'quit' to exit.")
        print("=" * 70)

        while True:
            try:
                print()
                diagnosis = input("  Diagnosis: ").strip()
                if diagnosis.lower() in ("quit", "exit", "q"):
                    break
                treatment = input("  Treatment: ").strip()
                if treatment.lower() in ("quit", "exit", "q"):
                    break
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not diagnosis or not treatment:
                print("  Please provide both a diagnosis and treatment.")
                continue

            result = agent.research(diagnosis, treatment)
            print(f"\n{'='*70}")
            print("  RESULTS")
            print(f"{'='*70}")
            print(json.dumps(result, indent=2, default=str))

    else:
        # Use provided args or default example
        diagnosis = args.diagnosis or "Wet age-related macular degeneration (wet AMD)"
        treatment = args.treatment or "Aflibercept (Eylea) intravitreal injection"

        result = agent.research(diagnosis, treatment)

        print(f"\n{'='*70}")
        print("  RESULTS")
        print(f"{'='*70}")
        print(json.dumps(result, indent=2, default=str))

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n  Results written to: {args.output}")

    print()


if __name__ == "__main__":
    main()
