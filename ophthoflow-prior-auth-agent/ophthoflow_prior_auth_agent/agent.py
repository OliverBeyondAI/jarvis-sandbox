"""
Core agent loop for the OphthoFlow Prior Auth Agent.

Uses the Anthropic Python SDK with Claude's tool-use capabilities,
configured for AWS Bedrock or direct API access.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from .config import (
    BEDROCK_REGION,
    MAX_TOKENS,
    MODEL_BEDROCK,
    MODEL_DIRECT,
    USE_BEDROCK,
)
from .tools import TOOL_DISPATCH, TOOL_SCHEMAS

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are OphthoFlow, an AI-powered ophthalmology prior authorization agent.

You process patient cases (provided as structured JSON) through a complete PA workflow:

## Workflow Steps

1. **PARSE** the patient JSON to extract demographics, insurance, diagnoses (ICD-10), \
procedures (CPT/HCPCS), clinical findings, imaging, and prior treatments. \
Use the `parse_patient_json` tool.

2. **CHECK** payer-specific PA requirements using `check_pa_requirements`. Provide the \
payer name, CPT code, ICD-10 codes, and prior treatment names from the parsed data.

3. **ANALYZE** for missing information using `analyze_missing_information`. Identify gaps \
in the clinical record that could delay or prevent PA approval.

4. **ASSESS** denial risk using `assess_denial_risk`. Evaluate step therapy compliance, \
clinical criteria coverage, documentation completeness, and payer-specific risk factors.

5. **DRAFT** a professional PA request letter using `draft_pa_letter` if PA is required. \
Include all clinical justification, diagnosis codes, treatment history, and provider details.

## Guidelines

- Always call tools in sequence: parse → check → analyze → assess → draft.
- Extract ALL ICD-10 and CPT codes from the patient data.
- When checking PA requirements, always pass the ICD-10 codes and prior treatment names.
- Identify step therapy requirements early — they are the #1 cause of denials.
- Flag missing documentation that payers commonly require (OCT imaging, visual acuity, etc.).
- If PA is NOT required (e.g., Medicare, bevacizumab), clearly state so and skip the letter.
- After running all tools, provide a clear summary with:
  - PA determination (required / not required)
  - Risk level and key risk factors
  - Missing information that should be addressed
  - The draft PA letter (if applicable)
  - Specific recommendations to strengthen the submission
- Use a professional, clinical tone throughout.
"""


# ---------------------------------------------------------------------------
# Client Factory
# ---------------------------------------------------------------------------

def create_client() -> tuple[anthropic.Anthropic, str]:
    """Create an Anthropic client. Returns (client, model_id)."""
    if USE_BEDROCK:
        client = anthropic.AnthropicBedrock(aws_region=BEDROCK_REGION)
        return client, MODEL_BEDROCK
    return anthropic.Anthropic(), MODEL_DIRECT


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

def run_agent(
    user_message: str,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Run the PA agent loop: send a message, process tool calls, return final text.

    Args:
        user_message: The patient JSON or clinical query.
        client: Pre-configured Anthropic client (created if None).
        model: Model ID override.
        verbose: Print intermediate steps.

    Returns:
        The agent's final text response.
    """
    if client is None:
        client, model = create_client()
    if model is None:
        _, model = create_client()

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]

    iteration = 0
    max_iterations = 15  # Safety limit

    while iteration < max_iterations:
        iteration += 1
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Collect text blocks and tool-use blocks
        text_parts: list[str] = []
        tool_results: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                if verbose:
                    print(f"  [tool] {tool_name}({json.dumps(tool_input, indent=2)[:200]}...)")

                # Dispatch to implementation
                handler = TOOL_DISPATCH.get(tool_name)
                if handler:
                    try:
                        result = handler(**tool_input)
                    except Exception as e:
                        result = {"error": f"Tool execution failed: {e}"}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                if verbose:
                    result_str = json.dumps(result, indent=2)
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."
                    print(f"  [result] {result_str}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        # If no tool calls, we're done
        if not tool_results:
            return "\n".join(text_parts)

        # Append assistant message and tool results, then loop
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "\n".join(text_parts) + "\n\n[Agent reached maximum iterations]"


# ---------------------------------------------------------------------------
# Convenience: Run agent on a patient JSON file
# ---------------------------------------------------------------------------

def process_patient_case(
    patient_json: str | dict,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> str:
    """
    Process a patient case through the full PA workflow.

    Args:
        patient_json: Patient case as JSON string or dict.
        client: Pre-configured Anthropic client.
        model: Model ID override.
        verbose: Print intermediate steps.

    Returns:
        The agent's full PA analysis and draft letter.
    """
    if isinstance(patient_json, dict):
        patient_json = json.dumps(patient_json, indent=2, default=str)

    prompt = (
        "Process the following ophthalmology patient case through the complete "
        "prior authorization workflow. Parse the patient data, check PA requirements, "
        "analyze for missing information, assess denial risk, and draft a PA letter "
        "if required.\n\n"
        f"```json\n{patient_json}\n```"
    )

    return run_agent(prompt, client=client, model=model, verbose=verbose)
