#!/usr/bin/env python3
"""
OphthoFlow Prior Authorization Agent

An AI-powered ophthalmology prior authorization agent built on the Anthropic SDK.
Uses Claude to parse patient records, check PA requirements against payer portals,
and draft preliminary PA request letters.

Usage:
    from ophthoflow_pa_agent.agent import OphthoFlowPAAgent

    agent = OphthoFlowPAAgent()
    result = await agent.process(
        "72-year-old male with wet AMD in the right eye, VA 20/200 OD. "
        "OCT shows subretinal fluid. Plan: Eylea injection. Insured by Aetna, "
        "member ID AET123456789."
    )
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import anthropic

from .payer_portal import PayerPortalAPI
from .payer_portal.models import PAStatus, UrgencyLevel
from .payer_portal.procedure_data import DIAGNOSIS_DESCRIPTIONS, PROCEDURE_CATALOG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-6"

PROCEDURE_LOOKUP: dict[str, str] = {
    "ranibizumab": "J2778",
    "lucentis": "J2778",
    "aflibercept": "J0178",
    "eylea": "J0178",
    "bevacizumab": "J9035",
    "avastin": "J9035",
    "faricimab": "J2503",
    "vabysmo": "J2503",
    "cataract surgery": "66984",
    "cataract extraction": "66984",
    "complex cataract": "66982",
    "vitrectomy retinal detachment": "67108",
    "retinal detachment repair": "67108",
    "complex vitrectomy": "67113",
    "scleral buckle": "67101",
    "slt": "65855",
    "laser trabeculoplasty": "65855",
    "prp": "67228",
    "panretinal photocoagulation": "67228",
    "epiretinal membrane peel": "67040",
    "membrane peel": "67040",
    "vitrectomy membrane peel": "67040",
}

PAYER_LOOKUP: dict[str, str] = {
    "aetna": "AETNA",
    "united": "UNITEDHEALTHCARE",
    "unitedhealthcare": "UNITEDHEALTHCARE",
    "uhc": "UNITEDHEALTHCARE",
    "cigna": "CIGNA",
    "bcbs": "BCBS",
    "blue cross": "BCBS",
    "blue cross blue shield": "BCBS",
    "medicare": "MEDICARE",
    "cms": "MEDICARE",
}


# ---------------------------------------------------------------------------
# Structured output type
# ---------------------------------------------------------------------------


@dataclass
class ParsedPatientRecord:
    """Structured patient record extracted from a free-text summary."""

    patient_name: Optional[str] = None
    patient_dob: Optional[str] = None
    diagnosis_description: Optional[str] = None
    diagnosis_codes: list[str] | None = None
    affected_eye: Optional[str] = None
    visual_acuity_od: Optional[str] = None
    visual_acuity_os: Optional[str] = None
    oct_findings: Optional[str] = None
    proposed_treatment: Optional[str] = None
    procedure_code: Optional[str] = None
    prior_treatments: list[str] | None = None
    symptoms_duration_days: Optional[int] = None
    functional_impairment: Optional[str] = None
    payer_name: Optional[str] = None
    payer_key: Optional[str] = None
    member_id: Optional[str] = None
    urgency: str = "routine"
    additional_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool implementations (plain functions, called by the agent loop)
# ---------------------------------------------------------------------------


def parse_patient_record(patient_record_text: str) -> dict[str, Any]:
    """
    Parse a patient record text summary into structured fields using Claude.
    """
    available_procedures = "\n".join(
        f"  {code}: {info['name']} — diagnoses: {', '.join(info['typical_diagnoses'])}"
        for code, info in PROCEDURE_CATALOG.items()
    )
    available_diagnoses = "\n".join(
        f"  {code}: {desc}" for code, desc in DIAGNOSIS_DESCRIPTIONS.items()
    )
    available_payers = ", ".join(PAYER_LOOKUP.keys())

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    Extract structured fields from this patient record for prior
                    authorization processing. Return ONLY a valid JSON object with
                    these fields (use null for missing values):

                    {{
                        "patient_name": "string or null",
                        "patient_dob": "YYYY-MM-DD or null",
                        "diagnosis_description": "string",
                        "diagnosis_codes": ["ICD-10 codes"],
                        "affected_eye": "OD/OS/OU or null",
                        "visual_acuity_od": "e.g. 20/200 or null",
                        "visual_acuity_os": "e.g. 20/25 or null",
                        "oct_findings": "string or null",
                        "proposed_treatment": "drug or procedure name",
                        "procedure_code": "CPT/HCPCS code",
                        "prior_treatments": ["list of prior treatments"] or null,
                        "symptoms_duration_days": integer or null,
                        "functional_impairment": "string or null",
                        "payer_name": "string or null",
                        "payer_key": "normalized payer key or null",
                        "member_id": "string or null",
                        "urgency": "routine/urgent/emergent",
                        "additional_notes": "string or null"
                    }}

                    Available procedure codes:
                    {available_procedures}

                    Available ICD-10 diagnosis codes:
                    {available_diagnoses}

                    Recognized payer names: {available_payers}

                    PATIENT RECORD:
                    {patient_record_text}
                """),
            }
        ],
    )

    response_text = response.content[0].text.strip()

    # Handle markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        response_text = "\n".join(lines)

    parsed = json.loads(response_text)

    # Normalize procedure code via lookup if not already a valid code
    if not parsed.get("procedure_code") and parsed.get("proposed_treatment"):
        treatment_lower = parsed["proposed_treatment"].lower()
        for keyword, code in PROCEDURE_LOOKUP.items():
            if keyword in treatment_lower:
                parsed["procedure_code"] = code
                break

    # Normalize payer key via lookup if not already set
    if not parsed.get("payer_key") and parsed.get("payer_name"):
        payer_lower = parsed["payer_name"].lower()
        for keyword, key in PAYER_LOOKUP.items():
            if keyword in payer_lower:
                parsed["payer_key"] = key
                break

    return parsed


def check_pa_requirements(procedure_code: str, payer_key: str) -> dict[str, Any]:
    """
    Check PA requirements against the mock payer portal.
    """
    api = PayerPortalAPI(seed=42)

    try:
        result = api.check_pa_requirement(procedure_code, payer_key)
    except ValueError as e:
        return {"error": str(e), "requires_pa": None}

    return {
        "procedure_code": result.procedure_code,
        "procedure_name": result.procedure_name,
        "payer_id": result.payer_id,
        "payer_name": result.payer_name,
        "requires_pa": result.requires_pa,
        "status": result.status.value,
        "required_documents": result.required_documents,
        "step_therapy_required": result.step_therapy_required,
        "step_therapy_details": result.step_therapy_details,
        "estimated_review_hours": result.estimated_review_hours,
        "max_units_per_auth": result.max_units_per_auth,
        "auth_duration_days": result.auth_duration_days,
        "approved_indications": result.approved_indications,
        "message": result.message,
    }


def draft_pa_letter(
    patient_record: dict[str, Any],
    pa_requirements: dict[str, Any],
    provider_name: str = "Dr. Smith, MD",
    provider_npi: str = "1234567890",
    practice_name: str = "Retina & Ophthalmology Associates",
    practice_address: str = "123 Medical Center Dr, Suite 200",
    practice_phone: str = "(555) 123-4567",
    practice_fax: str = "(555) 123-4568",
) -> dict[str, str]:
    """
    Draft a preliminary PA request letter using Claude.
    """
    today = datetime.now().strftime("%B %d, %Y")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": textwrap.dedent(f"""\
                    Draft a professional prior authorization request letter for an
                    ophthalmology procedure. The letter should be clinically detailed,
                    reference specific diagnosis codes and medical necessity criteria,
                    and follow standard PA letter conventions.

                    PATIENT INFORMATION:
                    {json.dumps(patient_record, indent=2, default=str)}

                    PA REQUIREMENTS FROM PAYER:
                    {json.dumps(pa_requirements, indent=2, default=str)}

                    PROVIDER INFORMATION:
                    - Provider: {provider_name}
                    - NPI: {provider_npi}
                    - Practice: {practice_name}
                    - Address: {practice_address}
                    - Phone: {practice_phone}
                    - Fax: {practice_fax}
                    - Date: {today}

                    LETTER REQUIREMENTS:
                    1. Use formal medical letter format with letterhead block
                    2. Address to the payer's Medical Director / PA Department
                    3. Include patient name, DOB, member ID, and diagnosis codes
                    4. State the requested procedure with CPT/HCPCS code
                    5. Provide clinical justification referencing:
                       - Current examination findings (VA, OCT, etc.)
                       - Diagnosis and clinical rationale
                       - Why the proposed treatment is medically necessary
                       - Prior treatments attempted (if any)
                       - Functional impact on the patient
                    6. Reference the payer's approved indications to show criteria are met
                    7. Address step therapy requirements if applicable
                    8. List attached/enclosed supporting documentation
                    9. Include urgency level if urgent/emergent
                    10. Close with provider signature block

                    Return ONLY the letter text, no additional commentary.
                """),
            }
        ],
    )

    letter_text = response.content[0].text.strip()

    procedure_name = pa_requirements.get("procedure_name", "Unknown procedure")
    payer_name = pa_requirements.get("payer_name", "Unknown payer")
    patient_name = patient_record.get("patient_name", "Patient")
    summary = (
        f"PA request letter drafted for {patient_name}: "
        f"{procedure_name} via {payer_name}. "
        f"Letter includes clinical justification, diagnosis codes, "
        f"and addresses all payer-required documentation."
    )

    return {
        "letter_text": letter_text,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Tool definitions for the Anthropic SDK tool-use API
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "parse_patient_record",
        "description": (
            "Parse a free-text patient record summary into structured fields for "
            "prior authorization processing. Extracts diagnosis, proposed treatment, "
            "clinical findings, payer details, and other relevant information. "
            "Use this tool first when given a patient record or clinical note."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_record_text": {
                    "type": "string",
                    "description": "Free-text patient record or clinical note summary.",
                },
            },
            "required": ["patient_record_text"],
        },
    },
    {
        "name": "check_pa_requirements",
        "description": (
            "Check prior authorization requirements for a specific procedure and "
            "payer combination. Returns whether PA is required, required documents, "
            "step therapy requirements, estimated review time, and approved indications. "
            "Use this after parsing the patient record to determine PA needs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "procedure_code": {
                    "type": "string",
                    "description": 'CPT or HCPCS code (e.g., "J2778", "66984").',
                },
                "payer_key": {
                    "type": "string",
                    "description": 'Payer identifier (e.g., "AETNA", "UNITEDHEALTHCARE", "MEDICARE").',
                },
            },
            "required": ["procedure_code", "payer_key"],
        },
    },
    {
        "name": "draft_pa_letter",
        "description": (
            "Draft a preliminary prior authorization request letter based on the "
            "parsed patient record and PA requirements. Generates a professional, "
            "clinically detailed letter ready for provider review and signature. "
            "Use this after checking PA requirements to produce the final deliverable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_record": {
                    "type": "object",
                    "description": "Parsed patient record (output from parse_patient_record).",
                },
                "pa_requirements": {
                    "type": "object",
                    "description": "PA requirements (output from check_pa_requirements).",
                },
                "provider_name": {
                    "type": "string",
                    "description": "Treating physician's name and credentials.",
                    "default": "Dr. Smith, MD",
                },
                "provider_npi": {
                    "type": "string",
                    "description": "Provider's NPI number.",
                    "default": "1234567890",
                },
                "practice_name": {
                    "type": "string",
                    "description": "Name of the medical practice.",
                    "default": "Retina & Ophthalmology Associates",
                },
                "practice_address": {
                    "type": "string",
                    "description": "Practice address.",
                    "default": "123 Medical Center Dr, Suite 200",
                },
                "practice_phone": {
                    "type": "string",
                    "description": "Practice phone number.",
                    "default": "(555) 123-4567",
                },
                "practice_fax": {
                    "type": "string",
                    "description": "Practice fax number.",
                    "default": "(555) 123-4568",
                },
            },
            "required": ["patient_record", "pa_requirements"],
        },
    },
]

# Map tool names to their implementations
TOOL_DISPATCH: dict[str, callable] = {
    "parse_patient_record": lambda **kwargs: parse_patient_record(**kwargs),
    "check_pa_requirements": lambda **kwargs: check_pa_requirements(**kwargs),
    "draft_pa_letter": lambda **kwargs: draft_pa_letter(**kwargs),
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are OphthoFlow, an expert ophthalmology prior authorization (PA) agent.
    Your role is to help ophthalmology practices efficiently process prior
    authorization requests by:

    1. PARSING patient records — Extract structured clinical information from
       free-text notes, identifying diagnoses, proposed treatments, payer details,
       and relevant clinical findings.

    2. CHECKING PA requirements — Query the payer portal to determine if prior
       authorization is needed, what documentation is required, and whether
       step therapy or other conditions apply.

    3. DRAFTING PA letters — Generate professional, clinically detailed prior
       authorization request letters that address payer-specific requirements
       and demonstrate medical necessity.

    WORKFLOW:
    When given a patient record or PA request:
    1. First use parse_patient_record to extract structured fields
    2. Then use check_pa_requirements with the procedure code and payer
    3. Finally use draft_pa_letter to generate the request letter

    Always explain your findings clearly, flag any issues (missing info,
    step therapy requirements, documentation gaps), and provide actionable
    next steps for the practice staff.

    You are knowledgeable about:
    - Ophthalmology procedures: anti-VEGF injections, cataract surgery,
      retinal detachment repair, glaucoma procedures, retinal laser
    - ICD-10 diagnosis coding for eye conditions
    - CPT/HCPCS procedure coding
    - Payer-specific PA policies and step therapy requirements
    - Medical necessity documentation standards
""")


# ---------------------------------------------------------------------------
# Agent class — implements agentic tool-use loop via the Anthropic SDK
# ---------------------------------------------------------------------------


class OphthoFlowPAAgent:
    """
    OphthoFlow Prior Authorization Agent.

    An AI agent that automates the ophthalmology prior authorization workflow:
    1. Parses free-text patient records into structured data
    2. Checks PA requirements against payer portals
    3. Drafts preliminary PA request letters

    Built on the Anthropic Python SDK using Claude's tool-use capabilities.

    Usage:
        agent = OphthoFlowPAAgent()

        # Process a patient record end-to-end
        result = await agent.process(
            "72-year-old male with wet AMD, VA 20/200 OD. "
            "Plan: Eylea injection. Payer: Aetna, ID: AET123."
        )
        print(result["response"])  # Agent's final summary
        print(result["tool_results"])  # All tool call results

        # Or run interactively
        await agent.run()
    """

    def __init__(
        self,
        model: str = MODEL,
        max_turns: int = 10,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_turns = max_turns

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call and return the JSON-serialized result."""
        handler = TOOL_DISPATCH.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**tool_input)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    async def process(self, patient_text: str) -> dict[str, Any]:
        """
        Process a patient record through the full PA workflow using the
        agentic tool-use loop.

        The agent autonomously decides which tools to call and in what order,
        then returns a structured result with its final response and all
        intermediate tool results.

        Args:
            patient_text: Free-text patient record or clinical note summary.

        Returns:
            Dictionary with:
            - response: The agent's final text response summarizing findings.
            - tool_results: Dict mapping tool names to their outputs.
            - messages: The full conversation history (for debugging).
        """
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Process the following patient record through the full PA "
                    f"workflow (parse → check requirements → draft letter if "
                    f"PA is required):\n\n{patient_text}"
                ),
            }
        ]

        tool_results: dict[str, Any] = {}

        for _ in range(self.max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # If the model is done (no more tool calls), return the result
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                return {
                    "response": final_text,
                    "tool_results": tool_results,
                    "messages": messages,
                }

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_use_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    result_str = self._execute_tool(block.name, block.input)
                    tool_results[block.name] = json.loads(result_str)
                    tool_use_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )

            if tool_use_results:
                messages.append({"role": "user", "content": tool_use_results})

        # If we hit the turn limit, return what we have
        return {
            "response": "[Agent reached maximum turns without completing.]",
            "tool_results": tool_results,
            "messages": messages,
        }

    async def run(self) -> None:
        """
        Run the agent in interactive conversational mode.

        Reads user input from stdin and responds using the agentic tool-use
        loop. Type 'quit' or 'exit' to end the session.
        """
        print("=" * 70)
        print("  OphthoFlow PA Agent — Interactive Mode")
        print("  Type a patient record or question. Type 'quit' to exit.")
        print("=" * 70)

        messages: list[dict[str, Any]] = []

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            for _ in range(self.max_turns):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )

                if response.stop_reason == "end_turn":
                    assistant_text = ""
                    for block in response.content:
                        if block.type == "text":
                            assistant_text += block.text
                    messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                    print(f"\nOphthoFlow: {assistant_text}")
                    break

                # Handle tool use
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_use_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        print(f"\n  [Calling tool: {block.name}...]")
                        result_str = self._execute_tool(block.name, block.input)
                        tool_use_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_str,
                            }
                        )

                if tool_use_results:
                    messages.append({"role": "user", "content": tool_use_results})


# ---------------------------------------------------------------------------
# Convenience function for direct pipeline usage (no agentic loop)
# ---------------------------------------------------------------------------


async def process_patient_record(
    patient_text: str,
    provider_name: str = "Dr. Smith, MD",
    provider_npi: str = "1234567890",
) -> dict[str, Any]:
    """
    Process a patient record through the full PA workflow.

    This is a convenience function that runs the three tools sequentially
    without the agent loop, useful for scripting and testing.

    Args:
        patient_text: Free-text patient record summary.
        provider_name: Treating physician's name.
        provider_npi: Provider NPI number.

    Returns:
        Dictionary with parsed_record, pa_requirements, and pa_letter keys.
    """
    # Step 1: Parse the patient record
    parsed = parse_patient_record(patient_text)

    # Step 2: Check PA requirements
    procedure_code = parsed.get("procedure_code")
    payer_key = parsed.get("payer_key")

    if not procedure_code:
        return {
            "parsed_record": parsed,
            "error": "Could not determine procedure code from patient record.",
        }
    if not payer_key:
        return {
            "parsed_record": parsed,
            "error": "Could not determine payer from patient record.",
        }

    requirements = check_pa_requirements(procedure_code, payer_key)

    # Step 3: Draft PA letter (only if PA is required)
    pa_letter = None
    if requirements.get("requires_pa"):
        letter_result = draft_pa_letter(
            patient_record=parsed,
            pa_requirements=requirements,
            provider_name=provider_name,
            provider_npi=provider_npi,
        )
        pa_letter = letter_result

    return {
        "parsed_record": parsed,
        "pa_requirements": requirements,
        "pa_letter": pa_letter,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import sys

    SAMPLE_RECORD = textwrap.dedent("""\
        Patient: Margaret Thompson, DOB: 1951-06-14
        Visit Date: 2026-04-28

        Chief Complaint: Decreased vision OD x 3 months, progressive.

        History: 74-year-old female with known wet age-related macular
        degeneration (wAMD) in the right eye, diagnosed 6 months ago.
        Previously received 3 monthly injections of bevacizumab (Avastin)
        with partial response — subretinal fluid persists on OCT.

        Examination:
        - VA OD: 20/100 (corrected), OS: 20/25 (corrected)
        - IOP: 14 mmHg OD, 15 mmHg OS
        - Anterior segment: Normal OU
        - Fundus OD: Subfoveal CNV with subretinal hemorrhage, subretinal fluid
        - Fundus OS: Few small drusen, no CNV
        - OCT OD: Central subfield thickness 385 microns, persistent SRF
          and intraretinal fluid despite 3 prior bevacizumab injections

        Assessment & Plan:
        1. Wet AMD OD (H35.32) — incomplete response to bevacizumab
        2. Switch to aflibercept (Eylea) 2mg intravitreal injection OD
        3. Treat-and-extend protocol planned
        4. PA required — patient insured by Aetna, Member ID: AET887654321

        Treating Physician: Dr. Sarah Chen, MD
        NPI: 1987654321
        Practice: Pacific Retina Specialists
    """)

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        async def run_interactive():
            pa_agent = OphthoFlowPAAgent()
            await pa_agent.run()

        asyncio.run(run_interactive())
    else:
        async def run_sample():
            print("=" * 70)
            print("OphthoFlow PA Agent — Sample Pipeline Run")
            print("=" * 70)

            result = await process_patient_record(
                SAMPLE_RECORD,
                provider_name="Dr. Sarah Chen, MD",
                provider_npi="1987654321",
            )

            print("\n--- Parsed Patient Record ---")
            print(json.dumps(result["parsed_record"], indent=2, default=str))

            print("\n--- PA Requirements ---")
            print(json.dumps(result["pa_requirements"], indent=2, default=str))

            if result.get("pa_letter"):
                print("\n--- PA Request Letter ---")
                print(result["pa_letter"]["letter_text"])
                print("\n--- Summary ---")
                print(result["pa_letter"]["summary"])
            elif result.get("error"):
                print(f"\n--- Error ---\n{result['error']}")
            else:
                print("\n--- No PA Required ---")
                print("Prior authorization is not required for this procedure/payer.")

            print("\n" + "=" * 70)

        asyncio.run(run_sample())
