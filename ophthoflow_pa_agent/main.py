#!/usr/bin/env python3
"""
OphthoFlow PA Agent — Main Entry Point

Accepts a patient record (from a fixture or stdin), runs the full PA workflow
(parse -> check requirements -> draft letter), and outputs the determination
and draft request.

Usage:
    # Run with a built-in example fixture (1, 2, or 3):
    python -m ophthoflow_pa_agent.main --example 1

    # Run all three example fixtures:
    python -m ophthoflow_pa_agent.main --all

    # Run with a custom patient record from a file:
    python -m ophthoflow_pa_agent.main --file patient_note.txt

    # Run with a patient record from stdin:
    echo "Patient: ..." | python -m ophthoflow_pa_agent.main --stdin

    # Run the interactive agent (conversational mode):
    python -m ophthoflow_pa_agent.main --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from typing import Any

from .fixtures import FIXTURES

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

DIVIDER = "=" * 72
SUBDIV = "-" * 72


def _header(title: str) -> str:
    return f"\n{SUBDIV}\n  {title}\n{SUBDIV}"


def print_result(result: dict[str, Any], label: str = "Patient") -> None:
    """Pretty-print the output of process_patient_record."""
    print(f"\n{DIVIDER}")
    print(f"  OphthoFlow PA Agent — {label}")
    print(DIVIDER)

    # ---- Parsed Record ----
    parsed = result.get("parsed_record", {})
    print(_header("PARSED PATIENT RECORD"))
    print(f"  Patient:      {parsed.get('patient_name', 'N/A')}")
    print(f"  DOB:          {parsed.get('patient_dob', 'N/A')}")
    print(f"  Diagnosis:    {parsed.get('diagnosis_description', 'N/A')}")
    dx_codes = parsed.get("diagnosis_codes") or []
    print(f"  ICD-10:       {', '.join(dx_codes) if dx_codes else 'N/A'}")
    print(f"  Eye:          {parsed.get('affected_eye', 'N/A')}")
    print(f"  VA OD / OS:   {parsed.get('visual_acuity_od', 'N/A')} / {parsed.get('visual_acuity_os', 'N/A')}")
    print(f"  OCT:          {parsed.get('oct_findings', 'N/A')}")
    print(f"  Treatment:    {parsed.get('proposed_treatment', 'N/A')} ({parsed.get('procedure_code', 'N/A')})")
    prior = parsed.get("prior_treatments") or []
    print(f"  Prior Tx:     {', '.join(prior) if prior else 'None'}")
    print(f"  Payer:        {parsed.get('payer_name', 'N/A')} (key: {parsed.get('payer_key', 'N/A')})")
    print(f"  Member ID:    {parsed.get('member_id', 'N/A')}")
    print(f"  Urgency:      {parsed.get('urgency', 'routine')}")

    # ---- PA Requirements ----
    reqs = result.get("pa_requirements")
    if reqs:
        print(_header("PA DETERMINATION"))
        requires = reqs.get("requires_pa")
        if requires is True:
            print("  ** PRIOR AUTHORIZATION REQUIRED **")
        elif requires is False:
            print("  Prior authorization is NOT required.")
        else:
            print(f"  Status unknown — {reqs.get('error', 'N/A')}")

        print(f"  Procedure:    {reqs.get('procedure_name', 'N/A')} ({reqs.get('procedure_code', 'N/A')})")
        print(f"  Payer:        {reqs.get('payer_name', 'N/A')}")
        print(f"  Review time:  {reqs.get('estimated_review_hours', 'N/A')} hours")
        print(f"  Auth valid:   {reqs.get('auth_duration_days', 'N/A')} days")
        print(f"  Max units:    {reqs.get('max_units_per_auth', 'N/A')}")

        docs = reqs.get("required_documents") or []
        if docs:
            print(f"  Required docs:")
            for doc in docs:
                print(f"    - {doc}")

        if reqs.get("step_therapy_required"):
            print(f"  Step therapy: YES — {reqs.get('step_therapy_details', '')}")

        indications = reqs.get("approved_indications") or []
        if indications:
            print(f"  Approved indications:")
            for ind in indications:
                print(f"    - {ind}")

        if reqs.get("message"):
            print(f"  Note: {reqs['message']}")

    # ---- PA Letter ----
    letter = result.get("pa_letter")
    if letter:
        print(_header("DRAFT PA REQUEST LETTER"))
        print(letter["letter_text"])
        print(f"\n  >> {letter['summary']}")
    elif result.get("error"):
        print(_header("ERROR"))
        print(f"  {result['error']}")
    elif reqs and not reqs.get("requires_pa"):
        print(_header("NO LETTER NEEDED"))
        print("  PA is not required — no letter was generated.")

    print(f"\n{DIVIDER}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _import_agent():
    """Lazy import to avoid requiring anthropic/claude_agent_sdk for --list."""
    from .agent import OphthoFlowPAAgent, process_patient_record
    return OphthoFlowPAAgent, process_patient_record


async def run_example(index: int) -> None:
    """Run a single example fixture through the PA pipeline."""
    _, process_patient_record = _import_agent()
    fixture = FIXTURES[index]
    print(f"\nLoading example fixture #{index + 1}: {fixture['label']}")
    result = await process_patient_record(
        fixture["record"],
        provider_name=fixture.get("provider_name", "Dr. Smith, MD"),
        provider_npi=fixture.get("provider_npi", "1234567890"),
    )
    print_result(result, label=fixture["label"])


async def run_all_examples() -> None:
    """Run all example fixtures sequentially."""
    for i in range(len(FIXTURES)):
        await run_example(i)


async def run_from_text(text: str, label: str = "Custom Record") -> None:
    """Run an arbitrary patient record through the PA pipeline."""
    _, process_patient_record = _import_agent()
    result = await process_patient_record(text)
    print_result(result, label=label)


async def run_interactive() -> None:
    """Launch the agent in conversational mode."""
    OphthoFlowPAAgent, _ = _import_agent()
    pa_agent = OphthoFlowPAAgent()
    await pa_agent.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ophthoflow_pa_agent",
        description="OphthoFlow PA Agent — Ophthalmology Prior Authorization Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m ophthoflow_pa_agent.main --example 1
              python -m ophthoflow_pa_agent.main --all
              python -m ophthoflow_pa_agent.main --file patient_note.txt
              cat note.txt | python -m ophthoflow_pa_agent.main --stdin
              python -m ophthoflow_pa_agent.main --interactive
        """),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--example", "-e",
        type=int,
        choices=range(1, len(FIXTURES) + 1),
        metavar="N",
        help=f"Run example fixture N (1-{len(FIXTURES)})",
    )
    group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all example fixtures",
    )
    group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a text file containing a patient record",
    )
    group.add_argument(
        "--stdin", "-s",
        action="store_true",
        help="Read patient record from stdin",
    )
    group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch the agent in conversational mode",
    )
    group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available example fixtures and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable example fixtures:\n")
        for i, fx in enumerate(FIXTURES):
            print(f"  {i + 1}. {fx['label']}")
            # Show first line of record as preview
            first_line = fx["record"].strip().split("\n")[0].strip()
            print(f"     {first_line}\n")
        return

    if args.interactive:
        asyncio.run(run_interactive())
    elif args.all:
        asyncio.run(run_all_examples())
    elif args.example:
        asyncio.run(run_example(args.example - 1))
    elif args.file:
        with open(args.file) as f:
            text = f.read()
        asyncio.run(run_from_text(text, label=args.file))
    elif args.stdin:
        text = sys.stdin.read()
        if not text.strip():
            print("Error: No input received from stdin.", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_from_text(text))


if __name__ == "__main__":
    main()
