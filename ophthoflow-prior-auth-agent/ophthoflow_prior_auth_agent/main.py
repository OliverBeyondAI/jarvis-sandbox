"""
CLI entry point for the OphthoFlow Prior Auth Agent.
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import create_client, process_patient_case, run_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OphthoFlow — AI-powered ophthalmology prior authorization agent",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a patient JSON file or clinical note.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read patient data from stdin.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        choices=["intravitreal_injections", "cataract_surgery", "retinal_imaging"],
        help="Run against a built-in sample dataset.",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Index of the case within the sample dataset (default: 0).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a local demo (no LLM) showing tool outputs for a sample case.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate tool calls and results.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use direct Anthropic API instead of Bedrock.",
    )
    args = parser.parse_args()

    # Demo mode — runs locally without LLM
    if args.demo:
        _run_demo(args.sample, args.case_index)
        return

    # Read input
    if args.stdin:
        content = sys.stdin.read()
    elif args.file:
        with open(args.file) as f:
            content = f.read()
    elif args.sample:
        from .sample_data.loader import load_samples

        cases = load_samples(args.sample)
        if args.case_index >= len(cases):
            print(f"Error: sample has {len(cases)} cases, index {args.case_index} out of range.", file=sys.stderr)
            sys.exit(1)
        content = json.dumps(cases[args.case_index].model_dump(mode="json"), indent=2, default=str)
    else:
        parser.print_help()
        print("\nProvide patient data via --file, --stdin, --sample, or --demo.")
        sys.exit(1)

    if not content.strip():
        print("Error: empty input.", file=sys.stderr)
        sys.exit(1)

    # Run the agent
    client, model = create_client()
    if args.direct:
        import anthropic

        client = anthropic.Anthropic()
        model = "claude-opus-4-7-20250501"

    result = process_patient_case(content, client=client, model=model, verbose=args.verbose)
    print(result)


def _run_demo(sample_name: str | None = None, case_index: int = 0) -> None:
    """Run a local demonstration of the PA workflow without calling the LLM."""
    from .sample_data.loader import load_samples
    from .tools import (
        analyze_missing_information,
        assess_denial_risk,
        draft_pa_letter,
        parse_patient_json,
        check_pa_requirements,
    )

    dataset = sample_name or "intravitreal_injections"
    cases = load_samples(dataset)
    if case_index >= len(cases):
        print(f"Error: sample has {len(cases)} cases, index {case_index} out of range.", file=sys.stderr)
        sys.exit(1)

    case_model = cases[case_index]
    case = case_model.model_dump(mode="json")
    case_json = json.dumps(case, indent=2, default=str)

    print("=" * 72)
    print("  OPHTHOFLOW PRIOR AUTHORIZATION AGENT — LOCAL DEMO")
    print("=" * 72)
    print(f"\nDataset: {dataset} | Case: {case.get('case_id', 'unknown')}")
    print(f"Completeness: {case.get('completeness', 'unknown')}")

    # Step 1: Parse
    print("\n" + "-" * 72)
    print("STEP 1: PARSE PATIENT JSON")
    print("-" * 72)
    parsed = parse_patient_json(case_json)
    print(f"  Patient: {parsed['patient_name']}")
    print(f"  DOB: {parsed['date_of_birth']}")
    print(f"  Payer: {parsed['payer']}")
    print(f"  Primary Diagnosis: {parsed['primary_diagnosis']}")
    print(f"  ICD-10 Codes: {', '.join(parsed['icd10_codes'])}")
    print(f"  Primary Procedure: {parsed['primary_procedure']}")
    print(f"  CPT Code: {parsed['primary_cpt']}")
    print(f"  Prior Treatments: {', '.join(parsed['prior_treatment_names']) or 'None'}")

    payer = parsed["payer"]
    cpt = parsed["primary_cpt"]
    icd10s = parsed["icd10_codes"]
    tx_names = parsed["prior_treatment_names"]

    # Step 2: Check PA requirements
    print("\n" + "-" * 72)
    print("STEP 2: CHECK PA REQUIREMENTS")
    print("-" * 72)
    pa_result = check_pa_requirements(
        payer=payer, cpt_code=cpt, icd10_codes=icd10s, prior_treatments=tx_names
    )
    print(f"  PA Required: {pa_result['pa_required']}")
    print(f"  Review Timeline: {pa_result['review_timeline_days']} days")
    print(f"  Step Therapy Met: {pa_result['step_therapy_met']}")
    if pa_result.get("step_therapy_details"):
        detail = pa_result["step_therapy_details"]
        if len(detail) > 120:
            detail = detail[:120] + "..."
        print(f"  Step Therapy Details: {detail}")
    if pa_result.get("approved_indications"):
        print("  Approved Indications:")
        for ind in pa_result["approved_indications"]:
            print(f"    - {ind}")
    if pa_result.get("required_documents"):
        print(f"  Required Documents: {len(pa_result['required_documents'])} items")

    # Step 3: Analyze missing info
    print("\n" + "-" * 72)
    print("STEP 3: ANALYZE MISSING INFORMATION")
    print("-" * 72)
    gaps = analyze_missing_information(case_json, payer=payer, cpt_code=cpt)
    print(f"  Overall: {gaps['overall_assessment']}")
    print(f"  Total Gaps: {gaps['total_gaps']} (Critical: {gaps['critical_gaps']}, High: {gaps['high_gaps']}, Moderate: {gaps['moderate_gaps']})")
    if gaps["gaps"]:
        print("  Gap Details:")
        for g in gaps["gaps"][:8]:
            print(f"    [{g['severity']}] {g['message']}")
        if len(gaps["gaps"]) > 8:
            print(f"    ... and {len(gaps['gaps']) - 8} more")

    # Step 4: Assess denial risk
    print("\n" + "-" * 72)
    print("STEP 4: ASSESS DENIAL RISK")
    print("-" * 72)
    risk = assess_denial_risk(
        payer=payer,
        cpt_code=cpt,
        patient_name=parsed["patient_name"],
        icd10_codes=icd10s,
        prior_treatments=tx_names,
        has_imaging=bool(parsed.get("imaging_studies")),
        has_visual_acuity=bool(parsed.get("visual_acuity")),
        step_therapy_met=pa_result.get("step_therapy_met"),
        completeness_level=parsed.get("completeness", "unknown"),
    )
    print(f"  Risk Level: {risk['risk_level']}")
    print(f"  Risk Score: {risk['risk_score']}/100")
    print(f"  Summary: {risk['summary']}")
    if risk.get("risk_factors"):
        print("  Risk Factors:")
        for rf in risk["risk_factors"]:
            print(f"    [{rf['impact']}] {rf['factor']}")
    if risk.get("recommendations"):
        print("  Recommendations:")
        for rec in risk["recommendations"]:
            if len(rec) > 100:
                rec = rec[:100] + "..."
            print(f"    - {rec}")

    # Step 5: Draft PA letter (only if PA is required)
    if pa_result["pa_required"]:
        print("\n" + "-" * 72)
        print("STEP 5: DRAFT PA LETTER")
        print("-" * 72)
        letter = draft_pa_letter(
            patient_name=parsed["patient_name"],
            date_of_birth=parsed["date_of_birth"],
            member_id=parsed.get("insurance_member_id", ""),
            diagnosis=parsed["primary_diagnosis"],
            icd10_codes=icd10s,
            procedure=parsed["primary_procedure"],
            cpt_code=cpt,
            payer=payer,
            clinical_findings=parsed.get("clinical_findings", ""),
            imaging_summary=parsed.get("imaging_summary", ""),
            prior_treatments=tx_names,
            step_therapy_met=pa_result.get("step_therapy_met"),
            provider_name=parsed.get("provider_name", ""),
            provider_npi=parsed.get("provider_npi", ""),
            practice_name=parsed.get("practice_name", ""),
            urgency=parsed.get("urgency", "routine"),
            requested_duration_months=parsed.get("requested_duration_months"),
            requested_total_doses=parsed.get("requested_total_doses"),
        )
        print(f"  Reference ID: {letter['reference_id']}")
        print(f"\n{letter['letter_text']}")
    else:
        print("\n" + "-" * 72)
        print("STEP 5: PA LETTER — SKIPPED")
        print("-" * 72)
        print(f"  PA is NOT required by {payer} for CPT {cpt}.")
        print("  Proceed with standard billing. Ensure medical necessity is documented in clinical notes.")

    print("\n" + "=" * 72)
    print("  DEMO COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
