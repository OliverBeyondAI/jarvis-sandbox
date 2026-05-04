#!/usr/bin/env python3
"""
OphthoFlow Prior Auth Agent — Demo Runner

Runs the complete PA workflow locally (no LLM required) against all
sample patient cases, demonstrating parsing, payer rule checks,
missing information analysis, denial risk assessment, and PA letter drafting.

Usage:
    python run_demo.py                              # Run all demos
    python run_demo.py --dataset intravitreal_injections --case 0  # Specific case
    python run_demo.py --list                       # List available cases
"""

import argparse
import json
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from ophthoflow_prior_auth_agent.sample_data.loader import load_samples, AVAILABLE_DATASETS
from ophthoflow_prior_auth_agent.tools import (
    parse_patient_json,
    check_pa_requirements,
    analyze_missing_information,
    assess_denial_risk,
    draft_pa_letter,
)


def list_cases() -> None:
    """Print all available sample cases."""
    print("Available Sample Cases:")
    print("=" * 60)
    for dataset in AVAILABLE_DATASETS:
        cases = load_samples(dataset)
        print(f"\n  {dataset}:")
        for i, case in enumerate(cases):
            d = case.model_dump(mode="json")
            patient = d.get("patient", {})
            name = f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip()
            completeness = d.get("completeness", "?")
            case_id = d.get("case_id", "?")
            print(f"    [{i}] {case_id} — {name} ({completeness})")


def run_single_demo(dataset: str, case_index: int) -> None:
    """Run the full PA workflow for a single case."""
    cases = load_samples(dataset)
    if case_index >= len(cases):
        print(f"Error: {dataset} has {len(cases)} cases, index {case_index} is out of range.")
        sys.exit(1)

    case = cases[case_index].model_dump(mode="json")
    case_json = json.dumps(case, indent=2, default=str)

    patient = case.get("patient", {})
    name = f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip()

    print("\n" + "=" * 72)
    print(f"  CASE: {case.get('case_id', '?')} — {name}")
    print(f"  Dataset: {dataset} | Completeness: {case.get('completeness', '?')}")
    print("=" * 72)

    # Step 1: Parse
    print("\n" + "-" * 72)
    print("  STEP 1: PARSE PATIENT JSON")
    print("-" * 72)
    parsed = parse_patient_json(case_json)
    print(f"  Patient:           {parsed['patient_name']}")
    print(f"  DOB:               {parsed['date_of_birth']}")
    print(f"  Payer:             {parsed['payer']}")
    print(f"  Primary Diagnosis: {parsed['primary_diagnosis']}")
    print(f"  ICD-10 Codes:      {', '.join(parsed['icd10_codes'])}")
    print(f"  Primary Procedure: {parsed['primary_procedure']}")
    print(f"  CPT Code:          {parsed['primary_cpt']}")
    print(f"  Prior Treatments:  {', '.join(parsed['prior_treatment_names']) or 'None documented'}")
    if parsed.get("visual_acuity"):
        for va in parsed["visual_acuity"]:
            print(f"  Visual Acuity:     {va['eye']} = {va['best_corrected']} ({va.get('method', '')})")
    if parsed.get("imaging_summary"):
        summary = parsed["imaging_summary"]
        if len(summary) > 100:
            summary = summary[:100] + "..."
        print(f"  Imaging:           {summary}")

    payer = parsed["payer"]
    cpt = parsed["primary_cpt"]
    icd10s = parsed["icd10_codes"]
    tx_names = parsed["prior_treatment_names"]

    if not payer or not cpt:
        print("\n  [!] Insufficient data for PA workflow — missing payer or CPT code.")
        return

    # Step 2: Check PA Requirements
    print("\n" + "-" * 72)
    print("  STEP 2: CHECK PA REQUIREMENTS")
    print("-" * 72)
    pa_result = check_pa_requirements(
        payer=payer, cpt_code=cpt, icd10_codes=icd10s, prior_treatments=tx_names
    )
    pa_req = pa_result["pa_required"]
    print(f"  PA Required:       {'YES' if pa_req else 'NO'}")
    print(f"  Review Timeline:   {pa_result['review_timeline_days']} business days")
    st = pa_result.get("step_therapy_met")
    if st is not None:
        print(f"  Step Therapy Met:  {'YES' if st else 'NO'}")
    if pa_result.get("approved_indications"):
        print("  Approved Indications:")
        for ind in pa_result["approved_indications"][:4]:
            print(f"    - {ind}")
    if pa_result.get("required_documents"):
        print(f"  Required Documents ({len(pa_result['required_documents'])}):")
        for doc in pa_result["required_documents"][:5]:
            print(f"    - {doc}")
        remaining = len(pa_result["required_documents"]) - 5
        if remaining > 0:
            print(f"    ... and {remaining} more")
    if pa_result.get("notes"):
        notes = pa_result["notes"]
        if len(notes) > 150:
            notes = notes[:150] + "..."
        print(f"  Notes: {notes}")

    # Step 3: Analyze Missing Information
    print("\n" + "-" * 72)
    print("  STEP 3: ANALYZE MISSING INFORMATION")
    print("-" * 72)
    gaps_result = analyze_missing_information(case_json, payer=payer, cpt_code=cpt)
    print(f"  Assessment:    {gaps_result['overall_assessment']}")
    print(f"  Total Gaps:    {gaps_result['total_gaps']}")
    print(f"    Critical:    {gaps_result['critical_gaps']}")
    print(f"    High:        {gaps_result['high_gaps']}")
    print(f"    Moderate:    {gaps_result['moderate_gaps']}")
    if gaps_result["gaps"]:
        print("  Details:")
        for g in gaps_result["gaps"][:6]:
            print(f"    [{g['severity']:8s}] {g['message']}")
        if len(gaps_result["gaps"]) > 6:
            print(f"    ... and {len(gaps_result['gaps']) - 6} more")

    # Step 4: Assess Denial Risk
    print("\n" + "-" * 72)
    print("  STEP 4: ASSESS DENIAL RISK")
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
    level = risk["risk_level"]
    indicator = {"LOW": "[OK]", "MODERATE": "[!!]", "HIGH": "[XX]"}.get(level, "[??]")
    print(f"  Risk Level:    {indicator} {level}")
    print(f"  Risk Score:    {risk['risk_score']}/100")
    print(f"  Summary:       {risk['summary']}")
    if risk.get("risk_factors"):
        print("  Risk Factors:")
        for rf in risk["risk_factors"]:
            print(f"    [{rf['impact']:8s}] {rf['factor']}")
    if risk.get("recommendations"):
        print("  Recommendations:")
        for rec in risk["recommendations"]:
            if len(rec) > 100:
                rec = rec[:100] + "..."
            print(f"    -> {rec}")

    # Step 5: Draft PA Letter
    if pa_req:
        print("\n" + "-" * 72)
        print("  STEP 5: DRAFT PA LETTER")
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
        print()
        # Indent the letter for visual clarity
        for line in letter["letter_text"].split("\n"):
            print(f"  | {line}")
    else:
        print("\n" + "-" * 72)
        print("  STEP 5: PA LETTER — NOT NEEDED")
        print("-" * 72)
        print(f"  PA is not required by {payer} for CPT {cpt}.")
        print("  Standard billing can proceed with medical necessity documentation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OphthoFlow PA Agent — Demo Runner")
    parser.add_argument("--dataset", type=str, choices=list(AVAILABLE_DATASETS), help="Sample dataset to use")
    parser.add_argument("--case", type=int, default=0, help="Case index within dataset (default: 0)")
    parser.add_argument("--list", action="store_true", help="List all available sample cases")
    parser.add_argument("--all", action="store_true", help="Run demo for all cases in all datasets")
    args = parser.parse_args()

    if args.list:
        list_cases()
        return

    if args.all:
        for dataset in AVAILABLE_DATASETS:
            cases = load_samples(dataset)
            for i in range(len(cases)):
                run_single_demo(dataset, i)
                print()
        return

    dataset = args.dataset or "intravitreal_injections"
    run_single_demo(dataset, args.case)


if __name__ == "__main__":
    main()
