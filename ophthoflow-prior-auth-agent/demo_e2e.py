#!/usr/bin/env python3
"""
OphthoFlow Prior Auth Agent — End-to-End Demonstration
======================================================

Runs the agent against three representative patient scenarios:

  1. COMPLETE AUTHORIZATION — A fully-documented case that sails through
     PA checks with low denial risk and produces a ready-to-submit letter.

  2. MISSING INFORMATION — A partially-documented case where gaps are
     identified before submission, preventing unnecessary denials.

  3. DENIAL RISK — A minimally-documented case that triggers high denial
     risk due to step therapy non-compliance and missing clinical evidence.

Usage:
    python3 demo_e2e.py           # Run all three scenarios
    python3 demo_e2e.py --json    # Output structured JSON results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from ophthoflow_prior_auth_agent.sample_data.loader import load_samples
from ophthoflow_prior_auth_agent.tools import (
    parse_patient_json,
    check_pa_requirements,
    analyze_missing_information,
    assess_denial_risk,
    draft_pa_letter,
)


# ---------------------------------------------------------------------------
# Terminal Formatting
# ---------------------------------------------------------------------------

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    WHITE = "\033[97m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"


def banner(text: str, color: str = Colors.CYAN) -> None:
    width = 76
    print()
    print(f"{color}{Colors.BOLD}{'═' * width}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}  {text}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{'═' * width}{Colors.RESET}")


def section(text: str, color: str = Colors.WHITE) -> None:
    print(f"\n{color}{Colors.BOLD}  ┌{'─' * 68}┐{Colors.RESET}")
    print(f"{color}{Colors.BOLD}  │  {text:<66}│{Colors.RESET}")
    print(f"{color}{Colors.BOLD}  └{'─' * 68}┘{Colors.RESET}")


def step_header(num: int, title: str) -> None:
    print(f"\n  {Colors.DIM}{'─' * 68}{Colors.RESET}")
    print(f"  {Colors.BOLD}Step {num}: {title}{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 68}{Colors.RESET}")


def kv(key: str, value: str, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{Colors.DIM}{key:20s}{Colors.RESET} {value}")


def status_badge(level: str) -> str:
    badges = {
        "LOW": f"{Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD}  LOW  {Colors.RESET}",
        "MODERATE": f"{Colors.BG_YELLOW}{Colors.WHITE}{Colors.BOLD}  MOD  {Colors.RESET}",
        "HIGH": f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}  HIGH {Colors.RESET}",
    }
    return badges.get(level, level)


def completeness_badge(level: str) -> str:
    badges = {
        "COMPLETE": f"{Colors.GREEN}{Colors.BOLD}● COMPLETE{Colors.RESET}",
        "PARTIALLY COMPLETE": f"{Colors.YELLOW}{Colors.BOLD}◐ PARTIAL{Colors.RESET}",
        "INCOMPLETE": f"{Colors.RED}{Colors.BOLD}○ INCOMPLETE{Colors.RESET}",
        "MOSTLY COMPLETE": f"{Colors.GREEN}{Colors.BOLD}◑ MOSTLY COMPLETE{Colors.RESET}",
    }
    for key, badge in badges.items():
        if key in level.upper():
            return badge
    return level


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "title": "COMPLETE AUTHORIZATION",
        "subtitle": "Fully-documented wet AMD patient — expect approval",
        "dataset": "intravitreal_injections",
        "case_index": 0,  # IVT-001-COMPLETE
        "color": Colors.GREEN,
        "description": (
            "This case demonstrates the ideal workflow: a patient with full clinical\n"
            "    documentation, imaging, visual acuity, step therapy history, and insurance\n"
            "    details. The agent validates completeness, confirms low denial risk, and\n"
            "    generates a ready-to-submit PA letter."
        ),
    },
    {
        "title": "MISSING INFORMATION",
        "subtitle": "Partially-documented cataract patient — gaps identified",
        "dataset": "cataract_surgery",
        "case_index": 1,  # CAT-002-PARTIAL
        "color": Colors.YELLOW,
        "description": (
            "This case shows how the agent identifies documentation gaps before\n"
            "    submission. Missing imaging, incomplete treatment history, and absent\n"
            "    clinical findings are flagged with severity ratings, preventing a\n"
            "    premature submission that would likely result in an information request."
        ),
    },
    {
        "title": "DENIAL RISK",
        "subtitle": "Minimal retinal case — high risk of denial",
        "dataset": "retinal_imaging",
        "case_index": 2,  # IMG-003-MINIMAL
        "color": Colors.RED,
        "description": (
            "This case demonstrates a worst-case scenario: minimal documentation,\n"
            "    no imaging, no visual acuity, and no treatment history. The agent\n"
            "    assigns a HIGH denial risk and provides actionable recommendations\n"
            "    to remediate before submission."
        ),
    },
]


# ---------------------------------------------------------------------------
# Workflow Runner
# ---------------------------------------------------------------------------

def run_scenario(scenario: dict[str, Any], output_json: bool = False) -> dict[str, Any]:
    """Run the full PA workflow for a scenario and return structured results."""
    dataset = scenario["dataset"]
    case_index = scenario["case_index"]
    color = scenario["color"]

    cases = load_samples(dataset)
    case = cases[case_index].model_dump(mode="json")
    case_json = json.dumps(case, indent=2, default=str)

    results: dict[str, Any] = {
        "scenario": scenario["title"],
        "case_id": case.get("case_id"),
        "steps": {},
    }

    if not output_json:
        banner(f"SCENARIO: {scenario['title']}", color)
        print(f"\n  {Colors.DIM}{scenario['subtitle']}{Colors.RESET}")
        print(f"\n    {scenario['description']}")

    # --- Step 1: Parse Patient Data ---
    if not output_json:
        step_header(1, "PARSE PATIENT DATA")

    parsed = parse_patient_json(case_json)
    results["steps"]["parse"] = parsed

    if not output_json:
        kv("Patient", parsed["patient_name"])
        kv("DOB", parsed["date_of_birth"])
        kv("Payer", parsed["payer"] or "(missing)")
        kv("Plan Type", parsed["plan_type"] or "(missing)")
        kv("Member ID", parsed["insurance_member_id"] or "(missing)")
        kv("Primary Dx", parsed["primary_diagnosis"] or "(missing)")
        kv("ICD-10", ", ".join(parsed["icd10_codes"]) or "(none)")
        kv("Procedure", parsed["primary_procedure"] or "(missing)")
        kv("CPT Code", parsed["primary_cpt"] or "(missing)")
        kv("Provider", parsed["provider_name"] or "(missing)")
        kv("Practice", parsed["practice_name"] or "(missing)")
        if parsed.get("visual_acuity"):
            for va in parsed["visual_acuity"]:
                kv(f"VA ({va['eye']})", f"{va['best_corrected']} ({va.get('method', '')})")
        if parsed.get("prior_treatment_names"):
            kv("Prior Tx", ", ".join(parsed["prior_treatment_names"]))
        if parsed.get("imaging_summary"):
            summary = parsed["imaging_summary"]
            if len(summary) > 80:
                summary = summary[:80] + "…"
            kv("Imaging", summary)

    payer = parsed["payer"]
    cpt = parsed["primary_cpt"]
    icd10s = parsed["icd10_codes"]
    tx_names = parsed["prior_treatment_names"]

    # --- Step 2: Check PA Requirements ---
    if not output_json:
        step_header(2, "CHECK PA REQUIREMENTS")

    if payer and cpt:
        pa_result = check_pa_requirements(
            payer=payer, cpt_code=cpt, icd10_codes=icd10s, prior_treatments=tx_names
        )
    else:
        pa_result = {"pa_required": None, "review_timeline_days": "N/A", "notes": "Insufficient data"}

    results["steps"]["pa_requirements"] = pa_result

    if not output_json:
        pa_req = pa_result.get("pa_required")
        if pa_req is True:
            kv("PA Required", f"{Colors.YELLOW}YES{Colors.RESET}")
        elif pa_req is False:
            kv("PA Required", f"{Colors.GREEN}NO{Colors.RESET}")
        else:
            kv("PA Required", f"{Colors.RED}UNKNOWN (insufficient data){Colors.RESET}")
        kv("Timeline", f"{pa_result.get('review_timeline_days', '?')} business days")
        st = pa_result.get("step_therapy_met")
        if st is True:
            kv("Step Therapy", f"{Colors.GREEN}MET{Colors.RESET}")
        elif st is False:
            kv("Step Therapy", f"{Colors.RED}NOT MET{Colors.RESET}")
        elif st is None and cpt in ("J2778", "J0178", "J2503"):
            kv("Step Therapy", f"{Colors.YELLOW}UNKNOWN{Colors.RESET}")
        if pa_result.get("required_documents"):
            print(f"\n    {Colors.DIM}Required Documents:{Colors.RESET}")
            for doc in pa_result["required_documents"][:5]:
                print(f"      • {doc}")
            remaining = len(pa_result["required_documents"]) - 5
            if remaining > 0:
                print(f"      {Colors.DIM}… and {remaining} more{Colors.RESET}")

    # --- Step 3: Analyze Missing Information ---
    if not output_json:
        step_header(3, "ANALYZE MISSING INFORMATION")

    gaps_result = analyze_missing_information(case_json, payer=payer, cpt_code=cpt)
    results["steps"]["missing_info"] = gaps_result

    if not output_json:
        assessment = gaps_result["overall_assessment"]
        print(f"    {completeness_badge(assessment)}")
        print(f"    {Colors.DIM}{assessment}{Colors.RESET}")
        print()
        total = gaps_result["total_gaps"]
        critical = gaps_result["critical_gaps"]
        high = gaps_result["high_gaps"]
        moderate = gaps_result["moderate_gaps"]
        kv("Total Gaps", str(total))
        if critical:
            kv("Critical", f"{Colors.RED}{critical}{Colors.RESET}")
        if high:
            kv("High", f"{Colors.YELLOW}{high}{Colors.RESET}")
        if moderate:
            kv("Moderate", f"{Colors.DIM}{moderate}{Colors.RESET}")
        if gaps_result["gaps"]:
            print(f"\n    {Colors.DIM}Gap Details:{Colors.RESET}")
            for g in gaps_result["gaps"][:8]:
                sev = g["severity"]
                sev_color = {
                    "CRITICAL": Colors.RED,
                    "HIGH": Colors.YELLOW,
                    "MODERATE": Colors.DIM,
                }.get(sev, "")
                print(f"      {sev_color}[{sev:8s}]{Colors.RESET} {g['message']}")
            if len(gaps_result["gaps"]) > 8:
                print(f"      {Colors.DIM}… and {len(gaps_result['gaps']) - 8} more{Colors.RESET}")

    # --- Step 4: Assess Denial Risk ---
    if not output_json:
        step_header(4, "ASSESS DENIAL RISK")

    risk = assess_denial_risk(
        payer=payer or "Unknown",
        cpt_code=cpt or "Unknown",
        patient_name=parsed["patient_name"],
        icd10_codes=icd10s,
        prior_treatments=tx_names,
        has_imaging=bool(parsed.get("imaging_studies")),
        has_visual_acuity=bool(parsed.get("visual_acuity")),
        step_therapy_met=pa_result.get("step_therapy_met"),
        completeness_level=parsed.get("completeness", "unknown"),
    )
    results["steps"]["denial_risk"] = risk

    if not output_json:
        print(f"    {status_badge(risk['risk_level'])}  Score: {risk['risk_score']}/100")
        print(f"\n    {risk['summary']}")
        if risk.get("risk_factors"):
            print(f"\n    {Colors.DIM}Risk Factors:{Colors.RESET}")
            for rf in risk["risk_factors"]:
                impact_color = {
                    "HIGH": Colors.RED,
                    "MODERATE": Colors.YELLOW,
                }.get(rf["impact"], "")
                print(f"      {impact_color}[{rf['impact']:8s}]{Colors.RESET} {rf['factor']}")
                if rf.get("detail"):
                    detail = rf["detail"]
                    if len(detail) > 70:
                        detail = detail[:70] + "…"
                    print(f"               {Colors.DIM}{detail}{Colors.RESET}")
        if risk.get("recommendations"):
            print(f"\n    {Colors.BOLD}Recommendations:{Colors.RESET}")
            for i, rec in enumerate(risk["recommendations"], 1):
                if len(rec) > 80:
                    rec = rec[:80] + "…"
                print(f"      {i}. {rec}")

    # --- Step 5: Draft PA Letter (if required) ---
    pa_req = pa_result.get("pa_required")
    if pa_req and payer and cpt:
        if not output_json:
            step_header(5, "DRAFT PA LETTER")

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
        results["steps"]["pa_letter"] = {
            "reference_id": letter["reference_id"],
            "status": letter["status"],
            "letter_length": len(letter["letter_text"]),
        }

        if not output_json:
            print(f"    {Colors.GREEN}✓ Letter generated{Colors.RESET}  "
                  f"Ref: {letter['reference_id']}")
            print(f"    Length: {len(letter['letter_text'])} characters\n")
            print(f"    {Colors.DIM}{'─' * 60}{Colors.RESET}")
            for line in letter["letter_text"].split("\n"):
                print(f"    {Colors.DIM}│{Colors.RESET} {line}")
            print(f"    {Colors.DIM}{'─' * 60}{Colors.RESET}")
    else:
        if not output_json:
            step_header(5, "PA LETTER")
            if not pa_req and pa_req is not None:
                print(f"    {Colors.GREEN}PA not required — standard billing can proceed.{Colors.RESET}")
            else:
                print(f"    {Colors.RED}⚠ Cannot generate letter — insufficient case data.{Colors.RESET}")
                print(f"    {Colors.DIM}Resolve critical gaps above before attempting submission.{Colors.RESET}")
        results["steps"]["pa_letter"] = {"status": "skipped", "reason": "PA not required or data insufficient"}

    # --- Outcome Summary ---
    if not output_json:
        print(f"\n  {color}{'━' * 68}{Colors.RESET}")
        risk_level = risk["risk_level"]
        if risk_level == "LOW" and gaps_result["critical_gaps"] == 0:
            outcome = f"{Colors.GREEN}{Colors.BOLD}✓ OUTCOME: Ready for submission{Colors.RESET}"
        elif risk_level == "HIGH" or gaps_result["critical_gaps"] > 0:
            outcome = f"{Colors.RED}{Colors.BOLD}✗ OUTCOME: NOT ready — resolve issues before submission{Colors.RESET}"
        else:
            outcome = f"{Colors.YELLOW}{Colors.BOLD}◐ OUTCOME: Submittable with caveats — strengthen documentation{Colors.RESET}"
        print(f"  {outcome}")
        print(f"  {color}{'━' * 68}{Colors.RESET}")

    results["outcome"] = {
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "total_gaps": gaps_result["total_gaps"],
        "critical_gaps": gaps_result["critical_gaps"],
        "ready_for_submission": (
            risk["risk_level"] == "LOW" and gaps_result["critical_gaps"] == 0
        ),
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OphthoFlow PA Agent — End-to-End Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scenarios:\n"
            "  1. Complete Authorization — fully-documented case, low risk\n"
            "  2. Missing Information — partial docs, gaps flagged\n"
            "  3. Denial Risk — minimal docs, high risk of denial\n"
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output structured JSON results instead of formatted text",
    )
    parser.add_argument(
        "--scenario", type=int, choices=[1, 2, 3],
        help="Run a specific scenario (1-3) instead of all",
    )
    args = parser.parse_args()

    scenarios_to_run = (
        [SCENARIOS[args.scenario - 1]] if args.scenario else SCENARIOS
    )

    if not args.json:
        print(f"\n{Colors.BOLD}{Colors.CYAN}")
        print("  ┌──────────────────────────────────────────────────────────────────┐")
        print("  │                                                                  │")
        print("  │           OphthoFlow Prior Authorization Agent                   │")
        print("  │           End-to-End Demonstration                               │")
        print("  │                                                                  │")
        print("  │   Demonstrates the complete PA workflow across three scenarios:   │")
        print("  │     1. Complete Authorization (low risk, ready to submit)         │")
        print("  │     2. Missing Information (gaps identified pre-submission)       │")
        print("  │     3. Denial Risk (high risk, remediation required)              │")
        print("  │                                                                  │")
        print("  └──────────────────────────────────────────────────────────────────┘")
        print(f"{Colors.RESET}")

    all_results: list[dict[str, Any]] = []
    for scenario in scenarios_to_run:
        result = run_scenario(scenario, output_json=args.json)
        all_results.append(result)

    if args.json:
        print(json.dumps(all_results, indent=2, default=str))
    else:
        # Final summary table
        print()
        banner("SUMMARY", Colors.CYAN)
        print()
        header = (
            f"    {'Scenario':<26s} {'Case ID':<20s} "
            f"{'Risk':<8s} {'Gaps':<6s} {'Ready?'}"
        )
        print(f"  {Colors.BOLD}{header}{Colors.RESET}")
        print(f"    {'─' * 70}")
        for r in all_results:
            scenario_name = r["scenario"]
            case_id = r["case_id"] or "?"
            risk = r["outcome"]["risk_level"]
            gaps = r["outcome"]["total_gaps"]
            ready = r["outcome"]["ready_for_submission"]
            risk_color = {
                "LOW": Colors.GREEN,
                "MODERATE": Colors.YELLOW,
                "HIGH": Colors.RED,
            }.get(risk, "")
            ready_str = (
                f"{Colors.GREEN}YES{Colors.RESET}" if ready
                else f"{Colors.RED}NO{Colors.RESET}"
            )
            print(
                f"    {scenario_name:<26s} {case_id:<20s} "
                f"{risk_color}{risk:<8s}{Colors.RESET} {gaps:<6} {ready_str}"
            )
        print(f"    {'─' * 70}")
        print(f"\n  {Colors.DIM}Demo complete. {len(all_results)} scenarios processed.{Colors.RESET}\n")


if __name__ == "__main__":
    main()
