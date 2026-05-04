"""
Tool definitions for the OphthoFlow PA Agent.

Each tool is defined as an MCP-compatible schema alongside its implementation
for easy registration with the Claude Agent SDK.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime
from typing import Any


# ---------------------------------------------------------------------------
# Tool Schemas (MCP-compatible format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "parse_patient_json",
        "description": (
            "Parse and validate structured patient JSON input. Extracts key clinical "
            "fields (demographics, insurance, diagnoses, procedures, imaging, prior "
            "treatments) and normalizes them for downstream PA processing. Returns a "
            "structured summary with extracted ICD-10 codes, CPT codes, payer info, "
            "and clinical findings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_json": {
                    "type": "string",
                    "description": "JSON string containing the patient case data.",
                },
            },
            "required": ["patient_json"],
        },
    },
    {
        "name": "check_pa_requirements",
        "description": (
            "Query the payer portal for prior authorization requirements given a "
            "payer name and CPT procedure code. Returns whether PA is required, "
            "required documents, step therapy status, and review timeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "payer": {
                    "type": "string",
                    "description": "Insurance payer name (e.g. 'Aetna', 'Medicare').",
                },
                "cpt_code": {
                    "type": "string",
                    "description": "CPT or HCPCS procedure code.",
                },
                "icd10_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ICD-10 diagnosis codes.",
                },
                "prior_treatments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of prior treatments attempted.",
                },
            },
            "required": ["payer", "cpt_code"],
        },
    },
    {
        "name": "analyze_missing_information",
        "description": (
            "Analyze a patient case for missing or incomplete information that could "
            "delay or jeopardize a prior authorization request. Checks for missing "
            "demographics, insurance details, clinical documentation, imaging, and "
            "treatment history. Returns a categorized list of gaps with severity ratings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_json": {
                    "type": "string",
                    "description": "JSON string of the patient case data to analyze.",
                },
                "payer": {
                    "type": "string",
                    "description": "Insurance payer name for requirement-specific checks.",
                },
                "cpt_code": {
                    "type": "string",
                    "description": "CPT code of the requested procedure.",
                },
            },
            "required": ["patient_json"],
        },
    },
    {
        "name": "assess_denial_risk",
        "description": (
            "Assess the risk of prior authorization denial based on the patient case, "
            "payer rules, clinical evidence, and step therapy compliance. Returns a "
            "risk level (LOW, MODERATE, HIGH), specific risk factors, and actionable "
            "recommendations to reduce denial probability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "payer": {"type": "string"},
                "cpt_code": {"type": "string"},
                "icd10_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "prior_treatments": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "has_imaging": {"type": "boolean"},
                "has_visual_acuity": {"type": "boolean"},
                "step_therapy_met": {"type": "boolean"},
                "completeness_level": {"type": "string"},
            },
            "required": ["payer", "cpt_code"],
        },
    },
    {
        "name": "draft_pa_letter",
        "description": (
            "Generate a professional prior authorization request letter using the "
            "parsed patient record and payer requirements. The letter includes "
            "clinical justification, diagnosis codes, procedure details, and "
            "payer-specific documentation. Returns a complete letter ready for "
            "submission along with metadata."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "date_of_birth": {"type": "string"},
                "member_id": {"type": "string"},
                "diagnosis": {"type": "string"},
                "icd10_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "procedure": {"type": "string"},
                "cpt_code": {"type": "string"},
                "payer": {"type": "string"},
                "clinical_findings": {"type": "string"},
                "imaging_summary": {"type": "string"},
                "prior_treatments": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "step_therapy_met": {"type": "boolean"},
                "provider_name": {"type": "string"},
                "provider_npi": {"type": "string"},
                "practice_name": {"type": "string"},
                "urgency": {"type": "string"},
                "requested_duration_months": {"type": "integer"},
                "requested_total_doses": {"type": "integer"},
            },
            "required": ["patient_name", "diagnosis", "procedure", "payer", "cpt_code"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Implementations (pure functions called by the agent orchestrator)
# ---------------------------------------------------------------------------


def parse_patient_json(patient_json: str) -> dict[str, Any]:
    """Parse and validate structured patient JSON, extracting key PA fields."""
    try:
        data = json.loads(patient_json) if isinstance(patient_json, str) else patient_json
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "status": "parse_error"}

    result: dict[str, Any] = {"status": "parsed"}

    # Extract demographics
    patient = data.get("patient", {})
    result["patient_name"] = f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip()
    result["date_of_birth"] = str(patient.get("date_of_birth", ""))
    result["gender"] = patient.get("gender", "unknown")
    result["member_id"] = patient.get("member_id", "")

    # Extract insurance
    insurance = data.get("insurance", {})
    result["payer"] = insurance.get("payer_name", "")
    result["plan_type"] = insurance.get("plan_type", "")
    result["insurance_member_id"] = insurance.get("member_id", "")
    result["group_number"] = insurance.get("group_number", "")

    # Extract provider
    provider = data.get("requesting_provider", {})
    result["provider_name"] = provider.get("provider_name", "")
    result["provider_npi"] = provider.get("npi", "")
    result["practice_name"] = provider.get("practice_name", "")

    # Extract diagnoses
    diagnoses = data.get("diagnoses", [])
    result["diagnoses"] = [
        {
            "icd10_code": d.get("icd10_code", ""),
            "description": d.get("description", ""),
            "is_primary": d.get("is_primary", False),
        }
        for d in diagnoses
    ]
    result["icd10_codes"] = [d.get("icd10_code", "") for d in diagnoses if d.get("icd10_code")]
    primary = [d for d in diagnoses if d.get("is_primary")]
    result["primary_diagnosis"] = primary[0].get("description", "") if primary else (
        diagnoses[0].get("description", "") if diagnoses else ""
    )

    # Extract procedures
    procedures = data.get("procedures", [])
    result["procedures"] = [
        {
            "cpt_code": p.get("cpt_code", ""),
            "description": p.get("description", ""),
            "laterality": p.get("laterality"),
            "dosage": p.get("dosage"),
            "frequency": p.get("frequency"),
            "quantity": p.get("quantity", 1),
        }
        for p in procedures
    ]
    result["cpt_codes"] = [p.get("cpt_code", "") for p in procedures if p.get("cpt_code")]
    result["primary_cpt"] = result["cpt_codes"][0] if result["cpt_codes"] else ""
    result["primary_procedure"] = procedures[0].get("description", "") if procedures else ""

    # Extract visual acuity
    va = data.get("visual_acuity", [])
    result["visual_acuity"] = [
        {
            "eye": v.get("eye", ""),
            "best_corrected": v.get("best_corrected", ""),
            "method": v.get("method", ""),
        }
        for v in va
    ]

    # Extract exam findings
    findings = data.get("exam_findings", [])
    result["clinical_findings"] = "; ".join(
        f"{f.get('eye', '')} {f.get('structure', '')}: {f.get('finding', '')}"
        + (f" ({f.get('severity', '')})" if f.get("severity") else "")
        for f in findings
    )

    # Extract imaging
    imaging = data.get("imaging_studies", [])
    result["imaging_studies"] = [
        {
            "modality": im.get("modality", ""),
            "eye": im.get("eye"),
            "findings": im.get("findings", ""),
            "cst_um": im.get("central_subfield_thickness_um"),
            "subretinal_fluid": im.get("subretinal_fluid"),
            "intraretinal_fluid": im.get("intraretinal_fluid"),
        }
        for im in imaging
    ]
    result["imaging_summary"] = "; ".join(
        f"{im.get('modality', '')} ({im.get('eye', '')}): {im.get('findings', '')}"
        for im in imaging
    )

    # Extract prior treatments
    treatments = data.get("prior_treatments", [])
    result["prior_treatments"] = [
        {
            "name": t.get("treatment_name", ""),
            "code": t.get("cpt_or_j_code", ""),
            "total_doses": t.get("total_doses"),
            "response": t.get("response", ""),
            "reason_discontinued": t.get("reason_discontinued", ""),
        }
        for t in treatments
    ]
    result["prior_treatment_names"] = [t.get("treatment_name", "") for t in treatments]

    # Extract metadata
    result["case_id"] = data.get("case_id", "")
    result["completeness"] = data.get("completeness", "unknown")
    result["urgency"] = data.get("urgency", "routine")
    result["procedure_category"] = data.get("procedure_category", "")
    result["requested_duration_months"] = data.get("requested_duration_months")
    result["requested_total_doses"] = data.get("requested_total_doses")

    # Clinical notes
    notes = data.get("clinical_notes", [])
    result["clinical_notes_count"] = len(notes)
    result["clinical_notes_text"] = "\n\n".join(n.get("text", "") for n in notes)

    return result


def check_pa_requirements(
    payer: str,
    cpt_code: str,
    icd10_codes: list[str] | None = None,
    prior_treatments: list[str] | None = None,
) -> dict[str, Any]:
    """Check PA requirements against the payer rules engine."""
    from ophthoflow_prior_auth_agent.payer_portal import check_requirements

    return check_requirements(
        payer=payer,
        cpt_code=cpt_code,
        icd10_codes=icd10_codes,
        prior_treatments=prior_treatments,
    )


def analyze_missing_information(
    patient_json: str,
    payer: str | None = None,
    cpt_code: str | None = None,
) -> dict[str, Any]:
    """Analyze patient case for missing information that could delay PA."""
    try:
        data = json.loads(patient_json) if isinstance(patient_json, str) else patient_json
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

    gaps: list[dict[str, str]] = []

    # Check demographics
    patient = data.get("patient", {})
    if not patient:
        gaps.append({"field": "patient", "severity": "CRITICAL", "message": "No patient demographics found"})
    else:
        if not patient.get("first_name") or not patient.get("last_name"):
            gaps.append({"field": "patient.name", "severity": "CRITICAL", "message": "Patient name is missing"})
        if not patient.get("date_of_birth"):
            gaps.append({"field": "patient.date_of_birth", "severity": "HIGH", "message": "Date of birth is missing"})
        if not patient.get("member_id"):
            gaps.append({"field": "patient.member_id", "severity": "HIGH", "message": "Insurance member ID is missing"})

    # Check insurance
    insurance = data.get("insurance", {})
    if not insurance:
        gaps.append({"field": "insurance", "severity": "CRITICAL", "message": "No insurance information provided"})
    else:
        if not insurance.get("payer_name"):
            gaps.append({"field": "insurance.payer_name", "severity": "CRITICAL", "message": "Payer name is missing"})
        if not insurance.get("member_id"):
            gaps.append({"field": "insurance.member_id", "severity": "HIGH", "message": "Insurance member ID is missing"})
        if not insurance.get("group_number"):
            gaps.append({"field": "insurance.group_number", "severity": "MODERATE", "message": "Group number not provided"})

    # Check provider
    provider = data.get("requesting_provider", {})
    if not provider:
        gaps.append({"field": "requesting_provider", "severity": "HIGH", "message": "No requesting provider information"})
    else:
        if not provider.get("npi"):
            gaps.append({"field": "provider.npi", "severity": "HIGH", "message": "Provider NPI number is missing"})
        if not provider.get("provider_name"):
            gaps.append({"field": "provider.provider_name", "severity": "HIGH", "message": "Provider name is missing"})

    # Check diagnoses
    diagnoses = data.get("diagnoses", [])
    if not diagnoses:
        gaps.append({"field": "diagnoses", "severity": "CRITICAL", "message": "No diagnosis codes provided"})
    else:
        has_primary = any(d.get("is_primary") for d in diagnoses)
        if not has_primary:
            gaps.append({"field": "diagnoses.is_primary", "severity": "MODERATE", "message": "No diagnosis marked as primary"})
        for d in diagnoses:
            if not d.get("icd10_code"):
                gaps.append({"field": "diagnoses.icd10_code", "severity": "CRITICAL", "message": f"Missing ICD-10 code for diagnosis: {d.get('description', 'unknown')}"})

    # Check procedures
    procedures = data.get("procedures", [])
    if not procedures:
        gaps.append({"field": "procedures", "severity": "CRITICAL", "message": "No procedure codes provided"})
    else:
        for p in procedures:
            if not p.get("cpt_code"):
                gaps.append({"field": "procedures.cpt_code", "severity": "CRITICAL", "message": f"Missing CPT code for procedure: {p.get('description', 'unknown')}"})

    # Check clinical documentation
    va = data.get("visual_acuity", [])
    if not va:
        gaps.append({"field": "visual_acuity", "severity": "HIGH", "message": "No visual acuity measurements provided"})

    exam = data.get("exam_findings", [])
    if not exam:
        gaps.append({"field": "exam_findings", "severity": "HIGH", "message": "No clinical exam findings documented"})

    imaging = data.get("imaging_studies", [])
    if not imaging:
        gaps.append({"field": "imaging_studies", "severity": "HIGH", "message": "No imaging studies (OCT, FA) documented"})
    else:
        has_oct = any(im.get("modality", "").upper() == "OCT" for im in imaging)
        if not has_oct:
            gaps.append({"field": "imaging_studies.oct", "severity": "MODERATE", "message": "No OCT imaging found — most payers require OCT within 30 days"})

    # Check prior treatments (especially for step therapy)
    treatments = data.get("prior_treatments", [])
    if cpt_code and cpt_code in ("J2778", "J0178", "J2503"):
        if not treatments:
            gaps.append({
                "field": "prior_treatments",
                "severity": "CRITICAL",
                "message": "No prior treatment history — step therapy documentation likely required for brand anti-VEGF agents",
            })
        else:
            has_avastin = any(
                "bevacizumab" in t.get("treatment_name", "").lower() or "avastin" in t.get("treatment_name", "").lower()
                for t in treatments
            )
            if not has_avastin:
                gaps.append({
                    "field": "prior_treatments.bevacizumab",
                    "severity": "HIGH",
                    "message": "No bevacizumab (Avastin) treatment history found — required for step therapy by many payers",
                })

    # Clinical notes
    notes = data.get("clinical_notes", [])
    if not notes:
        gaps.append({"field": "clinical_notes", "severity": "MODERATE", "message": "No clinical notes attached"})

    # Categorize
    critical = [g for g in gaps if g["severity"] == "CRITICAL"]
    high = [g for g in gaps if g["severity"] == "HIGH"]
    moderate = [g for g in gaps if g["severity"] == "MODERATE"]

    completeness = data.get("completeness", "unknown")
    if critical:
        overall = "INCOMPLETE — critical gaps prevent PA submission"
    elif high:
        overall = "PARTIALLY COMPLETE — high-priority gaps should be addressed"
    elif moderate:
        overall = "MOSTLY COMPLETE — minor gaps may slow processing"
    else:
        overall = "COMPLETE — all key fields populated"

    return {
        "overall_assessment": overall,
        "completeness_level": completeness,
        "total_gaps": len(gaps),
        "critical_gaps": len(critical),
        "high_gaps": len(high),
        "moderate_gaps": len(moderate),
        "gaps": gaps,
    }


def assess_denial_risk(
    payer: str,
    cpt_code: str,
    patient_name: str = "",
    icd10_codes: list[str] | None = None,
    prior_treatments: list[str] | None = None,
    has_imaging: bool = False,
    has_visual_acuity: bool = False,
    step_therapy_met: bool | None = None,
    completeness_level: str = "unknown",
) -> dict[str, Any]:
    """Assess denial risk for a PA request."""
    from ophthoflow_prior_auth_agent.payer_portal import check_requirements

    risk_factors: list[dict[str, str]] = []
    recommendations: list[str] = []
    score = 0  # 0-100, higher = more risk

    # Check payer rules
    pa_result = check_requirements(
        payer=payer,
        cpt_code=cpt_code,
        icd10_codes=icd10_codes,
        prior_treatments=prior_treatments,
    )

    # Factor 1: PA required at all?
    if not pa_result.get("pa_required", True):
        return {
            "patient_name": patient_name,
            "risk_level": "LOW",
            "risk_score": 5,
            "summary": f"PA is NOT required by {payer} for CPT {cpt_code}. Proceed with standard billing.",
            "risk_factors": [],
            "recommendations": ["Ensure medical necessity documentation is in clinical notes for audit protection."],
        }

    # Factor 2: Step therapy
    if pa_result.get("step_therapy_met") is False:
        score += 35
        risk_factors.append({
            "factor": "Step therapy NOT met",
            "impact": "HIGH",
            "detail": pa_result.get("step_therapy_details", ""),
        })
        recommendations.append(
            "Document prior bevacizumab (Avastin) treatment history with dates, doses, "
            "and clinical response. If not available, document exception criteria "
            "(allergy, contraindication, urgent clinical need)."
        )
    elif pa_result.get("step_therapy_met") is None and cpt_code in ("J2778", "J0178", "J2503"):
        score += 15
        risk_factors.append({
            "factor": "Step therapy status unknown",
            "impact": "MODERATE",
            "detail": "Unable to determine step therapy compliance — treatment history may be incomplete.",
        })
        recommendations.append("Provide complete prior treatment history for step therapy evaluation.")

    # Factor 3: ICD-10 coverage
    unmet = pa_result.get("clinical_criteria_unmet", [])
    if unmet:
        score += 20
        risk_factors.append({
            "factor": "Clinical criteria not fully met",
            "impact": "HIGH",
            "detail": f"Unmet criteria: {'; '.join(unmet)}",
        })
        recommendations.append("Review and update ICD-10 codes to match approved indications. Ensure clinical documentation supports each criterion.")

    # Factor 4: Missing imaging
    if not has_imaging:
        score += 15
        risk_factors.append({
            "factor": "No imaging documentation",
            "impact": "MODERATE",
            "detail": "OCT and/or FA imaging not found in case data.",
        })
        recommendations.append("Include recent OCT imaging (within 30 days) and fluorescein angiography if available.")

    # Factor 5: Missing visual acuity
    if not has_visual_acuity:
        score += 10
        risk_factors.append({
            "factor": "No visual acuity data",
            "impact": "MODERATE",
            "detail": "Best corrected visual acuity measurements not found.",
        })
        recommendations.append("Document BCVA for the affected eye — most payers require this for medical necessity.")

    # Factor 6: Completeness
    if completeness_level == "minimal":
        score += 15
        risk_factors.append({
            "factor": "Minimal case documentation",
            "impact": "HIGH",
            "detail": "Case record has minimal data — many required fields are missing.",
        })
        recommendations.append("Obtain complete clinical documentation before submission to avoid information requests.")
    elif completeness_level == "partial":
        score += 8
        risk_factors.append({
            "factor": "Partial case documentation",
            "impact": "MODERATE",
            "detail": "Some supporting documentation is missing.",
        })
        recommendations.append("Fill in missing clinical data to strengthen the submission.")

    # Factor 7: No prior treatments for brand agents
    if cpt_code in ("J2778", "J0178", "J2503") and not prior_treatments:
        score += 10
        risk_factors.append({
            "factor": "No treatment history provided",
            "impact": "MODERATE",
            "detail": "Brand anti-VEGF agents typically require documented treatment history.",
        })

    # Determine risk level
    if score >= 50:
        risk_level = "HIGH"
    elif score >= 25:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Build summary
    if risk_level == "HIGH":
        summary = (
            f"HIGH denial risk for {patient_name or 'patient'}'s PA request. "
            f"{len(risk_factors)} risk factor(s) identified. Address recommendations "
            f"before submission to improve approval probability."
        )
    elif risk_level == "MODERATE":
        summary = (
            f"MODERATE denial risk. {len(risk_factors)} risk factor(s) found. "
            f"Strengthening documentation as recommended would reduce risk."
        )
    else:
        summary = (
            f"LOW denial risk. Case appears well-documented for {payer} requirements."
        )

    return {
        "patient_name": patient_name,
        "risk_level": risk_level,
        "risk_score": min(score, 100),
        "summary": summary,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "payer_review_timeline_days": pa_result.get("review_timeline_days", 14),
    }


def draft_pa_letter(
    patient_name: str,
    diagnosis: str,
    procedure: str,
    payer: str,
    cpt_code: str,
    date_of_birth: str = "",
    member_id: str = "",
    icd10_codes: list[str] | None = None,
    clinical_findings: str = "",
    imaging_summary: str = "",
    prior_treatments: list[str] | None = None,
    step_therapy_met: bool | None = None,
    provider_name: str = "",
    provider_npi: str = "",
    practice_name: str = "",
    urgency: str = "routine",
    requested_duration_months: int | None = None,
    requested_total_doses: int | None = None,
) -> dict[str, Any]:
    """Generate a professional PA request letter."""
    today = date.today().strftime("%B %d, %Y")
    ref_id = f"PA-{uuid.uuid4().hex[:8].upper()}"
    icd10_codes = icd10_codes or []
    prior_treatments = prior_treatments or []

    # Build letter sections
    lines: list[str] = []

    # Header
    lines.append(f"Date: {today}")
    lines.append(f"PA Reference: {ref_id}")
    if urgency.lower() in ("urgent", "emergent"):
        lines.append(f"*** {urgency.upper()} REVIEW REQUESTED ***")
    lines.append("")

    # Addressee
    lines.append(f"To: {payer} Prior Authorization Department")
    lines.append(f"Re: Prior Authorization Request — {procedure}")
    lines.append("")

    # Provider info
    if provider_name or practice_name:
        lines.append("From:")
        if provider_name:
            lines.append(f"  {provider_name}")
        if provider_npi:
            lines.append(f"  NPI: {provider_npi}")
        if practice_name:
            lines.append(f"  {practice_name}")
        lines.append("")

    # Patient info
    lines.append("PATIENT INFORMATION")
    lines.append(f"  Name: {patient_name}")
    if date_of_birth:
        lines.append(f"  Date of Birth: {date_of_birth}")
    if member_id:
        lines.append(f"  Member ID: {member_id}")
    lines.append("")

    # Clinical summary
    lines.append("CLINICAL SUMMARY")
    lines.append(f"  Primary Diagnosis: {diagnosis}")
    if icd10_codes:
        lines.append(f"  ICD-10 Code(s): {', '.join(icd10_codes)}")
    lines.append(f"  Requested Procedure: {procedure}")
    lines.append(f"  CPT/HCPCS Code: {cpt_code}")
    if requested_total_doses:
        lines.append(f"  Requested Doses: {requested_total_doses}")
    if requested_duration_months:
        lines.append(f"  Requested Duration: {requested_duration_months} months")
    lines.append("")

    # Medical necessity
    lines.append("MEDICAL NECESSITY JUSTIFICATION")
    lines.append(
        f"I am requesting prior authorization for {procedure} (CPT {cpt_code}) "
        f"for the above-named patient who has been diagnosed with {diagnosis}"
        + (f" ({', '.join(icd10_codes)})" if icd10_codes else "")
        + "."
    )
    lines.append("")

    if clinical_findings:
        lines.append("Clinical Findings:")
        lines.append(f"  {clinical_findings}")
        lines.append("")

    if imaging_summary:
        lines.append("Diagnostic Imaging:")
        lines.append(f"  {imaging_summary}")
        lines.append("")

    # Prior treatment / step therapy
    if prior_treatments:
        lines.append("PRIOR TREATMENT HISTORY")
        for tx in prior_treatments:
            lines.append(f"  - {tx}")
        lines.append("")

        if step_therapy_met is True:
            lines.append(
                "The patient has completed the required step therapy protocol. "
                "Despite adequate trial of prior agents, the patient's condition "
                "has not adequately responded, necessitating escalation to the "
                "requested therapy."
            )
            lines.append("")
        elif step_therapy_met is False:
            lines.append(
                "NOTE: Step therapy documentation may be incomplete. Please see "
                "attached clinical records for the patient's full treatment history "
                "and clinical rationale for the requested therapy."
            )
            lines.append("")

    # Urgency
    if urgency.lower() in ("urgent", "emergent"):
        lines.append("URGENCY")
        lines.append(
            f"This request is marked as {urgency.upper()}. Delayed treatment "
            f"poses a significant risk of irreversible vision loss. Expedited "
            f"review is respectfully requested."
        )
        lines.append("")

    # Conclusion
    lines.append("CONCLUSION")
    lines.append(
        f"Based on the above clinical evidence, {procedure} is medically "
        f"necessary for this patient. The requested treatment meets the "
        f"established criteria for the diagnosis and clinical presentation. "
        f"I respectfully request authorization for this procedure."
    )
    lines.append("")
    lines.append(
        "Please do not hesitate to contact our office if additional information "
        "is required. Thank you for your prompt review of this request."
    )
    lines.append("")

    # Signature
    lines.append("Respectfully,")
    lines.append("")
    if provider_name:
        lines.append(provider_name)
    if provider_npi:
        lines.append(f"NPI: {provider_npi}")
    if practice_name:
        lines.append(practice_name)

    letter_text = "\n".join(lines)

    return {
        "status": "draft_complete",
        "reference_id": ref_id,
        "letter_text": letter_text,
        "metadata": {
            "patient_name": patient_name,
            "payer": payer,
            "cpt_code": cpt_code,
            "icd10_codes": icd10_codes,
            "urgency": urgency,
            "generated_date": today,
        },
    }


# Map tool name → implementation
TOOL_DISPATCH: dict[str, Any] = {
    "parse_patient_json": parse_patient_json,
    "check_pa_requirements": check_pa_requirements,
    "analyze_missing_information": analyze_missing_information,
    "assess_denial_risk": assess_denial_risk,
    "draft_pa_letter": draft_pa_letter,
}
