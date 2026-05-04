"""Tests for the PA agent tools."""

import json

from ophthoflow_prior_auth_agent.tools import (
    parse_patient_json,
    check_pa_requirements,
    analyze_missing_information,
    assess_denial_risk,
    draft_pa_letter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMPLETE_CASE = {
    "case_id": "TEST-001",
    "completeness": "complete",
    "urgency": "routine",
    "procedure_category": "intravitreal_injection",
    "patient": {
        "first_name": "Jane",
        "last_name": "Doe",
        "date_of_birth": "1946-03-15",
        "gender": "female",
        "member_id": "MEM-12345",
    },
    "insurance": {
        "payer_name": "Aetna",
        "plan_type": "PPO",
        "member_id": "AET-67890",
        "group_number": "GRP-001",
    },
    "requesting_provider": {
        "provider_name": "Dr. Smith",
        "npi": "1234567890",
        "specialty": "Ophthalmology",
        "practice_name": "Eye Care Associates",
    },
    "diagnoses": [
        {"icd10_code": "H35.3210", "description": "Wet AMD, right eye", "is_primary": True},
    ],
    "procedures": [
        {"cpt_code": "J0178", "description": "Aflibercept (Eylea) injection", "laterality": "OD"},
    ],
    "visual_acuity": [
        {"eye": "OD", "best_corrected": "20/80", "method": "Snellen"},
    ],
    "exam_findings": [
        {"eye": "OD", "structure": "macula", "finding": "subretinal fluid", "severity": "moderate"},
    ],
    "imaging_studies": [
        {"modality": "OCT", "eye": "OD", "findings": "SRF with PED", "central_subfield_thickness_um": 380},
    ],
    "prior_treatments": [
        {"treatment_name": "Bevacizumab (Avastin)", "total_doses": 6, "response": "inadequate"},
    ],
    "clinical_notes": [
        {"text": "Patient with wAMD OD, failed bevacizumab, switching to Eylea."},
    ],
}

MINIMAL_CASE = {
    "case_id": "TEST-002",
    "completeness": "minimal",
    "patient": {
        "first_name": "John",
        "last_name": "Smith",
        "date_of_birth": "1955-01-01",
    },
    "diagnoses": [
        {"icd10_code": "H35.31", "description": "Wet AMD"},
    ],
    "procedures": [
        {"cpt_code": "J2778", "description": "Ranibizumab (Lucentis) injection"},
    ],
}


# ---------------------------------------------------------------------------
# parse_patient_json tests
# ---------------------------------------------------------------------------

def test_parse_complete_case():
    """Parsing a complete case extracts all key fields."""
    result = parse_patient_json(json.dumps(COMPLETE_CASE))
    assert result["status"] == "parsed"
    assert result["patient_name"] == "Jane Doe"
    assert result["payer"] == "Aetna"
    assert "H35.3210" in result["icd10_codes"]
    assert result["primary_cpt"] == "J0178"
    assert "Bevacizumab (Avastin)" in result["prior_treatment_names"]
    assert result["completeness"] == "complete"


def test_parse_minimal_case():
    """Parsing a minimal case handles missing fields gracefully."""
    result = parse_patient_json(json.dumps(MINIMAL_CASE))
    assert result["status"] == "parsed"
    assert result["patient_name"] == "John Smith"
    assert result["payer"] == ""  # No insurance block
    assert result["primary_cpt"] == "J2778"


def test_parse_invalid_json():
    """Invalid JSON returns an error."""
    result = parse_patient_json("not json at all {{{")
    assert "error" in result


# ---------------------------------------------------------------------------
# analyze_missing_information tests
# ---------------------------------------------------------------------------

def test_analyze_complete_case_few_gaps():
    """A complete case should have few or no critical gaps."""
    result = analyze_missing_information(json.dumps(COMPLETE_CASE), payer="Aetna", cpt_code="J0178")
    assert result["critical_gaps"] == 0


def test_analyze_minimal_case_has_gaps():
    """A minimal case should have multiple gaps."""
    result = analyze_missing_information(json.dumps(MINIMAL_CASE), cpt_code="J2778")
    assert result["total_gaps"] > 0
    assert result["critical_gaps"] > 0  # Missing insurance, etc.


def test_analyze_anti_vegf_no_treatments():
    """Brand anti-VEGF without prior treatments flags step therapy gap."""
    result = analyze_missing_information(json.dumps(MINIMAL_CASE), cpt_code="J2778")
    step_gap = [g for g in result["gaps"] if "step therapy" in g["message"].lower() or "treatment" in g["message"].lower()]
    assert len(step_gap) > 0


# ---------------------------------------------------------------------------
# assess_denial_risk tests
# ---------------------------------------------------------------------------

def test_risk_low_for_no_pa():
    """Medicare bevacizumab = no PA required → low risk."""
    result = assess_denial_risk(
        payer="Medicare Part B",
        cpt_code="J9035",
        patient_name="Test Patient",
    )
    assert result["risk_level"] == "LOW"


def test_risk_high_for_missing_step_therapy():
    """Aetna brand anti-VEGF without step therapy → high risk factors."""
    result = assess_denial_risk(
        payer="Aetna",
        cpt_code="J2778",
        patient_name="Test Patient",
        icd10_codes=["H35.31"],
        prior_treatments=[],
        has_imaging=False,
        has_visual_acuity=False,
        step_therapy_met=False,
        completeness_level="minimal",
    )
    assert result["risk_level"] in ("HIGH", "MODERATE")
    assert len(result["risk_factors"]) > 0
    assert len(result["recommendations"]) > 0


def test_risk_includes_recommendations():
    """Risk assessment always provides actionable recommendations."""
    result = assess_denial_risk(
        payer="Cigna",
        cpt_code="J0178",
        step_therapy_met=False,
        has_imaging=False,
    )
    assert len(result["recommendations"]) > 0


# ---------------------------------------------------------------------------
# draft_pa_letter tests
# ---------------------------------------------------------------------------

def test_draft_letter_has_required_sections():
    """Drafted letter contains all key sections."""
    result = draft_pa_letter(
        patient_name="Jane Doe",
        date_of_birth="1946-03-15",
        member_id="AET-67890",
        diagnosis="Wet AMD, right eye",
        icd10_codes=["H35.3210"],
        procedure="Aflibercept (Eylea) injection",
        cpt_code="J0178",
        payer="Aetna",
        clinical_findings="Subretinal fluid in macula",
        prior_treatments=["Bevacizumab (Avastin) x6"],
        step_therapy_met=True,
        provider_name="Dr. Smith",
    )
    assert result["status"] == "draft_complete"
    letter = result["letter_text"]
    assert "Jane Doe" in letter
    assert "Aetna" in letter
    assert "H35.3210" in letter
    assert "J0178" in letter
    assert "MEDICAL NECESSITY" in letter
    assert "CLINICAL SUMMARY" in letter
    assert result["reference_id"].startswith("PA-")


def test_draft_letter_urgent():
    """Urgent letters include urgency section."""
    result = draft_pa_letter(
        patient_name="Test Patient",
        diagnosis="Retinal detachment",
        procedure="Vitrectomy",
        cpt_code="67108",
        payer="Aetna",
        urgency="urgent",
    )
    assert "URGENT" in result["letter_text"]


def test_draft_letter_step_therapy_not_met():
    """Letter notes when step therapy is incomplete."""
    result = draft_pa_letter(
        patient_name="Test Patient",
        diagnosis="Wet AMD",
        procedure="Lucentis injection",
        cpt_code="J2778",
        payer="Aetna",
        step_therapy_met=False,
        prior_treatments=["Avastin x2"],
    )
    assert "step therapy" in result["letter_text"].lower() or "incomplete" in result["letter_text"].lower()
