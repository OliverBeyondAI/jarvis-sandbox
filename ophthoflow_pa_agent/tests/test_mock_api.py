"""
Tests for the OphthoFlow mock payer portal API.
"""

import pytest

from ophthoflow_pa_agent.payer_portal import (
    ClinicalInfo,
    PARequirementCheck,
    PASubmission,
    PASubmissionResponse,
    PAStatusResponse,
    PayerPortalAPI,
)
from ophthoflow_pa_agent.payer_portal.models import PAStatus, UrgencyLevel


@pytest.fixture
def api():
    return PayerPortalAPI(seed=42)


# ---------------------------------------------------------------------------
# check_pa_requirement
# ---------------------------------------------------------------------------


class TestCheckPARequirement:
    def test_anti_vegf_requires_pa(self, api):
        result = api.check_pa_requirement("J2778", "AETNA")
        assert isinstance(result, PARequirementCheck)
        assert result.requires_pa is True
        assert result.status == PAStatus.REQUIRED
        assert "Ranibizumab" in result.procedure_name

    def test_bevacizumab_no_pa_required(self, api):
        result = api.check_pa_requirement("J9035", "AETNA")
        assert result.requires_pa is False
        assert result.status == PAStatus.NOT_REQUIRED

    def test_cataract_surgery_requires_pa(self, api):
        result = api.check_pa_requirement("66984", "UNITEDHEALTHCARE")
        assert result.requires_pa is True
        assert "Cataract" in result.procedure_name

    def test_retinal_detachment_requires_pa(self, api):
        result = api.check_pa_requirement("67108", "BCBS")
        assert result.requires_pa is True
        assert result.estimated_review_hours == 24

    def test_step_therapy_enforced_by_aetna(self, api):
        result = api.check_pa_requirement("J2778", "AETNA")
        assert result.step_therapy_required is True
        assert "bevacizumab" in result.step_therapy_details.lower()

    def test_no_step_therapy_for_uhc(self, api):
        result = api.check_pa_requirement("J2778", "UNITEDHEALTHCARE")
        assert result.step_therapy_required is False

    def test_unknown_procedure_raises(self, api):
        with pytest.raises(ValueError, match="Unknown procedure code"):
            api.check_pa_requirement("99999", "AETNA")

    def test_unknown_payer_raises(self, api):
        with pytest.raises(ValueError, match="Unknown payer"):
            api.check_pa_requirement("J2778", "FAKEPAYER")

    def test_laser_trabeculoplasty_no_pa(self, api):
        result = api.check_pa_requirement("65855", "MEDICARE")
        assert result.requires_pa is False

    def test_prp_no_pa(self, api):
        result = api.check_pa_requirement("67228", "CIGNA")
        assert result.requires_pa is False


# ---------------------------------------------------------------------------
# submit_pa
# ---------------------------------------------------------------------------


def _make_submission(
    procedure_code="J2778",
    payer_id="AETNA",
    urgency=UrgencyLevel.ROUTINE,
    include_all_docs=True,
) -> PASubmission:
    clinical = ClinicalInfo(
        diagnosis_codes=["H35.32"],
        visual_acuity_od="20/200" if include_all_docs else None,
        visual_acuity_os="20/30" if include_all_docs else None,
        oct_findings="Subretinal fluid with CNV" if include_all_docs else None,
        prior_treatments=["Bevacizumab x 3"] if include_all_docs else [],
        symptoms_duration_days=30 if include_all_docs else None,
        functional_impairment="Difficulty reading" if include_all_docs else None,
        additional_notes="B-scan normal" if include_all_docs else None,
    )
    return PASubmission(
        procedure_code=procedure_code,
        payer_id=payer_id,
        member_id="AET123456789",
        provider_npi="1234567890",
        provider_name="Dr. Jane Smith, MD",
        patient_name="John Doe",
        patient_dob="1955-03-15",
        date_of_service="2026-05-01",
        place_of_service="11",
        diagnosis_codes=["H35.32"],
        clinical_info=clinical,
        urgency=urgency,
        quantity=3,
    )


class TestSubmitPA:
    def test_successful_submission(self, api):
        sub = _make_submission()
        resp = api.submit_pa(sub)
        assert isinstance(resp, PASubmissionResponse)
        assert resp.auth_reference_number.startswith("PA-2026-")
        assert resp.tracking_id.startswith("TRK-")
        assert resp.payer_name == "Aetna"
        assert resp.procedure_name == "Ranibizumab (Lucentis) injection"

    def test_submission_with_missing_docs(self, api):
        sub = _make_submission(include_all_docs=False)
        resp = api.submit_pa(sub)
        assert resp.status == PAStatus.INFO_REQUESTED
        assert len(resp.missing_documents) > 0

    def test_complete_submission_pending_review(self, api):
        sub = _make_submission(include_all_docs=True)
        resp = api.submit_pa(sub)
        assert resp.status == PAStatus.PENDING_REVIEW
        assert len(resp.missing_documents) == 0

    def test_urgent_submission(self, api):
        sub = _make_submission(urgency=UrgencyLevel.URGENT)
        resp = api.submit_pa(sub)
        assert resp.urgency == UrgencyLevel.URGENT

    def test_emergent_submission(self, api):
        sub = _make_submission(urgency=UrgencyLevel.EMERGENT)
        resp = api.submit_pa(sub)
        assert resp.urgency == UrgencyLevel.EMERGENT

    def test_cataract_submission(self, api):
        sub = _make_submission(procedure_code="66984", payer_id="BCBS")
        resp = api.submit_pa(sub)
        assert "Cataract" in resp.procedure_name

    def test_retinal_detachment_submission(self, api):
        sub = _make_submission(procedure_code="67108", payer_id="MEDICARE")
        resp = api.submit_pa(sub)
        assert "Retinal detachment" in resp.procedure_name


# ---------------------------------------------------------------------------
# check_pa_status
# ---------------------------------------------------------------------------


class TestCheckPAStatus:
    def test_status_after_submission(self, api):
        sub = _make_submission()
        submit_resp = api.submit_pa(sub)
        status = api.check_pa_status(submit_resp.auth_reference_number)
        assert isinstance(status, PAStatusResponse)
        assert status.auth_reference_number == submit_resp.auth_reference_number
        assert status.member_id == "AET123456789"

    def test_status_unknown_reference_raises(self, api):
        with pytest.raises(ValueError, match="not found"):
            api.check_pa_status("PA-2026-FAKE0000")

    def test_status_consistency(self, api):
        """Same reference should return same status on repeated calls."""
        sub = _make_submission()
        resp = api.submit_pa(sub)
        s1 = api.check_pa_status(resp.auth_reference_number)
        s2 = api.check_pa_status(resp.auth_reference_number)
        assert s1.status == s2.status

    def test_incomplete_submission_info_requested(self, api):
        sub = _make_submission(include_all_docs=False)
        resp = api.submit_pa(sub)
        status = api.check_pa_status(resp.auth_reference_number)
        assert status.status == PAStatus.INFO_REQUESTED


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


class TestListHelpers:
    def test_list_procedures(self, api):
        procs = api.list_supported_procedures()
        assert len(procs) > 0
        codes = {p["code"] for p in procs}
        assert "J2778" in codes
        assert "66984" in codes
        assert "67108" in codes

    def test_list_payers(self, api):
        payers = api.list_supported_payers()
        assert len(payers) > 0
        names = {p["name"] for p in payers}
        assert "Aetna" in names
        assert "Medicare (CMS)" in names
