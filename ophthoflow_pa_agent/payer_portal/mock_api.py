"""
Mock Payer Portal API for OphthoFlow.

Simulates real payer portal behavior for:
- Checking prior authorization requirements
- Submitting PA requests
- Querying PA status and determinations

All responses use realistic data structures and ophthalmology-specific
clinical criteria to enable end-to-end testing and development.
"""

from __future__ import annotations

import hashlib
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

from .models import (
    ClinicalInfo,
    PARequirementCheck,
    PAStatus,
    PAStatusResponse,
    PASubmission,
    PASubmissionResponse,
    UrgencyLevel,
)
from .procedure_data import (
    DIAGNOSIS_DESCRIPTIONS,
    PAYER_PROFILES,
    PROCEDURE_CATALOG,
)


class PayerPortalAPI:
    """
    Mock payer portal API that simulates checking PA requirements
    and accepting PA submissions for ophthalmology procedures.

    Usage:
        api = PayerPortalAPI()

        # Check if a procedure needs PA
        result = api.check_pa_requirement("J2778", "AETNA")

        # Submit a PA request
        submission = PASubmission(
            procedure_code="J2778",
            payer_id="AETNA",
            member_id="AET123456789",
            provider_npi="1234567890",
            ...
        )
        response = api.submit_pa(submission)

        # Check status
        status = api.check_pa_status("PA-2026-ABC123")
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the mock API.

        Args:
            seed: Optional random seed for reproducible responses.
        """
        self._rng = random.Random(seed)
        self._submitted_pas: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def check_pa_requirement(
        self, procedure_code: str, payer_key: str
    ) -> PARequirementCheck:
        """
        Check whether a procedure requires prior authorization for a given payer.

        Args:
            procedure_code: CPT or HCPCS code (e.g., "J2778", "66984").
            payer_key: Payer identifier key (e.g., "AETNA", "UNITEDHEALTHCARE").

        Returns:
            PARequirementCheck with full details about PA requirements.

        Raises:
            ValueError: If the procedure code or payer key is not recognized.
        """
        procedure = self._get_procedure(procedure_code)
        payer = self._get_payer(payer_key)

        requires_pa = procedure["requires_pa"]
        criteria = procedure["clinical_criteria"]

        # Some payers enforce step therapy for anti-VEGF agents
        step_therapy_required = False
        step_therapy_details = None
        if procedure["category"] == "anti-vegf" and payer.get("anti_vegf_step_therapy"):
            preferred = payer.get("preferred_anti_vegf")
            if preferred and procedure_code != preferred:
                step_therapy_required = True
                preferred_name = PROCEDURE_CATALOG[preferred]["name"]
                step_therapy_details = (
                    f"{payer['name']} requires trial of {preferred_name} "
                    f"({preferred}) before authorizing {procedure['name']}. "
                    f"Submit documentation of prior treatment and clinical "
                    f"rationale if step therapy exception is requested."
                )

        if requires_pa:
            status = PAStatus.REQUIRED
            message = (
                f"Prior authorization IS required for {procedure['name']} "
                f"({procedure_code}) under {payer['name']}. "
                f"Estimated review time: {criteria['review_turnaround_hours']} hours."
            )
        else:
            status = PAStatus.NOT_REQUIRED
            message = (
                f"Prior authorization is NOT required for {procedure['name']} "
                f"({procedure_code}) under {payer['name']}. "
                f"Procedure may be performed without pre-approval."
            )

        return PARequirementCheck(
            procedure_code=procedure_code,
            procedure_name=procedure["name"],
            payer_id=payer["payer_id"],
            payer_name=payer["name"],
            requires_pa=requires_pa,
            status=status,
            required_documents=criteria["required_docs"],
            step_therapy_required=step_therapy_required,
            step_therapy_details=step_therapy_details,
            estimated_review_hours=criteria["review_turnaround_hours"],
            max_units_per_auth=criteria["max_units_per_auth"],
            auth_duration_days=criteria["auth_duration_days"],
            approved_indications=procedure["approved_indications"],
            message=message,
        )

    def submit_pa(self, submission: PASubmission) -> PASubmissionResponse:
        """
        Submit a prior authorization request.

        Args:
            submission: A fully populated PASubmission object.

        Returns:
            PASubmissionResponse with tracking info and initial determination.

        Raises:
            ValueError: If procedure code or payer key is not recognized.
        """
        procedure = self._get_procedure(submission.procedure_code)
        payer = self._get_payer(submission.payer_id)
        criteria = procedure["clinical_criteria"]

        # Generate reference numbers
        auth_ref = self._generate_auth_reference()
        tracking_id = f"TRK-{uuid.uuid4().hex[:8].upper()}"

        now = datetime.now()
        submitted_at = now.isoformat()

        # Determine review timeline based on urgency
        if submission.urgency == UrgencyLevel.EMERGENT:
            review_hours = max(4, criteria["review_turnaround_hours"] // 6)
        elif submission.urgency == UrgencyLevel.URGENT:
            review_hours = max(8, criteria["review_turnaround_hours"] // 2)
        else:
            review_hours = criteria["review_turnaround_hours"]

        est_determination = now + timedelta(hours=review_hours)

        # Validate submitted documents against requirements
        submitted_docs = self._infer_documents_from_clinical_info(
            submission.clinical_info
        )
        missing_docs = [
            doc for doc in criteria["required_docs"] if doc not in submitted_docs
        ]

        # Determine initial status
        if missing_docs:
            status = PAStatus.INFO_REQUESTED
            message = (
                f"PA request {auth_ref} received but additional documentation "
                f"is required. Please submit: {', '.join(missing_docs)}."
            )
            next_steps = [
                f"Submit missing documentation: {doc}" for doc in missing_docs
            ] + [
                f"Fax documents to {payer['name']} PA department",
                f"Reference tracking ID {tracking_id} on all correspondence",
            ]
        else:
            status = PAStatus.PENDING_REVIEW
            message = (
                f"PA request {auth_ref} submitted successfully to {payer['name']}. "
                f"All required documentation received. "
                f"Estimated determination by {est_determination.strftime('%Y-%m-%d %H:%M')}."
            )
            next_steps = [
                f"Check status using reference number {auth_ref}",
                f"Expected determination by {est_determination.strftime('%Y-%m-%d')}",
                "No additional action required at this time",
            ]

        # Store for status lookups
        self._submitted_pas[auth_ref] = {
            "submission": submission,
            "tracking_id": tracking_id,
            "status": status,
            "submitted_at": submitted_at,
            "estimated_determination": est_determination.isoformat(),
            "procedure": procedure,
            "payer": payer,
            "missing_docs": missing_docs,
            "submitted_docs": submitted_docs,
        }

        return PASubmissionResponse(
            auth_reference_number=auth_ref,
            tracking_id=tracking_id,
            status=status,
            payer_id=payer["payer_id"],
            payer_name=payer["name"],
            procedure_code=submission.procedure_code,
            procedure_name=procedure["name"],
            member_id=submission.member_id,
            submitted_at=submitted_at,
            estimated_determination_date=est_determination.isoformat(),
            urgency=submission.urgency,
            message=message,
            next_steps=next_steps,
            documents_received=submitted_docs,
            missing_documents=missing_docs,
        )

    def check_pa_status(self, auth_reference_number: str) -> PAStatusResponse:
        """
        Check the status of a previously submitted PA request.

        For mock purposes, this simulates progression through the
        determination lifecycle. Calling it multiple times may return
        different statuses to simulate real-world progression.

        Args:
            auth_reference_number: The PA reference number from submission.

        Returns:
            PAStatusResponse with current status and determination details.

        Raises:
            ValueError: If the reference number is not found.
        """
        if auth_reference_number not in self._submitted_pas:
            raise ValueError(
                f"PA reference number '{auth_reference_number}' not found. "
                f"Valid references: {list(self._submitted_pas.keys())}"
            )

        record = self._submitted_pas[auth_reference_number]
        submission: PASubmission = record["submission"]
        procedure = record["procedure"]
        payer = record["payer"]
        now = datetime.now()

        # Simulate status progression
        current_status = self._simulate_determination(record)

        # Build response based on simulated status
        approved_units = None
        approved_from = None
        approved_through = None
        denial_reason = None
        appeal_deadline = None
        reviewer_notes = None
        determination_date = None

        if current_status == PAStatus.APPROVED:
            criteria = procedure["clinical_criteria"]
            determination_date = now.isoformat()
            approved_units = min(
                submission.quantity, criteria["max_units_per_auth"]
            )
            approved_from = submission.date_of_service
            from_date = datetime.fromisoformat(submission.date_of_service)
            approved_through = (
                from_date + timedelta(days=criteria["auth_duration_days"])
            ).strftime("%Y-%m-%d")
            reviewer_notes = "Meets medical necessity criteria. Approved as submitted."
            message = (
                f"PA {auth_reference_number} has been APPROVED. "
                f"Authorized for {approved_units} unit(s) from "
                f"{approved_from} through {approved_through}."
            )

        elif current_status == PAStatus.DENIED:
            determination_date = now.isoformat()
            denial_reason = self._rng.choice([
                "Does not meet medical necessity criteria per payer policy.",
                "Insufficient clinical documentation to support medical necessity.",
                "Step therapy requirement not met — trial of preferred agent required.",
                "Diagnosis code does not match approved indications for this procedure.",
            ])
            appeal_deadline = (now + timedelta(days=60)).strftime("%Y-%m-%d")
            reviewer_notes = denial_reason
            message = (
                f"PA {auth_reference_number} has been DENIED. "
                f"Reason: {denial_reason} "
                f"Appeal deadline: {appeal_deadline}."
            )

        elif current_status == PAStatus.PARTIALLY_APPROVED:
            criteria = procedure["clinical_criteria"]
            determination_date = now.isoformat()
            approved_units = max(1, submission.quantity // 2)
            approved_from = submission.date_of_service
            from_date = datetime.fromisoformat(submission.date_of_service)
            approved_through = (
                from_date + timedelta(days=criteria["auth_duration_days"] // 2)
            ).strftime("%Y-%m-%d")
            reviewer_notes = (
                "Partially approved — reduced units authorized. "
                "Resubmit with updated clinical data for additional units."
            )
            message = (
                f"PA {auth_reference_number} has been PARTIALLY APPROVED. "
                f"Authorized for {approved_units} unit(s) from "
                f"{approved_from} through {approved_through}. "
                f"Submit additional documentation for remaining units."
            )

        elif current_status == PAStatus.INFO_REQUESTED:
            message = (
                f"PA {auth_reference_number} requires additional information. "
                f"Missing: {', '.join(record['missing_docs'])}. "
                f"Please submit within 14 days to avoid cancellation."
            )

        else:  # PENDING_REVIEW
            message = (
                f"PA {auth_reference_number} is under review by {payer['name']}. "
                f"Estimated determination by "
                f"{record['estimated_determination']}."
            )

        return PAStatusResponse(
            auth_reference_number=auth_reference_number,
            tracking_id=record["tracking_id"],
            status=current_status,
            payer_id=payer["payer_id"],
            payer_name=payer["name"],
            procedure_code=submission.procedure_code,
            procedure_name=procedure["name"],
            member_id=submission.member_id,
            submitted_at=record["submitted_at"],
            last_updated=now.isoformat(),
            determination_date=determination_date,
            approved_units=approved_units,
            approved_from_date=approved_from,
            approved_through_date=approved_through,
            denial_reason=denial_reason,
            appeal_deadline=appeal_deadline,
            reviewer_notes=reviewer_notes,
            message=message,
        )

    def list_supported_procedures(self) -> list[dict]:
        """Return a summary of all supported procedures."""
        return [
            {
                "code": code,
                "name": info["name"],
                "category": info["category"],
                "requires_pa": info["requires_pa"],
            }
            for code, info in PROCEDURE_CATALOG.items()
        ]

    def list_supported_payers(self) -> list[dict]:
        """Return a summary of all supported payers."""
        return [
            {
                "key": key,
                "name": info["name"],
                "payer_id": info["payer_id"],
                "supports_electronic_pa": info["supports_electronic_pa"],
            }
            for key, info in PAYER_PROFILES.items()
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_procedure(self, code: str) -> dict:
        if code not in PROCEDURE_CATALOG:
            available = ", ".join(PROCEDURE_CATALOG.keys())
            raise ValueError(
                f"Unknown procedure code '{code}'. "
                f"Supported codes: {available}"
            )
        return PROCEDURE_CATALOG[code]

    def _get_payer(self, key: str) -> dict:
        normalized = key.upper().replace(" ", "").replace("-", "")
        if normalized not in PAYER_PROFILES:
            available = ", ".join(PAYER_PROFILES.keys())
            raise ValueError(
                f"Unknown payer '{key}'. Supported payers: {available}"
            )
        return PAYER_PROFILES[normalized]

    def _generate_auth_reference(self) -> str:
        uid = uuid.uuid4().hex[:8].upper()
        return f"PA-2026-{uid}"

    def _infer_documents_from_clinical_info(
        self, clinical: ClinicalInfo
    ) -> list[str]:
        """Map clinical info fields to document categories."""
        docs = []
        if clinical.visual_acuity_od or clinical.visual_acuity_os:
            docs.append("Visual acuity measurement")
            docs.append("Visual acuity (corrected and uncorrected)")
        if clinical.oct_findings:
            docs.append("OCT showing macular pathology")
            docs.append("OCT showing epiretinal membrane with macular distortion")
        if clinical.diagnosis_codes:
            docs.append("Fundus examination findings")
            docs.append("Dilated fundus examination documenting detachment")
            docs.append(
                "Dilated fundus examination documenting detachment and break location"
            )
            docs.append(
                "Dilated fundus examination documenting detachment and PVR"
            )
            docs.append("Slit-lamp examination documenting cataract grade")
            docs.append(
                "Slit-lamp examination documenting cataract and complicating factors"
            )
            docs.append(
                "Fundus examination documenting retinopathy severity"
            )
        if clinical.functional_impairment:
            docs.append("Documentation of functional impairment")
            docs.append("Documentation of why complex technique is required")
            docs.append("Glare testing if VA better than 20/40")
        if clinical.prior_treatments:
            docs.append("Documentation of prior anti-VEGF therapy and response")
            docs.append("Prior surgical history")
        if clinical.symptoms_duration_days is not None:
            docs.append("Duration of symptoms and macular status")
            docs.append("Duration of symptoms")
        if clinical.additional_notes:
            docs.append("B-scan ultrasonography if view is obscured")
            docs.append("B-scan ultrasonography")
            docs.append("Fluorescein angiography if applicable")
            docs.append("IOP measurements")
            docs.append("Visual field testing")
            docs.append("Gonioscopy confirming open angles")
        return docs

    def _simulate_determination(self, record: dict) -> PAStatus:
        """
        Simulate PA status progression for mock purposes.

        The outcome is deterministic based on the auth reference hash
        so repeated calls return consistent results.
        """
        # If missing docs, stay in info-requested state
        if record["missing_docs"]:
            return PAStatus.INFO_REQUESTED

        # Use a hash of the tracking ID to produce consistent outcomes
        hash_val = int(
            hashlib.md5(record["tracking_id"].encode()).hexdigest(), 16
        )
        outcome_roll = hash_val % 100

        # 70% approved, 15% denied, 10% partially approved, 5% still pending
        if outcome_roll < 70:
            return PAStatus.APPROVED
        elif outcome_roll < 85:
            return PAStatus.DENIED
        elif outcome_roll < 95:
            return PAStatus.PARTIALLY_APPROVED
        else:
            return PAStatus.PENDING_REVIEW
