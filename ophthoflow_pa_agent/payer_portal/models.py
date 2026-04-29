"""
Data models for the OphthoFlow payer portal API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class PAStatus(str, Enum):
    """Prior authorization status values."""

    REQUIRED = "pa_required"
    NOT_REQUIRED = "pa_not_required"
    SUBMITTED = "submitted"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    DENIED = "denied"
    PARTIALLY_APPROVED = "partially_approved"
    INFO_REQUESTED = "additional_info_requested"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class UrgencyLevel(str, Enum):
    """Urgency classification for PA requests."""

    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENT = "emergent"


@dataclass
class ClinicalInfo:
    """Clinical information submitted with a PA request."""

    diagnosis_codes: list[str]
    visual_acuity_od: Optional[str] = None  # e.g., "20/200"
    visual_acuity_os: Optional[str] = None
    oct_findings: Optional[str] = None
    prior_treatments: list[str] = field(default_factory=list)
    symptoms_duration_days: Optional[int] = None
    functional_impairment: Optional[str] = None
    additional_notes: Optional[str] = None


@dataclass
class PARequirementCheck:
    """Response from checking if a procedure requires prior authorization."""

    procedure_code: str
    procedure_name: str
    payer_id: str
    payer_name: str
    requires_pa: bool
    status: PAStatus
    required_documents: list[str]
    step_therapy_required: bool
    step_therapy_details: Optional[str]
    estimated_review_hours: int
    max_units_per_auth: int
    auth_duration_days: int
    approved_indications: list[str]
    message: str


@dataclass
class PASubmission:
    """A prior authorization submission request."""

    procedure_code: str
    payer_id: str
    member_id: str
    provider_npi: str
    provider_name: str
    patient_name: str
    patient_dob: str
    date_of_service: str
    place_of_service: str  # e.g., "11" (Office), "22" (Outpatient Hospital)
    diagnosis_codes: list[str]
    clinical_info: ClinicalInfo
    urgency: UrgencyLevel = UrgencyLevel.ROUTINE
    quantity: int = 1
    facility_name: Optional[str] = None
    referring_provider_npi: Optional[str] = None


@dataclass
class PASubmissionResponse:
    """Response after submitting a PA request."""

    auth_reference_number: str
    tracking_id: str
    status: PAStatus
    payer_id: str
    payer_name: str
    procedure_code: str
    procedure_name: str
    member_id: str
    submitted_at: str
    estimated_determination_date: str
    urgency: UrgencyLevel
    message: str
    next_steps: list[str]
    documents_received: list[str]
    missing_documents: list[str]


@dataclass
class PAStatusResponse:
    """Response when checking the status of an existing PA request."""

    auth_reference_number: str
    tracking_id: str
    status: PAStatus
    payer_id: str
    payer_name: str
    procedure_code: str
    procedure_name: str
    member_id: str
    submitted_at: str
    last_updated: str
    determination_date: Optional[str]
    approved_units: Optional[int]
    approved_from_date: Optional[str]
    approved_through_date: Optional[str]
    denial_reason: Optional[str]
    appeal_deadline: Optional[str]
    reviewer_notes: Optional[str]
    message: str
