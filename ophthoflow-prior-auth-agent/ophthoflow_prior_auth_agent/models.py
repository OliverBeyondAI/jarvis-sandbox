"""
Pydantic models for structured data flowing through the PA pipeline.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class PAStatus(str, Enum):
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    EXPEDITED = "expedited"
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class PatientRecord(BaseModel):
    """Structured patient record extracted from a clinical note."""

    patient_name: str = ""
    date_of_birth: str = ""
    diagnosis: str = ""
    icd10_codes: list[str] = Field(default_factory=list)
    procedure: str = ""
    cpt_code: str = ""
    payer: str = ""
    clinical_findings: str = ""
    prior_treatments: list[str] = Field(default_factory=list)
    raw_note: str = ""


class PARequirement(BaseModel):
    """Prior authorization requirements from a payer."""

    pa_required: bool = True
    status: PAStatus = PAStatus.REQUIRED
    required_documents: list[str] = Field(default_factory=list)
    step_therapy_met: bool | None = None
    review_timeline_days: int | None = None
    approved_indications: list[str] = Field(default_factory=list)
    notes: str = ""


class PALetter(BaseModel):
    """Generated prior authorization request letter."""

    letter_text: str = ""
    patient_record: PatientRecord | None = None
    requirement: PARequirement | None = None
    confidence_score: float = 0.0
