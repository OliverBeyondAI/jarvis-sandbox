"""
Comprehensive ophthalmology patient data schemas for prior authorization.

Covers common procedures: intravitreal injections, cataract surgery, and
retinal imaging. Designed with optional fields so inputs can range from
fully complete to minimally populated, reflecting real-world clinical data
variability.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class EyeLaterality(str, Enum):
    OD = "OD"          # Right eye
    OS = "OS"          # Left eye
    OU = "OU"          # Both eyes


class Urgency(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENT = "emergent"


class CompletenessLevel(str, Enum):
    """Metadata tag indicating how populated a record is."""
    COMPLETE = "complete"      # All fields populated
    PARTIAL = "partial"        # Required + some optional fields
    MINIMAL = "minimal"        # Only required fields


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

class PatientDemographics(BaseModel):
    """Core patient identifying information."""
    first_name: str
    last_name: str
    date_of_birth: date
    gender: Gender | None = None
    member_id: str | None = None
    group_number: str | None = None
    phone: str | None = None
    address: str | None = None


class InsuranceInfo(BaseModel):
    """Insurance / payer details."""
    payer_name: str
    plan_type: str | None = None                       # e.g. "HMO", "PPO", "Medicare Part B"
    member_id: str | None = None
    group_number: str | None = None
    policy_holder_name: str | None = None
    policy_holder_relationship: str | None = None      # "self", "spouse", "dependent"
    effective_date: date | None = None
    authorization_phone: str | None = None


class ProviderInfo(BaseModel):
    """Requesting / rendering provider details."""
    provider_name: str
    npi: str | None = None
    specialty: str | None = None
    practice_name: str | None = None
    practice_address: str | None = None
    phone: str | None = None
    fax: str | None = None
    tax_id: str | None = None


class DiagnosisEntry(BaseModel):
    """A single diagnosis with ICD-10 code."""
    icd10_code: str                                    # e.g. "H35.3210"
    description: str                                   # e.g. "Wet AMD, right eye"
    is_primary: bool = False


class ProcedureEntry(BaseModel):
    """A single procedure / service with CPT/HCPCS code."""
    cpt_code: str                                      # e.g. "67028" or "J0178"
    description: str
    laterality: EyeLaterality | None = None
    quantity: int = 1
    unit: str | None = None                            # e.g. "mg", "unit"
    dosage: str | None = None                          # e.g. "2 mg/0.05 mL"
    frequency: str | None = None                       # e.g. "every 8 weeks"


class VisualAcuity(BaseModel):
    """Visual acuity measurement for one eye."""
    eye: EyeLaterality
    best_corrected: str | None = None                  # e.g. "20/80"
    uncorrected: str | None = None
    pinhole: str | None = None
    method: str | None = None                          # "Snellen", "ETDRS"


class OcularExamFinding(BaseModel):
    """A single clinical finding from the ophthalmic exam."""
    eye: EyeLaterality
    structure: str                                     # e.g. "macula", "lens", "retina"
    finding: str                                       # free-text description
    severity: str | None = None                        # "mild", "moderate", "severe"


class ImagingStudy(BaseModel):
    """Diagnostic imaging study result."""
    modality: str                                      # "OCT", "FA", "ICG", "Fundus photo", "B-scan"
    date_performed: date | None = None
    eye: EyeLaterality | None = None
    findings: str | None = None
    central_subfield_thickness_um: int | None = None   # OCT-specific
    subretinal_fluid: bool | None = None               # OCT-specific
    intraretinal_fluid: bool | None = None             # OCT-specific
    pigment_epithelial_detachment: bool | None = None  # OCT-specific
    leakage_pattern: str | None = None                 # FA-specific


class PriorTreatment(BaseModel):
    """A previous treatment relevant to step therapy / medical necessity."""
    treatment_name: str                                # e.g. "Bevacizumab (Avastin)"
    cpt_or_j_code: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    total_doses: int | None = None
    response: str | None = None                        # "improved", "stable", "worsened", "no response"
    reason_discontinued: str | None = None
    adverse_effects: list[str] = Field(default_factory=list)


class ClinicalNote(BaseModel):
    """Free-text clinical note or letter of medical necessity."""
    note_date: date | None = None
    author: str | None = None
    note_type: str | None = None                       # "progress note", "letter of necessity", "consult"
    text: str


# ---------------------------------------------------------------------------
# Top-level Patient Case Schema
# ---------------------------------------------------------------------------

class OphthalmologyPatientCase(BaseModel):
    """
    Complete patient case record for prior authorization.

    This is the top-level schema that aggregates all clinical, demographic,
    insurance, and procedural data needed to process a PA request. Fields
    are optional where real-world data may be incomplete, allowing the
    schema to represent complete, partial, and minimal records.
    """

    # -- Metadata --
    case_id: str | None = None
    completeness: CompletenessLevel | None = None
    submission_date: date | None = None
    urgency: Urgency = Urgency.ROUTINE
    procedure_category: str | None = None              # "intravitreal_injection", "cataract_surgery", "retinal_imaging"

    # -- Core entities --
    patient: PatientDemographics
    insurance: InsuranceInfo | None = None
    requesting_provider: ProviderInfo | None = None
    rendering_provider: ProviderInfo | None = None     # if different from requesting

    # -- Clinical data --
    diagnoses: list[DiagnosisEntry] = Field(default_factory=list)
    procedures: list[ProcedureEntry] = Field(default_factory=list)
    visual_acuity: list[VisualAcuity] = Field(default_factory=list)
    exam_findings: list[OcularExamFinding] = Field(default_factory=list)
    imaging_studies: list[ImagingStudy] = Field(default_factory=list)
    prior_treatments: list[PriorTreatment] = Field(default_factory=list)

    # -- Supporting documentation --
    clinical_notes: list[ClinicalNote] = Field(default_factory=list)
    supporting_documents: list[str] = Field(default_factory=list)  # file references

    # -- Authorization context --
    prior_auth_number: str | None = None               # for renewals / extensions
    requested_duration_months: int | None = None
    requested_total_doses: int | None = None
