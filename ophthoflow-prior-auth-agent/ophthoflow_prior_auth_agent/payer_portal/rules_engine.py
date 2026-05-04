"""
Payer Rules Engine for Ophthalmology Prior Authorization.

Encodes typical PA requirements for common ophthalmology procedures across
major payers, including:
- Required documentation checklists
- Clinical criteria / approved indications
- Step-therapy protocols
- Review timelines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class StepTherapyRule:
    """Defines a step-therapy requirement for a procedure."""

    required_prior_agents: list[str]
    """Agents/treatments that must be tried first."""

    min_doses_required: int = 3
    """Minimum number of doses of the prior agent before escalation."""

    exception_criteria: list[str] = field(default_factory=list)
    """Clinical conditions that bypass the step-therapy requirement."""


@dataclass
class ClinicalCriterion:
    """A single clinical criterion for PA approval."""

    description: str
    icd10_codes: list[str] = field(default_factory=list)
    """ICD-10 codes that satisfy this criterion."""

    required: bool = True
    """If True, this criterion MUST be met. If False, it's supporting evidence."""


@dataclass
class ProcedureRule:
    """Complete PA rule set for a specific procedure under a specific payer."""

    payer: str
    cpt_code: str
    procedure_name: str

    pa_required: bool = True
    review_timeline_days: int = 5

    approved_indications: list[str] = field(default_factory=list)
    clinical_criteria: list[ClinicalCriterion] = field(default_factory=list)
    required_documents: list[str] = field(default_factory=list)
    step_therapy: StepTherapyRule | None = None

    urgency_expedite_days: int = 2
    """Timeline for urgent/expedited reviews."""

    notes: str = ""
    """Additional payer-specific notes or guidance."""


@dataclass
class PACheckResult:
    """Result of checking PA requirements."""

    pa_required: bool
    required_documents: list[str]
    step_therapy_met: bool | None
    step_therapy_details: str | None
    review_timeline_days: int
    approved_indications: list[str]
    clinical_criteria_met: list[str]
    clinical_criteria_unmet: list[str]
    notes: str


# ---------------------------------------------------------------------------
# Procedure Definitions (shared across payers)
# ---------------------------------------------------------------------------

PROCEDURE_NAMES: dict[str, str] = {
    "J2778": "Ranibizumab (Lucentis) intravitreal injection",
    "J0178": "Aflibercept (Eylea) intravitreal injection",
    "J9035": "Bevacizumab (Avastin) intravitreal injection",
    "J2503": "Faricimab (Vabysmo) intravitreal injection",
    "66984": "Cataract surgery — phacoemulsification (standard)",
    "66982": "Cataract surgery — extracapsular (complex)",
    "67108": "Vitrectomy for retinal detachment repair",
    "67228": "Panretinal photocoagulation (PRP)",
}

# ICD-10 codes commonly used in ophthalmology PA
ICD10_WET_AMD = ["H35.31", "H35.3210", "H35.3211", "H35.3212", "H35.3220", "H35.3221", "H35.3222"]
ICD10_DME = ["E11.311", "E11.3211", "E11.3212", "E11.3213", "E11.3291", "E11.3292", "E11.3293"]
ICD10_RVO = ["H34.811", "H34.812", "H34.813", "H34.831", "H34.832", "H34.833"]
ICD10_DR = ["E11.319", "E11.3211", "E11.3212", "E11.3213", "E11.3411", "E11.3412", "E11.3413"]
ICD10_CATARACT = ["H25.10", "H25.11", "H25.12", "H25.13", "H25.20", "H25.811", "H25.812", "H25.813"]
ICD10_RETINAL_DETACH = ["H33.001", "H33.002", "H33.003", "H33.011", "H33.012", "H33.013"]


# ---------------------------------------------------------------------------
# Common Documentation Requirements
# ---------------------------------------------------------------------------

ANTI_VEGF_BASE_DOCS = [
    "Clinical notes documenting diagnosis and visual acuity",
    "OCT (Optical Coherence Tomography) imaging within 30 days",
    "Fluorescein angiography (FA) or OCT-A confirming pathology",
    "ICD-10 diagnosis code(s)",
    "CPT/HCPCS procedure code",
    "Treatment history and prior authorization numbers (if applicable)",
]

ANTI_VEGF_STEP_THERAPY_DOCS = [
    "Documentation of prior bevacizumab (Avastin) treatment",
    "Number of bevacizumab injections administered",
    "Clinical response (or lack thereof) to bevacizumab",
    "OCT showing persistent/worsening edema despite bevacizumab",
]

CATARACT_BASE_DOCS = [
    "Visual acuity testing (best corrected)",
    "Documentation of functional impairment",
    "Slit-lamp examination findings",
    "Dilated fundus exam",
    "Manifest refraction",
    "ICD-10 diagnosis code(s)",
]

VITRECTOMY_DOCS = [
    "Clinical notes documenting retinal detachment",
    "B-scan ultrasonography or fundus photography",
    "Documentation of symptoms (floaters, flashes, visual field loss)",
    "Dilated fundus examination findings",
    "Urgency assessment and time since symptom onset",
]

PRP_DOCS = [
    "Clinical notes documenting proliferative diabetic retinopathy",
    "Fluorescein angiography showing neovascularization",
    "Dilated fundus exam findings",
    "HbA1c results (within 3 months)",
    "Documentation of diabetes management plan",
]


# ---------------------------------------------------------------------------
# Payer-Specific Rules
# ---------------------------------------------------------------------------


def _build_aetna_rules() -> list[ProcedureRule]:
    """Aetna — step-therapy required for brand anti-VEGFs."""
    return [
        ProcedureRule(
            payer="Aetna",
            cpt_code="J2778",
            procedure_name=PROCEDURE_NAMES["J2778"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
                "Myopic choroidal neovascularization (mCNV)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed neovascular pathology on OCT or FA",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Visual acuity between 20/40 and 20/320 in affected eye",
                    required=True,
                ),
                ClinicalCriterion(
                    description="Failure or inadequate response to bevacizumab (Avastin) therapy",
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS + ANTI_VEGF_STEP_THERAPY_DOCS,
            step_therapy=StepTherapyRule(
                required_prior_agents=["bevacizumab", "Avastin"],
                min_doses_required=3,
                exception_criteria=[
                    "Documented allergy or contraindication to bevacizumab",
                    "Patient enrolled in clinical trial requiring specific agent",
                    "Macular hemorrhage threatening fovea (urgent treatment needed)",
                ],
            ),
        ),
        ProcedureRule(
            payer="Aetna",
            cpt_code="J0178",
            procedure_name=PROCEDURE_NAMES["J0178"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Diabetic retinopathy (DR)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed neovascular pathology on OCT or FA",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Failure or inadequate response to bevacizumab (Avastin) therapy",
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS + ANTI_VEGF_STEP_THERAPY_DOCS,
            step_therapy=StepTherapyRule(
                required_prior_agents=["bevacizumab", "Avastin"],
                min_doses_required=3,
                exception_criteria=[
                    "Documented allergy or contraindication to bevacizumab",
                    "Patient enrolled in clinical trial requiring specific agent",
                    "Macular hemorrhage threatening fovea (urgent treatment needed)",
                ],
            ),
        ),
        ProcedureRule(
            payer="Aetna",
            cpt_code="J2503",
            procedure_name=PROCEDURE_NAMES["J2503"],
            pa_required=True,
            review_timeline_days=7,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed neovascular pathology on OCT or FA",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Failure or inadequate response to bevacizumab (Avastin) therapy",
                    required=True,
                ),
                ClinicalCriterion(
                    description="Failure or suboptimal response to at least one other anti-VEGF agent",
                    required=False,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS + ANTI_VEGF_STEP_THERAPY_DOCS + [
                "Documentation of prior anti-VEGF treatment history (all agents tried)",
            ],
            step_therapy=StepTherapyRule(
                required_prior_agents=["bevacizumab", "Avastin"],
                min_doses_required=3,
                exception_criteria=[
                    "Documented allergy or contraindication to bevacizumab",
                    "Documented allergy or contraindication to both ranibizumab and aflibercept",
                ],
            ),
        ),
        ProcedureRule(
            payer="Aetna",
            cpt_code="J9035",
            procedure_name=PROCEDURE_NAMES["J9035"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
                "Proliferative diabetic retinopathy (PDR)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed neovascular or edematous pathology",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
            ],
            required_documents=[],
        ),
        ProcedureRule(
            payer="Aetna",
            cpt_code="66984",
            procedure_name=PROCEDURE_NAMES["66984"],
            pa_required=True,
            review_timeline_days=10,
            approved_indications=[
                "Visually significant cataract causing functional impairment",
                "Best corrected visual acuity 20/50 or worse",
                "Cataract interfering with diagnosis/treatment of posterior segment disease",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="BCVA 20/50 or worse in the operative eye",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Documented functional impairment affecting daily activities",
                    required=True,
                ),
                ClinicalCriterion(
                    description="Glare testing or contrast sensitivity documenting impairment",
                    required=False,
                ),
            ],
            required_documents=CATARACT_BASE_DOCS,
        ),
        ProcedureRule(
            payer="Aetna",
            cpt_code="66982",
            procedure_name=PROCEDURE_NAMES["66982"],
            pa_required=True,
            review_timeline_days=10,
            approved_indications=[
                "Complex cataract requiring extracapsular technique",
                "Dense brunescent/white cataract",
                "Cataract with small pupil, pseudoexfoliation, or zonular weakness",
                "Pediatric cataract",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="BCVA 20/50 or worse in the operative eye",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Documentation of complicating factor requiring complex technique",
                    required=True,
                ),
            ],
            required_documents=CATARACT_BASE_DOCS + [
                "Documentation of complicating factors (dense nucleus, zonular weakness, etc.)",
            ],
        ),
        ProcedureRule(
            payer="Aetna",
            cpt_code="67108",
            procedure_name=PROCEDURE_NAMES["67108"],
            pa_required=True,
            review_timeline_days=3,
            approved_indications=[
                "Rhegmatogenous retinal detachment",
                "Tractional retinal detachment threatening macula",
                "Combined tractional/rhegmatogenous retinal detachment",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Documented retinal detachment on clinical exam or imaging",
                    icd10_codes=ICD10_RETINAL_DETACH,
                    required=True,
                ),
            ],
            required_documents=VITRECTOMY_DOCS,
            urgency_expedite_days=1,
        ),
        ProcedureRule(
            payer="Aetna",
            cpt_code="67228",
            procedure_name=PROCEDURE_NAMES["67228"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Proliferative diabetic retinopathy (PDR) with neovascularization",
                "Severe non-proliferative diabetic retinopathy (high-risk)",
                "Neovascularization secondary to retinal vein occlusion",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Neovascularization confirmed on FA or clinical exam",
                    icd10_codes=ICD10_DR,
                    required=True,
                ),
            ],
            required_documents=PRP_DOCS,
        ),
    ]


def _build_unitedhealthcare_rules() -> list[ProcedureRule]:
    """UnitedHealthcare — no step-therapy for anti-VEGFs."""
    return [
        ProcedureRule(
            payer="UnitedHealthcare",
            cpt_code="J2778",
            procedure_name=PROCEDURE_NAMES["J2778"],
            pa_required=True,
            review_timeline_days=3,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
                "Myopic choroidal neovascularization (mCNV)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed diagnosis with OCT or FA within 30 days",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS,
        ),
        ProcedureRule(
            payer="UnitedHealthcare",
            cpt_code="J0178",
            procedure_name=PROCEDURE_NAMES["J0178"],
            pa_required=True,
            review_timeline_days=3,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Diabetic retinopathy (DR)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed diagnosis with OCT or FA within 30 days",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO + ICD10_DR,
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS,
        ),
        ProcedureRule(
            payer="UnitedHealthcare",
            cpt_code="J2503",
            procedure_name=PROCEDURE_NAMES["J2503"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed diagnosis with OCT or FA within 30 days",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME,
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS,
        ),
        ProcedureRule(
            payer="UnitedHealthcare",
            cpt_code="J9035",
            procedure_name=PROCEDURE_NAMES["J9035"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[],
            required_documents=[],
        ),
        ProcedureRule(
            payer="UnitedHealthcare",
            cpt_code="66984",
            procedure_name=PROCEDURE_NAMES["66984"],
            pa_required=True,
            review_timeline_days=7,
            approved_indications=[
                "Visually significant cataract with BCVA 20/50 or worse",
                "Cataract causing documented functional impairment",
                "Cataract preventing adequate fundus visualization for treatment",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="BCVA 20/50 or worse OR documented functional impairment",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
            ],
            required_documents=CATARACT_BASE_DOCS,
        ),
        ProcedureRule(
            payer="UnitedHealthcare",
            cpt_code="66982",
            procedure_name=PROCEDURE_NAMES["66982"],
            pa_required=True,
            review_timeline_days=7,
            approved_indications=[
                "Complex cataract with complicating factors",
                "Pediatric or traumatic cataract",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Documentation of factors requiring complex extraction technique",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
            ],
            required_documents=CATARACT_BASE_DOCS + [
                "Documentation of complicating factors",
            ],
        ),
        ProcedureRule(
            payer="UnitedHealthcare",
            cpt_code="67108",
            procedure_name=PROCEDURE_NAMES["67108"],
            pa_required=True,
            review_timeline_days=2,
            approved_indications=[
                "Rhegmatogenous retinal detachment",
                "Tractional retinal detachment",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Documented retinal detachment",
                    icd10_codes=ICD10_RETINAL_DETACH,
                    required=True,
                ),
            ],
            required_documents=VITRECTOMY_DOCS,
            urgency_expedite_days=1,
        ),
        ProcedureRule(
            payer="UnitedHealthcare",
            cpt_code="67228",
            procedure_name=PROCEDURE_NAMES["67228"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Proliferative diabetic retinopathy with neovascularization",
                "High-risk non-proliferative diabetic retinopathy",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Neovascularization or high-risk features on FA/exam",
                    icd10_codes=ICD10_DR,
                    required=True,
                ),
            ],
            required_documents=PRP_DOCS,
        ),
    ]


def _build_cigna_rules() -> list[ProcedureRule]:
    """Cigna — step-therapy for brand anti-VEGFs (similar to Aetna)."""
    return [
        ProcedureRule(
            payer="Cigna",
            cpt_code="J2778",
            procedure_name=PROCEDURE_NAMES["J2778"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed neovascular pathology on imaging",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Inadequate response to bevacizumab after minimum 3 injections",
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS + ANTI_VEGF_STEP_THERAPY_DOCS,
            step_therapy=StepTherapyRule(
                required_prior_agents=["bevacizumab", "Avastin"],
                min_doses_required=3,
                exception_criteria=[
                    "Documented allergy/contraindication to bevacizumab",
                    "Active endophthalmitis or ocular infection precluding bevacizumab",
                    "Threat of imminent vision loss requiring immediate branded agent",
                ],
            ),
        ),
        ProcedureRule(
            payer="Cigna",
            cpt_code="J0178",
            procedure_name=PROCEDURE_NAMES["J0178"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed neovascular pathology on imaging",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Inadequate response to bevacizumab after minimum 3 injections",
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS + ANTI_VEGF_STEP_THERAPY_DOCS,
            step_therapy=StepTherapyRule(
                required_prior_agents=["bevacizumab", "Avastin"],
                min_doses_required=3,
                exception_criteria=[
                    "Documented allergy/contraindication to bevacizumab",
                    "Threat of imminent vision loss requiring immediate branded agent",
                ],
            ),
        ),
        ProcedureRule(
            payer="Cigna",
            cpt_code="J2503",
            procedure_name=PROCEDURE_NAMES["J2503"],
            pa_required=True,
            review_timeline_days=7,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed diagnosis on imaging",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Failure of bevacizumab AND at least one other anti-VEGF",
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS + ANTI_VEGF_STEP_THERAPY_DOCS + [
                "Documentation of all prior anti-VEGF agents tried with response data",
            ],
            step_therapy=StepTherapyRule(
                required_prior_agents=["bevacizumab", "Avastin"],
                min_doses_required=3,
                exception_criteria=[
                    "Documented contraindication to all other anti-VEGF agents",
                ],
            ),
        ),
        ProcedureRule(
            payer="Cigna",
            cpt_code="J9035",
            procedure_name=PROCEDURE_NAMES["J9035"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[],
            required_documents=[],
        ),
        ProcedureRule(
            payer="Cigna",
            cpt_code="66984",
            procedure_name=PROCEDURE_NAMES["66984"],
            pa_required=True,
            review_timeline_days=10,
            approved_indications=[
                "Visually significant cataract with BCVA 20/50 or worse",
                "Cataract causing functional impairment documented by provider",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="BCVA 20/50 or worse in operative eye",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
                ClinicalCriterion(
                    description="Functional impairment affecting daily activities",
                    required=True,
                ),
            ],
            required_documents=CATARACT_BASE_DOCS,
        ),
        ProcedureRule(
            payer="Cigna",
            cpt_code="66982",
            procedure_name=PROCEDURE_NAMES["66982"],
            pa_required=True,
            review_timeline_days=10,
            approved_indications=[
                "Complex cataract with complicating surgical factors",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Complicating factors documented (zonular weakness, dense nucleus, etc.)",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
            ],
            required_documents=CATARACT_BASE_DOCS + [
                "Documentation of complicating factors requiring complex technique",
            ],
        ),
        ProcedureRule(
            payer="Cigna",
            cpt_code="67108",
            procedure_name=PROCEDURE_NAMES["67108"],
            pa_required=True,
            review_timeline_days=3,
            approved_indications=[
                "Rhegmatogenous retinal detachment",
                "Tractional retinal detachment threatening macula",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Documented retinal detachment on exam or imaging",
                    icd10_codes=ICD10_RETINAL_DETACH,
                    required=True,
                ),
            ],
            required_documents=VITRECTOMY_DOCS,
            urgency_expedite_days=1,
        ),
        ProcedureRule(
            payer="Cigna",
            cpt_code="67228",
            procedure_name=PROCEDURE_NAMES["67228"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Proliferative diabetic retinopathy (PDR)",
                "Severe NPDR at high risk of progression",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Neovascularization or high-risk features documented",
                    icd10_codes=ICD10_DR,
                    required=True,
                ),
            ],
            required_documents=PRP_DOCS,
        ),
    ]


def _build_bcbs_rules() -> list[ProcedureRule]:
    """Blue Cross Blue Shield — no step-therapy, standard PA."""
    return [
        ProcedureRule(
            payer="Blue Cross Blue Shield",
            cpt_code="J2778",
            procedure_name=PROCEDURE_NAMES["J2778"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
                "Myopic choroidal neovascularization (mCNV)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed diagnosis with OCT or FA",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS,
        ),
        ProcedureRule(
            payer="Blue Cross Blue Shield",
            cpt_code="J0178",
            procedure_name=PROCEDURE_NAMES["J0178"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed diagnosis with OCT or FA",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS,
        ),
        ProcedureRule(
            payer="Blue Cross Blue Shield",
            cpt_code="J2503",
            procedure_name=PROCEDURE_NAMES["J2503"],
            pa_required=True,
            review_timeline_days=7,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Confirmed diagnosis with OCT or FA",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME,
                    required=True,
                ),
            ],
            required_documents=ANTI_VEGF_BASE_DOCS,
        ),
        ProcedureRule(
            payer="Blue Cross Blue Shield",
            cpt_code="J9035",
            procedure_name=PROCEDURE_NAMES["J9035"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[],
            required_documents=[],
        ),
        ProcedureRule(
            payer="Blue Cross Blue Shield",
            cpt_code="66984",
            procedure_name=PROCEDURE_NAMES["66984"],
            pa_required=True,
            review_timeline_days=7,
            approved_indications=[
                "Visually significant cataract (BCVA 20/50 or worse)",
                "Cataract causing functional impairment",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="BCVA 20/50 or worse OR functional impairment documented",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
            ],
            required_documents=CATARACT_BASE_DOCS,
        ),
        ProcedureRule(
            payer="Blue Cross Blue Shield",
            cpt_code="66982",
            procedure_name=PROCEDURE_NAMES["66982"],
            pa_required=True,
            review_timeline_days=7,
            approved_indications=[
                "Complex cataract with documented complicating factors",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Complicating factors requiring complex extraction",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
            ],
            required_documents=CATARACT_BASE_DOCS + [
                "Documentation of complicating factors",
            ],
        ),
        ProcedureRule(
            payer="Blue Cross Blue Shield",
            cpt_code="67108",
            procedure_name=PROCEDURE_NAMES["67108"],
            pa_required=True,
            review_timeline_days=3,
            approved_indications=[
                "Retinal detachment (rhegmatogenous or tractional)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Documented retinal detachment",
                    icd10_codes=ICD10_RETINAL_DETACH,
                    required=True,
                ),
            ],
            required_documents=VITRECTOMY_DOCS,
            urgency_expedite_days=1,
        ),
        ProcedureRule(
            payer="Blue Cross Blue Shield",
            cpt_code="67228",
            procedure_name=PROCEDURE_NAMES["67228"],
            pa_required=True,
            review_timeline_days=5,
            approved_indications=[
                "Proliferative diabetic retinopathy with neovascularization",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Neovascularization documented on FA or clinical exam",
                    icd10_codes=ICD10_DR,
                    required=True,
                ),
            ],
            required_documents=PRP_DOCS,
        ),
    ]


def _build_medicare_rules() -> list[ProcedureRule]:
    """Medicare Part B — no step-therapy, standard medical necessity criteria."""
    return [
        ProcedureRule(
            payer="Medicare Part B",
            cpt_code="J2778",
            procedure_name=PROCEDURE_NAMES["J2778"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="FDA-approved indication documented",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_RVO,
                    required=True,
                ),
            ],
            required_documents=[
                "Clinical notes supporting medical necessity",
                "ICD-10 diagnosis code(s) on claim",
            ],
            notes="Medicare Part B covers anti-VEGF agents under Part B buy-and-bill; "
                  "no formal PA required but medical necessity must be documented.",
        ),
        ProcedureRule(
            payer="Medicare Part B",
            cpt_code="J0178",
            procedure_name=PROCEDURE_NAMES["J0178"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
                "Diabetic retinopathy (DR)",
                "Retinal vein occlusion (RVO) with macular edema",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="FDA-approved indication documented",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME + ICD10_DR + ICD10_RVO,
                    required=True,
                ),
            ],
            required_documents=[
                "Clinical notes supporting medical necessity",
                "ICD-10 diagnosis code(s) on claim",
            ],
        ),
        ProcedureRule(
            payer="Medicare Part B",
            cpt_code="J2503",
            procedure_name=PROCEDURE_NAMES["J2503"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Neovascular (wet) age-related macular degeneration (wAMD)",
                "Diabetic macular edema (DME)",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="FDA-approved indication documented",
                    icd10_codes=ICD10_WET_AMD + ICD10_DME,
                    required=True,
                ),
            ],
            required_documents=[
                "Clinical notes supporting medical necessity",
                "ICD-10 diagnosis code(s) on claim",
            ],
        ),
        ProcedureRule(
            payer="Medicare Part B",
            cpt_code="J9035",
            procedure_name=PROCEDURE_NAMES["J9035"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Off-label use for retinal vascular conditions (compendia-supported)",
            ],
            clinical_criteria=[],
            required_documents=[
                "Clinical notes supporting medical necessity",
            ],
        ),
        ProcedureRule(
            payer="Medicare Part B",
            cpt_code="66984",
            procedure_name=PROCEDURE_NAMES["66984"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Visually significant cataract",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Medical necessity documented in clinical notes",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
            ],
            required_documents=[
                "Clinical notes documenting visual impairment",
                "ICD-10 diagnosis code(s) on claim",
            ],
            notes="Medicare does not require PA for cataract surgery but may audit claims. "
                  "Document medical necessity clearly.",
        ),
        ProcedureRule(
            payer="Medicare Part B",
            cpt_code="66982",
            procedure_name=PROCEDURE_NAMES["66982"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Complex cataract with documented complicating factors",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Complicating factors documented to justify complex code",
                    icd10_codes=ICD10_CATARACT,
                    required=True,
                ),
            ],
            required_documents=[
                "Clinical notes documenting complicating factors",
                "ICD-10 diagnosis code(s) on claim",
            ],
        ),
        ProcedureRule(
            payer="Medicare Part B",
            cpt_code="67108",
            procedure_name=PROCEDURE_NAMES["67108"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Retinal detachment requiring surgical repair",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Retinal detachment documented",
                    icd10_codes=ICD10_RETINAL_DETACH,
                    required=True,
                ),
            ],
            required_documents=[
                "Clinical notes and imaging documenting retinal detachment",
            ],
        ),
        ProcedureRule(
            payer="Medicare Part B",
            cpt_code="67228",
            procedure_name=PROCEDURE_NAMES["67228"],
            pa_required=False,
            review_timeline_days=0,
            approved_indications=[
                "Proliferative diabetic retinopathy",
                "Neovascularization from retinal vein occlusion",
            ],
            clinical_criteria=[
                ClinicalCriterion(
                    description="Neovascularization or high-risk features documented",
                    icd10_codes=ICD10_DR + ICD10_RVO,
                    required=True,
                ),
            ],
            required_documents=[
                "Clinical notes and FA documenting neovascularization",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Rules Engine
# ---------------------------------------------------------------------------


class PayerRulesEngine:
    """
    Payer rules engine that encodes PA requirements for ophthalmology procedures.

    Supports lookup by payer + CPT code and evaluates step-therapy compliance
    based on patient treatment history.
    """

    def __init__(self) -> None:
        self._rules: dict[tuple[str, str], ProcedureRule] = {}
        self._load_all_rules()

    def _load_all_rules(self) -> None:
        """Load all payer rules into the lookup table."""
        all_rules = (
            _build_aetna_rules()
            + _build_unitedhealthcare_rules()
            + _build_cigna_rules()
            + _build_bcbs_rules()
            + _build_medicare_rules()
        )
        for rule in all_rules:
            key = (self._normalize_payer(rule.payer), rule.cpt_code.upper())
            self._rules[key] = rule

    @staticmethod
    def _normalize_payer(payer: str) -> str:
        """Normalize payer name for consistent lookup."""
        normalized = payer.strip().lower()
        # Common aliases
        aliases = {
            "uhc": "unitedhealthcare",
            "united": "unitedhealthcare",
            "united healthcare": "unitedhealthcare",
            "united health care": "unitedhealthcare",
            "bcbs": "blue cross blue shield",
            "blue cross": "blue cross blue shield",
            "bluecross": "blue cross blue shield",
            "bluecross blueshield": "blue cross blue shield",
            "medicare": "medicare part b",
            "medicare b": "medicare part b",
            "cms": "medicare part b",
        }
        return aliases.get(normalized, normalized)

    def get_rule(self, payer: str, cpt_code: str) -> ProcedureRule | None:
        """Look up PA rules for a payer + procedure combination."""
        key = (self._normalize_payer(payer), cpt_code.upper())
        return self._rules.get(key)

    def list_payers(self) -> list[str]:
        """Return list of supported payer names."""
        return sorted({rule.payer for rule in self._rules.values()})

    def list_procedures(self) -> list[str]:
        """Return list of supported CPT codes."""
        return sorted({rule.cpt_code for rule in self._rules.values()})

    def evaluate_step_therapy(
        self,
        payer: str,
        cpt_code: str,
        prior_treatments: list[str] | None = None,
    ) -> tuple[bool | None, str]:
        """
        Evaluate whether step-therapy requirements are met.

        Returns:
            (met, explanation) — met is True/False/None (None if no step-therapy rule).
        """
        rule = self.get_rule(payer, cpt_code)
        if rule is None:
            return None, "No rule found for this payer/procedure combination."

        if rule.step_therapy is None:
            return None, "No step-therapy requirement for this payer/procedure."

        st = rule.step_therapy
        if prior_treatments is None:
            prior_treatments = []

        # Normalize treatment names for matching
        normalized_treatments = [t.strip().lower() for t in prior_treatments]
        required_agents_lower = [a.lower() for a in st.required_prior_agents]

        # Check if any required prior agent appears in treatment history
        agent_found = any(
            any(req in treatment for req in required_agents_lower)
            for treatment in normalized_treatments
        )

        if agent_found:
            return True, (
                f"Step-therapy requirement MET: Patient has documented prior use of "
                f"{'/'.join(st.required_prior_agents)}. Minimum {st.min_doses_required} "
                f"doses required — ensure documentation includes number of injections "
                f"and clinical response."
            )

        # Check exception criteria
        exception_note = (
            "\n\nStep-therapy exceptions that may apply:\n"
            + "\n".join(f"  • {exc}" for exc in st.exception_criteria)
        )

        return False, (
            f"Step-therapy requirement NOT MET: No documented prior use of "
            f"{'/'.join(st.required_prior_agents)} found in treatment history. "
            f"Payer requires minimum {st.min_doses_required} doses of preferred agent "
            f"before approving {rule.procedure_name}."
            f"{exception_note}"
        )

    def check_icd10_coverage(
        self,
        payer: str,
        cpt_code: str,
        icd10_codes: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Check if provided ICD-10 codes match approved indications.

        Returns:
            (covered, matching_criteria_descriptions)
        """
        rule = self.get_rule(payer, cpt_code)
        if rule is None:
            return False, []

        if icd10_codes is None:
            return False, ["No ICD-10 codes provided for evaluation."]

        matching = []
        for criterion in rule.clinical_criteria:
            if criterion.icd10_codes:
                overlap = set(icd10_codes) & set(criterion.icd10_codes)
                if overlap:
                    matching.append(criterion.description)

        return len(matching) > 0, matching


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

# Singleton instance
_engine = PayerRulesEngine()


def check_requirements(
    payer: str,
    cpt_code: str,
    icd10_codes: list[str] | None = None,
    prior_treatments: list[str] | None = None,
) -> dict[str, Any]:
    """
    Check PA requirements for a given payer/procedure/patient combination.

    This is the main entry point used by the agent's check_pa_requirements tool.

    Args:
        payer: Insurance payer name.
        cpt_code: CPT or HCPCS procedure code.
        icd10_codes: Patient's ICD-10 diagnosis codes.
        prior_treatments: List of prior treatments the patient has received.

    Returns:
        Dictionary with PA determination including:
        - pa_required, required_documents, step_therapy_met,
        - step_therapy_details, review_timeline_days, approved_indications,
        - clinical_criteria_met, clinical_criteria_unmet, notes
    """
    rule = _engine.get_rule(payer, cpt_code)

    if rule is None:
        return PACheckResult(
            pa_required=True,
            required_documents=[
                "Clinical notes documenting medical necessity",
                "ICD-10 and CPT codes",
                "Contact payer directly — procedure/payer combination not in rules database",
            ],
            step_therapy_met=None,
            step_therapy_details="Unable to determine — payer/procedure not found in rules engine.",
            review_timeline_days=14,
            approved_indications=[],
            clinical_criteria_met=[],
            clinical_criteria_unmet=["Cannot evaluate — rule not found"],
            notes=(
                f"No specific rule found for payer '{payer}' with CPT code '{cpt_code}'. "
                f"Defaulting to PA required. Contact payer for specific requirements."
            ),
        ).__dict__

    # Evaluate step therapy
    step_met, step_details = _engine.evaluate_step_therapy(
        payer, cpt_code, prior_treatments
    )

    # Evaluate ICD-10 coverage
    icd_covered, matching_criteria = _engine.check_icd10_coverage(
        payer, cpt_code, icd10_codes
    )

    # Determine met/unmet clinical criteria
    criteria_met = []
    criteria_unmet = []
    for criterion in rule.clinical_criteria:
        if criterion.icd10_codes and icd10_codes:
            if set(icd10_codes) & set(criterion.icd10_codes):
                criteria_met.append(criterion.description)
            elif criterion.required:
                criteria_unmet.append(criterion.description)
        elif criterion.required:
            # Cannot auto-evaluate non-ICD criteria; flag for manual review
            criteria_unmet.append(f"[Requires manual review] {criterion.description}")

    # Build notes
    notes_parts = []
    if hasattr(rule, "notes") and rule.notes:
        notes_parts.append(rule.notes)
    if rule.step_therapy and step_met is False:
        notes_parts.append(
            "ACTION REQUIRED: Step-therapy documentation needed. "
            "Submit records of prior bevacizumab treatment or document an exception."
        )
    if not icd_covered and icd10_codes:
        notes_parts.append(
            f"WARNING: Provided ICD-10 codes {icd10_codes} may not match approved "
            f"indications. Verify diagnosis coding."
        )

    result = PACheckResult(
        pa_required=rule.pa_required,
        required_documents=rule.required_documents,
        step_therapy_met=step_met,
        step_therapy_details=step_details,
        review_timeline_days=rule.review_timeline_days,
        approved_indications=rule.approved_indications,
        clinical_criteria_met=criteria_met,
        clinical_criteria_unmet=criteria_unmet,
        notes="\n".join(notes_parts) if notes_parts else "Requirements check complete.",
    )

    return result.__dict__
