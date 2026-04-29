"""
Ophthalmology procedure reference data and sample payer responses.

Contains CPT codes, clinical criteria, and mock payer rules for
common ophthalmology procedures.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CPT / HCPCS codes for common ophthalmology procedures
# ---------------------------------------------------------------------------

PROCEDURE_CATALOG: dict[str, dict] = {
    # --- Anti-VEGF Injections ---
    "J2778": {
        "name": "Ranibizumab (Lucentis) injection",
        "category": "anti-vegf",
        "description": "Injection, ranibizumab, 0.1 mg",
        "requires_pa": True,
        "typical_diagnoses": ["H35.32", "H35.31", "E11.311", "E11.3211", "H34.8110"],
        "approved_indications": [
            "Wet age-related macular degeneration (wAMD)",
            "Diabetic macular edema (DME)",
            "Macular edema following retinal vein occlusion (RVO)",
            "Diabetic retinopathy",
            "Myopic choroidal neovascularization (mCNV)",
        ],
        "clinical_criteria": {
            "required_docs": [
                "OCT showing macular pathology",
                "Visual acuity measurement",
                "Fundus examination findings",
            ],
            "step_therapy": None,
            "max_units_per_auth": 6,
            "auth_duration_days": 180,
            "review_turnaround_hours": 48,
        },
    },
    "J0178": {
        "name": "Aflibercept (Eylea) injection",
        "category": "anti-vegf",
        "description": "Injection, aflibercept, 1 mg",
        "requires_pa": True,
        "typical_diagnoses": ["H35.32", "H35.31", "E11.311", "E11.3211"],
        "approved_indications": [
            "Wet age-related macular degeneration (wAMD)",
            "Diabetic macular edema (DME)",
            "Macular edema following retinal vein occlusion (RVO)",
            "Diabetic retinopathy",
        ],
        "clinical_criteria": {
            "required_docs": [
                "OCT showing macular pathology",
                "Visual acuity measurement",
                "Fundus examination findings",
            ],
            "step_therapy": None,
            "max_units_per_auth": 6,
            "auth_duration_days": 180,
            "review_turnaround_hours": 48,
        },
    },
    "J9035": {
        "name": "Bevacizumab (Avastin) injection",
        "category": "anti-vegf",
        "description": "Injection, bevacizumab, 10 mg",
        "requires_pa": False,
        "typical_diagnoses": ["H35.32", "H35.31", "E11.311"],
        "approved_indications": [
            "Wet age-related macular degeneration (wAMD)",
            "Diabetic macular edema (DME)",
            "Macular edema following retinal vein occlusion (RVO)",
        ],
        "clinical_criteria": {
            "required_docs": ["OCT showing macular pathology"],
            "step_therapy": None,
            "max_units_per_auth": 12,
            "auth_duration_days": 365,
            "review_turnaround_hours": 0,
        },
    },
    "J2503": {
        "name": "Faricimab-svoa (Vabysmo) injection",
        "category": "anti-vegf",
        "description": "Injection, faricimab-svoa, 0.1 mg",
        "requires_pa": True,
        "typical_diagnoses": ["H35.32", "E11.311"],
        "approved_indications": [
            "Wet age-related macular degeneration (wAMD)",
            "Diabetic macular edema (DME)",
        ],
        "clinical_criteria": {
            "required_docs": [
                "OCT showing macular pathology",
                "Visual acuity measurement",
                "Fundus examination findings",
                "Documentation of prior anti-VEGF therapy and response",
            ],
            "step_therapy": "Trial of bevacizumab or ranibizumab first preferred",
            "max_units_per_auth": 6,
            "auth_duration_days": 180,
            "review_turnaround_hours": 72,
        },
    },
    # --- Cataract Surgery ---
    "66984": {
        "name": "Cataract surgery — extracapsular with IOL insertion",
        "category": "cataract",
        "description": "Extracapsular cataract removal with insertion of intraocular lens prosthesis, manual or mechanical technique",
        "requires_pa": True,
        "typical_diagnoses": ["H25.11", "H25.12", "H25.13", "H25.811", "H26.001"],
        "approved_indications": [
            "Visually significant cataract with BCVA 20/50 or worse",
            "Cataract causing functional impairment despite best correction",
            "Cataract interfering with diagnosis/treatment of posterior segment disease",
            "Lens-induced conditions (phacomorphic/phacolytic glaucoma)",
        ],
        "clinical_criteria": {
            "required_docs": [
                "Visual acuity (corrected and uncorrected)",
                "Slit-lamp examination documenting cataract grade",
                "Documentation of functional impairment",
                "Glare testing if VA better than 20/40",
            ],
            "step_therapy": None,
            "max_units_per_auth": 1,
            "auth_duration_days": 90,
            "review_turnaround_hours": 72,
        },
    },
    "66982": {
        "name": "Cataract surgery — complex extracapsular with IOL",
        "category": "cataract",
        "description": "Complex extracapsular cataract removal with IOL (e.g., pediatric, subluxated lens, prior vitrectomy)",
        "requires_pa": True,
        "typical_diagnoses": ["H25.11", "H26.001", "H27.111", "Q12.0"],
        "approved_indications": [
            "Complex cataract requiring specialized technique",
            "Pediatric cataract",
            "Subluxated or dislocated lens",
            "Cataract with prior vitrectomy",
        ],
        "clinical_criteria": {
            "required_docs": [
                "Visual acuity (corrected and uncorrected)",
                "Slit-lamp examination documenting cataract and complicating factors",
                "Documentation of why complex technique is required",
            ],
            "step_therapy": None,
            "max_units_per_auth": 1,
            "auth_duration_days": 90,
            "review_turnaround_hours": 72,
        },
    },
    # --- Retinal Detachment Repair ---
    "67108": {
        "name": "Retinal detachment repair — vitrectomy with membrane peeling",
        "category": "retinal_surgery",
        "description": "Repair of retinal detachment; with vitrectomy, any method, including removal of vitreous body, with or without air-fluid exchange",
        "requires_pa": True,
        "typical_diagnoses": ["H33.001", "H33.011", "H33.021", "H33.031"],
        "approved_indications": [
            "Rhegmatogenous retinal detachment",
            "Tractional retinal detachment threatening macula",
            "Combined rhegmatogenous-tractional detachment",
        ],
        "clinical_criteria": {
            "required_docs": [
                "Dilated fundus examination documenting detachment",
                "B-scan ultrasonography if view is obscured",
                "Visual acuity measurement",
                "Duration of symptoms and macular status",
            ],
            "step_therapy": None,
            "max_units_per_auth": 1,
            "auth_duration_days": 30,
            "review_turnaround_hours": 24,
        },
    },
    "67113": {
        "name": "Retinal detachment repair — complex vitrectomy",
        "category": "retinal_surgery",
        "description": "Repair of complex retinal detachment with vitrectomy and membrane peeling (proliferative vitreoretinopathy)",
        "requires_pa": True,
        "typical_diagnoses": ["H33.4", "H33.001", "H33.011"],
        "approved_indications": [
            "Complex retinal detachment with proliferative vitreoretinopathy",
            "Recurrent retinal detachment after prior repair",
            "Giant retinal tear",
        ],
        "clinical_criteria": {
            "required_docs": [
                "Dilated fundus examination documenting detachment and PVR",
                "B-scan ultrasonography",
                "Visual acuity measurement",
                "Prior surgical history",
            ],
            "step_therapy": None,
            "max_units_per_auth": 1,
            "auth_duration_days": 30,
            "review_turnaround_hours": 24,
        },
    },
    "67101": {
        "name": "Retinal detachment repair — scleral buckle",
        "category": "retinal_surgery",
        "description": "Repair of retinal detachment, one or more sessions; cryotherapy or diathermy, with scleral buckling",
        "requires_pa": True,
        "typical_diagnoses": ["H33.001", "H33.011", "H33.021"],
        "approved_indications": [
            "Rhegmatogenous retinal detachment amenable to scleral buckle",
            "Inferior retinal detachment",
        ],
        "clinical_criteria": {
            "required_docs": [
                "Dilated fundus examination documenting detachment and break location",
                "Visual acuity measurement",
                "Duration of symptoms",
            ],
            "step_therapy": None,
            "max_units_per_auth": 1,
            "auth_duration_days": 30,
            "review_turnaround_hours": 24,
        },
    },
    # --- Other Common Ophthalmology Procedures ---
    "65855": {
        "name": "Laser trabeculoplasty (SLT/ALT)",
        "category": "glaucoma",
        "description": "Trabeculoplasty by laser surgery, one or more sessions",
        "requires_pa": False,
        "typical_diagnoses": ["H40.11X1", "H40.11X2", "H40.11X3"],
        "approved_indications": [
            "Primary open-angle glaucoma",
            "Ocular hypertension not controlled by medication",
        ],
        "clinical_criteria": {
            "required_docs": [
                "IOP measurements",
                "Visual field testing",
                "Gonioscopy confirming open angles",
            ],
            "step_therapy": None,
            "max_units_per_auth": 1,
            "auth_duration_days": 180,
            "review_turnaround_hours": 0,
        },
    },
    "67228": {
        "name": "Panretinal photocoagulation (PRP)",
        "category": "retinal_laser",
        "description": "Treatment of extensive or progressive retinopathy, photocoagulation",
        "requires_pa": False,
        "typical_diagnoses": ["E11.3511", "E11.3512", "E11.3521"],
        "approved_indications": [
            "Proliferative diabetic retinopathy",
            "Severe non-proliferative diabetic retinopathy at high risk",
        ],
        "clinical_criteria": {
            "required_docs": [
                "Fundus examination documenting retinopathy severity",
                "Fluorescein angiography if applicable",
            ],
            "step_therapy": None,
            "max_units_per_auth": 4,
            "auth_duration_days": 365,
            "review_turnaround_hours": 0,
        },
    },
    "67040": {
        "name": "Vitrectomy — epiretinal membrane peel",
        "category": "retinal_surgery",
        "description": "Vitrectomy, mechanical, pars plana approach; with removal of internal limiting membrane of retina (epiretinal membrane peel)",
        "requires_pa": True,
        "typical_diagnoses": ["H35.371", "H35.372"],
        "approved_indications": [
            "Symptomatic epiretinal membrane with visual impairment",
            "Epiretinal membrane with metamorphopsia and VA 20/40 or worse",
        ],
        "clinical_criteria": {
            "required_docs": [
                "OCT showing epiretinal membrane with macular distortion",
                "Visual acuity measurement",
                "Documentation of functional impairment",
            ],
            "step_therapy": None,
            "max_units_per_auth": 1,
            "auth_duration_days": 60,
            "review_turnaround_hours": 72,
        },
    },
}

# ---------------------------------------------------------------------------
# ICD-10 diagnosis descriptions (for readable responses)
# ---------------------------------------------------------------------------

DIAGNOSIS_DESCRIPTIONS: dict[str, str] = {
    "H35.32": "Exudative age-related macular degeneration (wet AMD)",
    "H35.31": "Nonexudative age-related macular degeneration (dry AMD)",
    "E11.311": "Type 2 diabetes with unspecified diabetic retinopathy with macular edema",
    "E11.3211": "Type 2 diabetes with mild nonproliferative diabetic retinopathy with macular edema, right eye",
    "E11.3511": "Type 2 diabetes with proliferative diabetic retinopathy with macular edema, right eye",
    "E11.3512": "Type 2 diabetes with proliferative diabetic retinopathy with macular edema, left eye",
    "E11.3521": "Type 2 diabetes with proliferative diabetic retinopathy with traction retinal detachment involving the macula, right eye",
    "H34.8110": "Central retinal vein occlusion, right eye, with macular edema",
    "H25.11": "Age-related nuclear cataract, right eye",
    "H25.12": "Age-related nuclear cataract, left eye",
    "H25.13": "Age-related nuclear cataract, bilateral",
    "H25.811": "Combined forms of age-related cataract, right eye",
    "H26.001": "Unspecified infantile and juvenile cataract, right eye",
    "H27.111": "Subluxation of lens, right eye",
    "Q12.0": "Congenital cataract",
    "H33.001": "Unspecified retinal detachment with retinal break, right eye",
    "H33.011": "Retinal detachment with single break, right eye",
    "H33.021": "Retinal detachment with multiple breaks, right eye",
    "H33.031": "Retinal detachment with giant retinal tear, right eye",
    "H33.4": "Traction detachment of retina",
    "H35.371": "Puckering of macula, right eye",
    "H35.372": "Puckering of macula, left eye",
    "H40.11X1": "Primary open-angle glaucoma, right eye, mild stage",
    "H40.11X2": "Primary open-angle glaucoma, right eye, moderate stage",
    "H40.11X3": "Primary open-angle glaucoma, right eye, severe stage",
}

# ---------------------------------------------------------------------------
# Mock payer configurations
# ---------------------------------------------------------------------------

PAYER_PROFILES: dict[str, dict] = {
    "AETNA": {
        "name": "Aetna",
        "payer_id": "60054",
        "portal_url": "https://portal.aetna.com/pa",
        "average_turnaround_hours": 48,
        "supports_electronic_pa": True,
        "anti_vegf_step_therapy": True,
        "preferred_anti_vegf": "J9035",  # Bevacizumab preferred first
    },
    "UNITEDHEALTHCARE": {
        "name": "UnitedHealthcare",
        "payer_id": "87726",
        "portal_url": "https://provider.uhc.com/prior-auth",
        "average_turnaround_hours": 72,
        "supports_electronic_pa": True,
        "anti_vegf_step_therapy": False,
        "preferred_anti_vegf": None,
    },
    "CIGNA": {
        "name": "Cigna",
        "payer_id": "62308",
        "portal_url": "https://provider.cigna.com/pa",
        "average_turnaround_hours": 48,
        "supports_electronic_pa": True,
        "anti_vegf_step_therapy": True,
        "preferred_anti_vegf": "J9035",
    },
    "BCBS": {
        "name": "Blue Cross Blue Shield",
        "payer_id": "00050",
        "portal_url": "https://provider.bcbs.com/pa",
        "average_turnaround_hours": 72,
        "supports_electronic_pa": True,
        "anti_vegf_step_therapy": False,
        "preferred_anti_vegf": None,
    },
    "MEDICARE": {
        "name": "Medicare (CMS)",
        "payer_id": "00882",
        "portal_url": "https://provider.cms.gov/pa",
        "average_turnaround_hours": 24,
        "supports_electronic_pa": True,
        "anti_vegf_step_therapy": False,
        "preferred_anti_vegf": None,
    },
}
