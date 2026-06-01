#!/usr/bin/env python3
"""
Prior Authorization Agent — Autonomous form-filling agent with Claude Agent SDK.

An agentic system that navigates the Prior Authorization Portal's multi-step form,
reading fields, filling inputs, selecting dropdowns, clicking buttons, and handling
conditional logic based on medication category and urgency level.

The agent uses a simulated browser abstraction (FormNavigator) that tracks form state,
validates transitions, and enforces the same conditional visibility rules as the
real portal. A Claude model drives the decision-making — choosing which fields to
fill, in what order, and handling branching logic for different medication categories.

Usage:
    python -m agents.prior_auth_agent --case specialty
    python -m agents.prior_auth_agent --case controlled --submit
    python -m agents.prior_auth_agent --case-file ./cases/patient_case.json
    python -m agents.prior_auth_agent --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.environ.get("PA_AGENT_MODEL", "claude-opus-4-7-20250501")
MAX_TOKENS = 4096
MAX_TURNS = 40

PORTAL_URL = os.environ.get("PA_PORTAL_URL", "http://localhost:8080")


# ---------------------------------------------------------------------------
# Form Schema — mirrors the HTML portal's field definitions
# ---------------------------------------------------------------------------

class FieldType(str, Enum):
    TEXT = "text"
    DATE = "date"
    TEL = "tel"
    EMAIL = "email"
    NUMBER = "number"
    SELECT = "select"
    TEXTAREA = "textarea"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    FILE = "file"


@dataclass
class FormField:
    """Definition of a single form field in the portal."""
    id: str
    label: str
    field_type: FieldType
    required: bool = False
    options: list[dict[str, str]] | None = None  # For select/radio: [{value, label}]
    placeholder: str = ""
    step: int = 1
    conditional_on: str | None = None  # field_id=value that makes this visible
    maxlength: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "type": self.field_type.value,
            "required": self.required,
            "step": self.step,
        }
        if self.options:
            d["options"] = self.options
        if self.placeholder:
            d["placeholder"] = self.placeholder
        if self.conditional_on:
            d["visible_when"] = self.conditional_on
        return d


# Complete field definitions matching the HTML portal
FORM_FIELDS: list[FormField] = [
    # --- Step 1: Patient Information ---
    FormField("patientFirstName", "First Name", FieldType.TEXT, required=True, placeholder="John", step=1),
    FormField("patientLastName", "Last Name", FieldType.TEXT, required=True, placeholder="Smith", step=1),
    FormField("patientDOB", "Date of Birth", FieldType.DATE, required=True, step=1),
    FormField("memberId", "Member ID", FieldType.TEXT, required=True, placeholder="MBR-000000000", step=1),
    FormField("insurancePlan", "Insurance Plan", FieldType.SELECT, required=True, step=1, options=[
        {"value": "gold_ppo", "label": "Gold PPO"},
        {"value": "silver_hmo", "label": "Silver HMO"},
        {"value": "bronze_hdhp", "label": "Bronze HDHP"},
        {"value": "platinum_ppo", "label": "Platinum PPO"},
        {"value": "medicaid", "label": "Medicaid Managed Care"},
        {"value": "medicare_advantage", "label": "Medicare Advantage"},
    ]),
    FormField("groupNumber", "Group Number", FieldType.TEXT, required=False, placeholder="GRP-00000", step=1),
    FormField("providerName", "Provider Name", FieldType.TEXT, required=True, placeholder="Dr. Jane Doe", step=1),
    FormField("npiNumber", "NPI Number", FieldType.TEXT, required=True, placeholder="1234567890", maxlength=10, step=1),
    FormField("providerPhone", "Phone", FieldType.TEL, required=True, placeholder="(555) 000-0000", step=1),
    FormField("providerFax", "Fax", FieldType.TEL, required=False, placeholder="(555) 000-0000", step=1),

    # --- Step 2: Medication Details ---
    FormField("medCategory", "Medication Category", FieldType.SELECT, required=True, step=2, options=[
        {"value": "specialty", "label": "Specialty / Biologic"},
        {"value": "controlled", "label": "Controlled Substance"},
        {"value": "brand_nonpreferred", "label": "Brand Name (Non-Preferred)"},
        {"value": "compound", "label": "Compounded Medication"},
        {"value": "gene_therapy", "label": "Gene / Cell Therapy"},
    ]),
    FormField("medName", "Medication Name", FieldType.TEXT, required=True, placeholder="e.g., Humira (adalimumab)", step=2),
    FormField("ndcCode", "NDC Code", FieldType.TEXT, required=True, placeholder="00000-0000-00", step=2),
    FormField("dosage", "Dosage", FieldType.TEXT, required=True, placeholder="40mg", step=2),
    FormField("frequency", "Frequency", FieldType.SELECT, required=True, step=2, options=[
        {"value": "daily", "label": "Once Daily"},
        {"value": "bid", "label": "Twice Daily (BID)"},
        {"value": "tid", "label": "Three Times Daily (TID)"},
        {"value": "weekly", "label": "Weekly"},
        {"value": "biweekly", "label": "Every 2 Weeks"},
        {"value": "monthly", "label": "Monthly"},
        {"value": "prn", "label": "As Needed (PRN)"},
        {"value": "other", "label": "Other"},
    ]),
    FormField("route", "Route of Administration", FieldType.SELECT, required=True, step=2, options=[
        {"value": "oral", "label": "Oral"},
        {"value": "subcutaneous", "label": "Subcutaneous Injection"},
        {"value": "intramuscular", "label": "Intramuscular Injection"},
        {"value": "intravenous", "label": "Intravenous (IV) Infusion"},
        {"value": "topical", "label": "Topical"},
        {"value": "inhalation", "label": "Inhalation"},
        {"value": "other", "label": "Other"},
    ]),
    FormField("duration", "Duration of Therapy", FieldType.SELECT, required=True, step=2, options=[
        {"value": "30", "label": "30 Days"},
        {"value": "60", "label": "60 Days"},
        {"value": "90", "label": "90 Days"},
        {"value": "180", "label": "6 Months"},
        {"value": "365", "label": "1 Year"},
        {"value": "ongoing", "label": "Ongoing / Indefinite"},
    ]),

    # Conditional: Specialty / Biologic
    FormField("stepTherapy", "Step Therapy: Prior Medications Tried", FieldType.TEXTAREA,
              required=True, step=2, conditional_on="medCategory=specialty"),
    FormField("specialtyPharmacy", "Specialty Pharmacy", FieldType.SELECT, step=2,
              conditional_on="medCategory=specialty", options=[
                  {"value": "accredo", "label": "Accredo"},
                  {"value": "optum", "label": "Optum Specialty"},
                  {"value": "cvs_specialty", "label": "CVS Specialty"},
                  {"value": "biologics", "label": "Biologics Inc."},
                  {"value": "other", "label": "Other"},
              ]),
    FormField("buyBill", "Buy & Bill?", FieldType.SELECT, step=2,
              conditional_on="medCategory=specialty", options=[
                  {"value": "yes", "label": "Yes - Provider will administer"},
                  {"value": "no", "label": "No - Patient self-administers"},
              ]),

    # Conditional: Controlled Substance
    FormField("deaSchedule", "DEA Schedule", FieldType.SELECT, step=2,
              conditional_on="medCategory=controlled", options=[
                  {"value": "II", "label": "Schedule II"},
                  {"value": "III", "label": "Schedule III"},
                  {"value": "IV", "label": "Schedule IV"},
                  {"value": "V", "label": "Schedule V"},
              ]),
    FormField("deaNumber", "DEA Number", FieldType.TEXT, step=2,
              conditional_on="medCategory=controlled", placeholder="AB1234567"),
    FormField("pdmpCheck", "PDMP Check Completed?", FieldType.RADIO, step=2,
              conditional_on="medCategory=controlled", options=[
                  {"value": "yes", "label": "Yes - PDMP reviewed, no concerns identified"},
                  {"value": "yes_concerns", "label": "Yes - PDMP reviewed, concerns addressed in clinical notes"},
                  {"value": "no", "label": "No - PDMP not yet checked"},
              ]),

    # Conditional: Brand Non-Preferred
    FormField("genericTried", "Generic Alternative Tried", FieldType.TEXT, step=2,
              conditional_on="medCategory=brand_nonpreferred"),
    FormField("genericReason", "Reason Generic is Not Appropriate", FieldType.SELECT, step=2,
              conditional_on="medCategory=brand_nonpreferred", options=[
                  {"value": "adverse_reaction", "label": "Adverse reaction to generic"},
                  {"value": "therapeutic_failure", "label": "Therapeutic failure with generic"},
                  {"value": "no_generic", "label": "No generic available"},
                  {"value": "narrow_therapeutic_index", "label": "Narrow therapeutic index drug"},
                  {"value": "other", "label": "Other (explain below)"},
              ]),
    FormField("brandDetails", "Additional Details", FieldType.TEXTAREA, step=2,
              conditional_on="medCategory=brand_nonpreferred"),

    # Conditional: Compounded Medication
    FormField("compoundPharmacy", "Compounding Pharmacy Name", FieldType.TEXT, step=2,
              conditional_on="medCategory=compound"),
    FormField("compoundIngredients", "List of Ingredients", FieldType.TEXTAREA, step=2,
              conditional_on="medCategory=compound"),
    FormField("compoundReason", "Reason Commercially Available Product Cannot Be Used", FieldType.TEXTAREA, step=2,
              conditional_on="medCategory=compound"),

    # Conditional: Gene / Cell Therapy
    FormField("remsEnrolled", "REMS Program Enrolled?", FieldType.SELECT, step=2,
              conditional_on="medCategory=gene_therapy", options=[
                  {"value": "yes", "label": "Yes"},
                  {"value": "no", "label": "No"},
                  {"value": "na", "label": "Not Applicable"},
              ]),
    FormField("treatmentCenter", "Certified Treatment Center", FieldType.TEXT, step=2,
              conditional_on="medCategory=gene_therapy"),
    FormField("fdaIndication", "FDA Approval / Indication", FieldType.TEXTAREA, step=2,
              conditional_on="medCategory=gene_therapy"),
    FormField("therapyCost", "Estimated Total Cost", FieldType.TEXT, step=2,
              conditional_on="medCategory=gene_therapy", placeholder="$0.00"),

    # --- Step 3: Clinical Details ---
    FormField("icd10Primary", "Primary ICD-10 Code", FieldType.TEXT, required=True, placeholder="e.g., M06.9", step=3),
    FormField("diagDescription", "Primary Diagnosis Description", FieldType.TEXT, required=True,
              placeholder="e.g., Rheumatoid arthritis, unspecified", step=3),
    FormField("icd10Secondary", "Secondary ICD-10 Code", FieldType.TEXT, required=False, step=3),
    FormField("diagSecondary", "Secondary Diagnosis", FieldType.TEXT, required=False, step=3),
    FormField("clinicalRationale", "Clinical Rationale", FieldType.TEXTAREA, required=True, step=3),
    FormField("urgency", "Urgency Level", FieldType.RADIO, required=True, step=3, options=[
        {"value": "routine", "label": "Routine - Standard review timeline (5-7 business days)"},
        {"value": "urgent", "label": "Urgent - Expedited review (24-72 hours)"},
        {"value": "emergency", "label": "Emergency - Immediate review. Patient safety at risk."},
    ]),
    FormField("urgentJustification", "Urgency Justification", FieldType.TEXTAREA, step=3,
              conditional_on="urgency=urgent,urgency=emergency"),
    FormField("recentVisit", "Has the patient been seen within the last 30 days?", FieldType.RADIO,
              required=True, step=3, options=[
                  {"value": "yes", "label": "Yes"},
                  {"value": "no", "label": "No"},
              ]),

    # --- Step 4: Supporting Documents ---
    FormField("additionalNotes", "Additional Notes", FieldType.TEXTAREA, required=False, step=4),

    # --- Step 5: Review & Submit ---
    FormField("attestation", "Attestation", FieldType.CHECKBOX, required=True, step=5),
]

# Index fields by ID for quick lookup
_FIELDS_BY_ID: dict[str, FormField] = {f.id: f for f in FORM_FIELDS}


# ---------------------------------------------------------------------------
# Form Navigator — simulated browser state machine
# ---------------------------------------------------------------------------

@dataclass
class FormState:
    """Tracks the full state of the prior authorization form."""
    current_step: int = 1
    values: dict[str, str] = field(default_factory=dict)
    uploaded_files: dict[str, list[str]] = field(default_factory=lambda: {
        "clinical": [], "lab": [], "letter": []
    })
    submitted: bool = False
    reference_number: str | None = None
    errors: list[str] = field(default_factory=list)


class FormNavigator:
    """
    Simulated browser for the Prior Authorization Portal.

    Maintains form state and enforces the same visibility/validation rules
    as the real HTML portal. Each method returns a JSON-serializable dict
    describing the result of the action.
    """

    def __init__(self):
        self.state = FormState()
        self._field_index = _FIELDS_BY_ID

    # --- Visibility logic ---

    def _is_field_visible(self, f: FormField) -> bool:
        """Check if a conditional field should be visible given current values."""
        if not f.conditional_on:
            return True
        # Support comma-separated conditions (OR logic)
        conditions = [c.strip() for c in f.conditional_on.split(",")]
        for cond in conditions:
            if "=" in cond:
                dep_field, dep_value = cond.split("=", 1)
                if self.state.values.get(dep_field) == dep_value:
                    return True
        return False

    def _get_visible_fields(self, step: int | None = None) -> list[FormField]:
        """Return all currently visible fields, optionally filtered by step."""
        fields = FORM_FIELDS if step is None else [f for f in FORM_FIELDS if f.step == step]
        return [f for f in fields if self._is_field_visible(f)]

    # --- Tool implementations ---

    def read_page(self) -> dict[str, Any]:
        """Read the current step's visible fields and their current values."""
        step = self.state.current_step
        visible = self._get_visible_fields(step)

        step_names = {1: "Patient Information", 2: "Medication Details",
                      3: "Clinical Details", 4: "Supporting Documents",
                      5: "Review & Submit"}

        fields_info = []
        for f in visible:
            info = f.to_dict()
            info["current_value"] = self.state.values.get(f.id, "")
            fields_info.append(info)

        result: dict[str, Any] = {
            "current_step": step,
            "step_name": step_names.get(step, f"Step {step}"),
            "total_steps": 5,
            "fields": fields_info,
        }

        if step == 4:
            result["uploaded_files"] = self.state.uploaded_files
            category = self.state.values.get("medCategory", "")
            required_docs = {
                "specialty": ["Recent clinical notes (within 90 days)",
                              "Prior medication history / step therapy records",
                              "Relevant lab results"],
                "controlled": ["PDMP report printout",
                               "Treatment agreement / opioid contract",
                               "Urine drug screen results"],
                "brand_nonpreferred": ["Documentation of generic trial and failure",
                                       "Adverse reaction report (if applicable)"],
                "compound": ["Compounding formula / recipe",
                             "Documentation that commercial product is unsuitable"],
                "gene_therapy": ["REMS enrollment confirmation",
                                 "Treatment center certification",
                                 "Multidisciplinary team review notes",
                                 "Genetic testing results"],
            }
            if category in required_docs:
                result["required_documents"] = required_docs[category]

        if step == 5:
            result["summary"] = self._build_summary()

        return result

    def fill_field(self, field_id: str, value: str) -> dict[str, Any]:
        """Fill a form field with a value."""
        if field_id not in self._field_index:
            return {"success": False, "error": f"Unknown field: {field_id}"}

        f = self._field_index[field_id]

        if not self._is_field_visible(f):
            return {"success": False, "error": f"Field '{field_id}' is not currently visible. "
                    f"It requires: {f.conditional_on}"}

        if f.step != self.state.current_step:
            return {"success": False, "error": f"Field '{field_id}' is on step {f.step}, "
                    f"but current step is {self.state.current_step}. Navigate first."}

        # Validate select options
        if f.field_type == FieldType.SELECT and f.options:
            valid_values = [o["value"] for o in f.options]
            if value not in valid_values:
                return {"success": False, "error": f"Invalid option '{value}' for {field_id}. "
                        f"Valid options: {valid_values}"}

        # Validate radio options
        if f.field_type == FieldType.RADIO and f.options:
            valid_values = [o["value"] for o in f.options]
            if value not in valid_values:
                return {"success": False, "error": f"Invalid option '{value}' for {field_id}. "
                        f"Valid options: {valid_values}"}

        # Validate checkbox
        if f.field_type == FieldType.CHECKBOX:
            value = "true" if value.lower() in ("true", "yes", "1", "checked") else "false"

        # Enforce maxlength
        if f.maxlength and len(value) > f.maxlength:
            return {"success": False, "error": f"Value exceeds max length of {f.maxlength} for {field_id}"}

        self.state.values[field_id] = value

        # Check if this triggers conditional visibility changes
        newly_visible = []
        for other in FORM_FIELDS:
            if other.conditional_on and field_id in other.conditional_on:
                if self._is_field_visible(other):
                    newly_visible.append(other.id)

        result: dict[str, Any] = {
            "success": True,
            "field_id": field_id,
            "value_set": value,
        }
        if newly_visible:
            result["newly_visible_fields"] = newly_visible
            result["note"] = (f"Setting '{field_id}' revealed additional fields: "
                              f"{', '.join(newly_visible)}. You should fill these too.")

        return result

    def click_button(self, button: str) -> dict[str, Any]:
        """Click a navigation button: 'next', 'back', or 'submit'."""
        button = button.lower().strip()

        if button == "next":
            return self._go_next()
        elif button == "back":
            return self._go_back()
        elif button == "submit":
            return self._submit()
        else:
            return {"success": False, "error": f"Unknown button: {button}. Use 'next', 'back', or 'submit'."}

    def upload_file(self, category: str, filename: str) -> dict[str, Any]:
        """Simulate uploading a file."""
        if category not in self.state.uploaded_files:
            return {"success": False, "error": f"Invalid upload category: {category}. "
                    f"Use: clinical, lab, or letter"}
        if self.state.current_step != 4:
            return {"success": False, "error": "File uploads are only available on step 4."}

        self.state.uploaded_files[category].append(filename)
        return {
            "success": True,
            "uploaded": filename,
            "category": category,
            "total_files": {k: len(v) for k, v in self.state.uploaded_files.items()},
        }

    def get_form_state(self) -> dict[str, Any]:
        """Get a snapshot of the entire form state."""
        return {
            "current_step": self.state.current_step,
            "submitted": self.state.submitted,
            "reference_number": self.state.reference_number,
            "filled_fields": len(self.state.values),
            "total_fields_on_step": len(self._get_visible_fields(self.state.current_step)),
            "values": dict(self.state.values),
            "uploaded_files": {k: list(v) for k, v in self.state.uploaded_files.items()},
        }

    # --- Internal navigation ---

    def _validate_step(self, step: int) -> list[str]:
        """Validate all required fields on a step. Returns list of error messages."""
        errors = []
        visible = self._get_visible_fields(step)
        for f in visible:
            if f.required:
                val = self.state.values.get(f.id, "").strip()
                if not val or val == "false":
                    errors.append(f"Required field '{f.label}' ({f.id}) is empty.")
        return errors

    def _go_next(self) -> dict[str, Any]:
        """Advance to next step with validation."""
        step = self.state.current_step
        if step >= 5:
            return {"success": False, "error": "Already on the last step. Use 'submit' to submit."}

        errors = self._validate_step(step)
        if errors:
            return {
                "success": False,
                "error": "Validation failed. Fill required fields before proceeding.",
                "missing_fields": errors,
            }

        self.state.current_step = step + 1
        next_page = self.read_page()
        return {
            "success": True,
            "navigated_to": self.state.current_step,
            "page": next_page,
        }

    def _go_back(self) -> dict[str, Any]:
        """Go back to previous step."""
        if self.state.current_step <= 1:
            return {"success": False, "error": "Already on the first step."}
        self.state.current_step -= 1
        return {
            "success": True,
            "navigated_to": self.state.current_step,
            "page": self.read_page(),
        }

    def _submit(self) -> dict[str, Any]:
        """Submit the form (from step 5)."""
        if self.state.current_step != 5:
            return {"success": False, "error": f"Must be on step 5 to submit. Currently on step {self.state.current_step}."}

        # Check attestation
        if self.state.values.get("attestation") != "true":
            return {"success": False, "error": "Attestation checkbox must be checked before submitting."}

        # Validate all steps
        all_errors = []
        for step in range(1, 6):
            all_errors.extend(self._validate_step(step))
        if all_errors:
            return {
                "success": False,
                "error": "Form has validation errors across steps.",
                "missing_fields": all_errors,
            }

        # Generate reference number
        import random
        ref = f"PA-2026-{random.randint(100000, 999999)}"
        self.state.submitted = True
        self.state.reference_number = ref

        return {
            "success": True,
            "submitted": True,
            "reference_number": ref,
            "message": "Prior authorization request submitted successfully.",
            "form_data": self._collect_form_data(),
        }

    def _build_summary(self) -> dict[str, Any]:
        """Build the review summary shown on step 5."""
        v = self.state.values.get

        def sel_label(field_id: str) -> str:
            val = v(field_id, "")
            f = self._field_index.get(field_id)
            if f and f.options:
                for opt in f.options:
                    if opt["value"] == val:
                        return opt["label"]
            return val

        return {
            "patient": {
                "name": f"{v('patientFirstName', '')} {v('patientLastName', '')}".strip(),
                "dob": v("patientDOB", ""),
                "member_id": v("memberId", ""),
                "insurance_plan": sel_label("insurancePlan"),
            },
            "provider": {
                "name": v("providerName", ""),
                "npi": v("npiNumber", ""),
                "phone": v("providerPhone", ""),
            },
            "medication": {
                "category": sel_label("medCategory"),
                "name": v("medName", ""),
                "ndc_code": v("ndcCode", ""),
                "dosage": v("dosage", ""),
                "frequency": sel_label("frequency"),
                "route": sel_label("route"),
                "duration": sel_label("duration"),
            },
            "clinical": {
                "primary_dx": f"{v('icd10Primary', '')} - {v('diagDescription', '')}",
                "urgency": sel_label("urgency"),
                "recent_visit": v("recentVisit", ""),
            },
            "documents": {
                "clinical_files": len(self.state.uploaded_files["clinical"]),
                "lab_files": len(self.state.uploaded_files["lab"]),
                "letter_files": len(self.state.uploaded_files["letter"]),
                "additional_notes": v("additionalNotes", ""),
            },
        }

    def _collect_form_data(self) -> dict[str, Any]:
        """Collect all form data in the same structure as the portal's collectFormData()."""
        v = self.state.values.get
        return {
            "patient": {
                "firstName": v("patientFirstName", ""),
                "lastName": v("patientLastName", ""),
                "dob": v("patientDOB", ""),
                "memberId": v("memberId", ""),
                "insurancePlan": v("insurancePlan", ""),
                "groupNumber": v("groupNumber", ""),
            },
            "provider": {
                "name": v("providerName", ""),
                "npi": v("npiNumber", ""),
                "phone": v("providerPhone", ""),
                "fax": v("providerFax", ""),
            },
            "medication": {
                "category": v("medCategory", ""),
                "name": v("medName", ""),
                "ndcCode": v("ndcCode", ""),
                "dosage": v("dosage", ""),
                "frequency": v("frequency", ""),
                "route": v("route", ""),
                "duration": v("duration", ""),
            },
            "clinical": {
                "icd10Primary": v("icd10Primary", ""),
                "diagDescription": v("diagDescription", ""),
                "icd10Secondary": v("icd10Secondary", ""),
                "diagSecondary": v("diagSecondary", ""),
                "clinicalRationale": v("clinicalRationale", ""),
                "urgency": v("urgency", ""),
                "recentVisit": v("recentVisit", ""),
            },
            "documents": {
                "clinical": list(self.state.uploaded_files["clinical"]),
                "lab": list(self.state.uploaded_files["lab"]),
                "letter": list(self.state.uploaded_files["letter"]),
                "additionalNotes": v("additionalNotes", ""),
            },
            "submittedAt": datetime.now().isoformat(),
            "referenceNumber": self.state.reference_number,
        }


# ---------------------------------------------------------------------------
# Tool Schemas (Anthropic tool-use format)
# ---------------------------------------------------------------------------

READ_PAGE_TOOL: dict[str, Any] = {
    "name": "read_page",
    "description": (
        "Read the current page of the prior authorization form. Returns all visible "
        "fields on the current step with their types, options, requirements, and "
        "current values. Use this to understand what needs to be filled before "
        "navigating forward."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

FILL_FIELD_TOOL: dict[str, Any] = {
    "name": "fill_field",
    "description": (
        "Fill a form field with a value. For text/date/tel fields, provide the text value. "
        "For select dropdowns, provide the option 'value' (not the label). "
        "For radio buttons, provide the option 'value'. "
        "For checkboxes, provide 'true' or 'false'. "
        "The field must be on the current step and visible (conditional fields must have "
        "their parent condition met)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field_id": {
                "type": "string",
                "description": "The ID of the form field to fill.",
            },
            "value": {
                "type": "string",
                "description": "The value to set. For selects/radios, use the option value, not the label.",
            },
        },
        "required": ["field_id", "value"],
    },
}

CLICK_BUTTON_TOOL: dict[str, Any] = {
    "name": "click_button",
    "description": (
        "Click a navigation button on the form. Available buttons:\n"
        "- 'next': Advance to the next step (validates required fields first)\n"
        "- 'back': Return to the previous step\n"
        "- 'submit': Submit the completed form (only from step 5, requires attestation)"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "button": {
                "type": "string",
                "enum": ["next", "back", "submit"],
                "description": "Which button to click.",
            },
        },
        "required": ["button"],
    },
}

UPLOAD_FILE_TOOL: dict[str, Any] = {
    "name": "upload_file",
    "description": (
        "Upload a supporting document on step 4. Categories: 'clinical' (clinical notes), "
        "'lab' (lab results), 'letter' (letter of medical necessity)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["clinical", "lab", "letter"],
                "description": "Document category.",
            },
            "filename": {
                "type": "string",
                "description": "Name of the file being uploaded.",
            },
        },
        "required": ["category", "filename"],
    },
}

GET_FORM_STATE_TOOL: dict[str, Any] = {
    "name": "get_form_state",
    "description": (
        "Get a full snapshot of the form state: current step, all filled values, "
        "uploaded files, and submission status. Use this to verify progress or "
        "debug issues."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

ALL_TOOLS = [READ_PAGE_TOOL, FILL_FIELD_TOOL, CLICK_BUTTON_TOOL, UPLOAD_FILE_TOOL, GET_FORM_STATE_TOOL]


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(navigator: FormNavigator, name: str, input_dict: dict[str, Any]) -> str:
    """Execute a tool call and return JSON result."""
    if name == "read_page":
        result = navigator.read_page()
    elif name == "fill_field":
        result = navigator.fill_field(**input_dict)
    elif name == "click_button":
        result = navigator.click_button(**input_dict)
    elif name == "upload_file":
        result = navigator.upload_file(**input_dict)
    elif name == "get_form_state":
        result = navigator.get_form_state()
    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Sample Patient Cases
# ---------------------------------------------------------------------------

SAMPLE_CASES: dict[str, dict[str, Any]] = {
    "specialty": {
        "description": "Specialty biologic medication for rheumatoid arthritis",
        "patient": {
            "firstName": "Maria", "lastName": "Gonzalez",
            "dob": "1978-03-15", "memberId": "MBR-992847561",
            "insurancePlan": "gold_ppo", "groupNumber": "GRP-44821",
        },
        "provider": {
            "name": "Dr. Sarah Chen", "npi": "1234567890",
            "phone": "(415) 555-0142", "fax": "(415) 555-0143",
        },
        "medication": {
            "category": "specialty", "name": "Humira (adalimumab)",
            "ndcCode": "00074-4339-02", "dosage": "40mg",
            "frequency": "biweekly", "route": "subcutaneous", "duration": "365",
        },
        "specialty_details": {
            "stepTherapy": "Patient tried methotrexate 15mg weekly for 6 months (Jan-Jun 2025) with inadequate response. Also tried sulfasalazine 1000mg BID for 4 months (Jul-Oct 2025) which was discontinued due to GI intolerance.",
            "specialtyPharmacy": "cvs_specialty",
            "buyBill": "no",
        },
        "clinical": {
            "icd10Primary": "M06.9", "diagDescription": "Rheumatoid arthritis, unspecified",
            "icd10Secondary": "M79.3", "diagSecondary": "Panniculitis, unspecified",
            "clinicalRationale": "Patient is a 48-year-old female with moderate-to-severe RA diagnosed in 2024. Disease Activity Score (DAS28) is 5.1 indicating high disease activity. Failed two conventional DMARDs (methotrexate and sulfasalazine). CRP elevated at 2.8 mg/dL, ESR 42 mm/hr. Joint erosions on hand X-rays. Biologic therapy with adalimumab is medically necessary to prevent further joint damage and improve quality of life.",
            "urgency": "routine",
            "recentVisit": "yes",
        },
        "documents": {
            "clinical": ["clinical_notes_gonzalez_20260501.pdf", "rheumatology_consult_20260415.pdf"],
            "lab": ["cbc_cmp_20260428.pdf", "esr_crp_results_20260428.pdf"],
            "letter": [],
            "additionalNotes": "Patient has been compliant with all prior therapies. Physical therapy ongoing.",
        },
    },
    "controlled": {
        "description": "Controlled substance for chronic pain management",
        "patient": {
            "firstName": "James", "lastName": "Thompson",
            "dob": "1965-11-22", "memberId": "MBR-118374629",
            "insurancePlan": "silver_hmo",
        },
        "provider": {
            "name": "Dr. Michael Roberts", "npi": "9876543210",
            "phone": "(212) 555-0198", "fax": "(212) 555-0199",
        },
        "medication": {
            "category": "controlled", "name": "OxyContin (oxycodone ER)",
            "ndcCode": "59011-0420-10", "dosage": "20mg",
            "frequency": "bid", "route": "oral", "duration": "90",
        },
        "controlled_details": {
            "deaSchedule": "II",
            "deaNumber": "BR1234563",
            "pdmpCheck": "yes",
        },
        "clinical": {
            "icd10Primary": "G89.29", "diagDescription": "Other chronic pain",
            "clinicalRationale": "Patient is a 60-year-old male with chronic lumbar radiculopathy post-laminectomy (2023). Current pain management with tramadol 50mg TID is insufficient — NRS pain score averages 7/10. MRI shows recurrent disc herniation at L4-L5. Patient has completed physical therapy, epidural steroid injections (x3), and cognitive behavioral therapy for pain management. PDMP reviewed — no concerning patterns. Urine drug screen current and consistent.",
            "urgency": "routine",
            "recentVisit": "yes",
        },
        "documents": {
            "clinical": ["pain_management_notes_20260505.pdf"],
            "lab": ["urine_drug_screen_20260501.pdf"],
            "letter": [],
            "additionalNotes": "Patient has signed opioid treatment agreement. PDMP checked on 2026-05-05.",
        },
    },
    "brand_nonpreferred": {
        "description": "Brand-name medication after generic failure",
        "patient": {
            "firstName": "Emily", "lastName": "Park",
            "dob": "1990-07-08", "memberId": "MBR-557293184",
            "insurancePlan": "bronze_hdhp", "groupNumber": "GRP-10055",
        },
        "provider": {
            "name": "Dr. Lisa Wang", "npi": "5551234567",
            "phone": "(310) 555-0167",
        },
        "medication": {
            "category": "brand_nonpreferred", "name": "Synthroid (levothyroxine)",
            "ndcCode": "00074-6624-90", "dosage": "100mcg",
            "frequency": "daily", "route": "oral", "duration": "365",
        },
        "brand_details": {
            "genericTried": "Levothyroxine (generic) by Mylan",
            "genericReason": "narrow_therapeutic_index",
            "brandDetails": "Patient experienced significant TSH fluctuations (range 0.3-8.2 mIU/L over 6 months) on generic levothyroxine despite consistent dosing. Levothyroxine is classified as a narrow therapeutic index drug by the FDA. After switching to brand Synthroid, TSH stabilized at 2.1 mIU/L. Endocrinology recommends maintaining brand name to ensure consistent bioavailability.",
        },
        "clinical": {
            "icd10Primary": "E03.9", "diagDescription": "Hypothyroidism, unspecified",
            "clinicalRationale": "36-year-old female with primary hypothyroidism diagnosed in 2022. Documented therapeutic failure with generic levothyroxine — TSH instability despite good adherence. Levothyroxine is an NTI drug where small variations in bioavailability can have significant clinical impact. Brand Synthroid provides consistent drug delivery. Endocrinology concurs with brand necessity.",
            "urgency": "routine",
            "recentVisit": "yes",
        },
        "documents": {
            "clinical": ["endocrinology_notes_20260420.pdf"],
            "lab": ["thyroid_panel_history_2025_2026.pdf"],
            "letter": ["medical_necessity_letter_synthroid.pdf"],
        },
    },
    "gene_therapy": {
        "description": "Gene therapy for sickle cell disease",
        "patient": {
            "firstName": "Darius", "lastName": "Williams",
            "dob": "2001-02-14", "memberId": "MBR-773920146",
            "insurancePlan": "medicaid",
        },
        "provider": {
            "name": "Dr. Aisha Johnson", "npi": "3216549870",
            "phone": "(404) 555-0234", "fax": "(404) 555-0235",
        },
        "medication": {
            "category": "gene_therapy", "name": "Casgevy (exagamglogene autotemcel)",
            "ndcCode": "00078-1104-01", "dosage": "Single dose (autologous)",
            "frequency": "other", "route": "intravenous", "duration": "30",
        },
        "gene_therapy_details": {
            "remsEnrolled": "yes",
            "treatmentCenter": "Emory University Hospital — FACT-accredited, Center ID: EMR-2024-GT",
            "fdaIndication": "FDA-approved for treatment of sickle cell disease in patients aged 12+ with recurrent vaso-occlusive crises. Approved December 2023 under BLA 125746.",
            "therapyCost": "$2,200,000.00",
        },
        "clinical": {
            "icd10Primary": "D57.1", "diagDescription": "Sickle-cell disease without crisis",
            "icd10Secondary": "D57.00", "diagSecondary": "Hb-SS disease with crisis, unspecified",
            "clinicalRationale": "25-year-old male with severe sickle cell disease (HbSS genotype). History of 6+ vaso-occlusive crises per year requiring hospitalization despite maximum-dose hydroxyurea (35mg/kg/day) and chronic transfusion therapy. HbF level 8.2%. Eligible for Casgevy based on FDA-approved indication. Multidisciplinary team review completed at Emory including hematology, transplant medicine, social work, and psychology. Patient meets all REMS enrollment criteria.",
            "urgency": "urgent",
            "urgentJustification": "Patient has had 3 hospitalizations for VOC in the past 8 weeks. Current hemoglobin 6.8 g/dL. Increasing transfusion dependence with emerging alloantibody formation limits future transfusion options. Timely authorization needed to proceed with apheresis scheduling and myeloablative conditioning.",
            "recentVisit": "yes",
        },
        "documents": {
            "clinical": ["hematology_comprehensive_eval_20260501.pdf", "multidisciplinary_review_20260428.pdf"],
            "lab": ["hemoglobin_electrophoresis_20260501.pdf", "hla_typing_20260415.pdf", "cbc_reticulocyte_20260501.pdf"],
            "letter": ["medical_necessity_casgevy_williams.pdf"],
            "additionalNotes": "REMS enrollment confirmed. Apheresis tentatively scheduled for June 2026 pending authorization. Patient has completed psychological evaluation and fertility preservation counseling.",
        },
    },
    "compound": {
        "description": "Compounded medication for dermatological condition",
        "patient": {
            "firstName": "Rachel", "lastName": "Kim",
            "dob": "1985-09-30", "memberId": "MBR-334861027",
            "insurancePlan": "platinum_ppo", "groupNumber": "GRP-77120",
        },
        "provider": {
            "name": "Dr. David Patel", "npi": "7778889990",
            "phone": "(617) 555-0311",
        },
        "medication": {
            "category": "compound",
            "name": "Custom topical (tretinoin/niacinamide/tranexamic acid)",
            "ndcCode": "99999-0001-01",
            "dosage": "Apply thin layer to affected areas",
            "frequency": "daily", "route": "topical", "duration": "90",
        },
        "compound_details": {
            "compoundPharmacy": "Boston Compounding Pharmacy",
            "compoundIngredients": "Tretinoin 0.025% (5g), Niacinamide 4% (5g), Tranexamic acid 3% (5g), VersaBase cream QS to 60g",
            "compoundReason": "No commercially available product contains this specific combination. Patient has documented sensitivity to the preservatives (parabens, phenoxyethanol) in available tretinoin formulations. The compounded formulation uses a paraben-free base. Individual commercial products cannot be layered due to vehicle incompatibility causing irritation.",
        },
        "clinical": {
            "icd10Primary": "L81.1", "diagDescription": "Chloasma (melasma)",
            "clinicalRationale": "40-year-old female with refractory melasma (Fitzpatrick type IV). Failed 6 months of triple combination cream (Tri-Luma) due to paraben sensitivity causing contact dermatitis. Individual OTC products tried (niacinamide serum, tranexamic acid serum) provided partial improvement. Compounded formulation allows preservative-free delivery of all three active ingredients in a compatible vehicle for optimal efficacy.",
            "urgency": "routine",
            "recentVisit": "yes",
        },
        "documents": {
            "clinical": ["dermatology_notes_20260425.pdf"],
            "lab": [],
            "letter": [],
            "additionalNotes": "Patch testing confirmed paraben allergy. Photos documenting melasma severity available on request.",
        },
    },
}

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Prior Authorization Agent — an AI assistant that fills out healthcare
    prior authorization forms on behalf of medical staff. You navigate a multi-step
    web portal, reading fields, entering data, handling conditional logic, and
    submitting the completed request.

    ## Your Tools

    1. **read_page** — Read the current step's fields, their types, options, and
       current values. Always start by reading the page.
    2. **fill_field** — Set a form field's value. Use the field_id and provide the
       appropriate value (option value for selects/radios, text for inputs).
    3. **click_button** — Navigate: 'next' (validates then advances), 'back', or 'submit'.
    4. **upload_file** — Upload a supporting document on step 4.
    5. **get_form_state** — Get a full snapshot of all filled values and progress.

    ## Protocol

    For each step:
    1. **Read** the page to see what fields are available.
    2. **Fill** all required fields (and optional ones if data is available).
    3. Pay attention to **conditional fields** — when you set a medication category
       or urgency level, new fields may appear. The fill_field response will tell you.
    4. **Click next** to advance. If validation fails, read the errors and fix them.
    5. On step 5 (Review), verify the summary, check the attestation box, then submit.

    ## Important Rules

    - Always read the page before filling fields on a new step.
    - For select/radio fields, use the option **value**, not the display label.
    - Fill conditional fields when they become visible (e.g., specialty pharmacy
      fields when category is 'specialty').
    - Check the attestation before submitting.
    - Be systematic: fill fields in order, don't skip steps.
    - If a field fill fails, read the error and correct your approach.
""")


# ---------------------------------------------------------------------------
# Agent Result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result from a prior authorization agent run."""
    success: bool = False
    reference_number: str | None = None
    form_data: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    errors: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def summary(self) -> str:
        status = "SUBMITTED" if self.success else "INCOMPLETE"
        ref = f" (Ref: {self.reference_number})" if self.reference_number else ""
        return (f"[{status}]{ref} — {self.turns} turns, "
                f"{len(self.tool_calls)} tool calls, {len(self.errors)} errors")


# ---------------------------------------------------------------------------
# Prior Authorization Agent
# ---------------------------------------------------------------------------

class PriorAuthAgent:
    """
    Autonomous agent that fills out the Prior Authorization Portal form.

    Uses Claude with tool-use to navigate the multi-step form, handling
    conditional logic, validation, and submission. The agent interacts with
    a FormNavigator that simulates browser state.

    Example:
        agent = PriorAuthAgent()
        result = await agent.run(case_data=SAMPLE_CASES["specialty"])
        print(result.summary)
    """

    def __init__(
        self,
        model: str = MODEL,
        verbose: bool = True,
        submit: bool = True,
    ):
        self.client = anthropic.AsyncAnthropic()
        self.model = model
        self.verbose = verbose
        self.submit = submit

    async def run(
        self,
        case_data: dict[str, Any] | None = None,
        case_name: str | None = None,
        custom_instructions: str = "",
    ) -> AgentResult:
        """
        Run the agent to fill out a prior authorization form.

        Args:
            case_data: Patient case data dict. If None, uses a sample case.
            case_name: Name of a built-in sample case (e.g., 'specialty', 'controlled').
            custom_instructions: Additional instructions for the agent.

        Returns:
            AgentResult with submission status, reference number, and transcript.
        """
        if case_data is None:
            case_name = case_name or "specialty"
            if case_name not in SAMPLE_CASES:
                raise ValueError(f"Unknown case: {case_name}. Available: {list(SAMPLE_CASES.keys())}")
            case_data = SAMPLE_CASES[case_name]

        navigator = FormNavigator()
        result = AgentResult()

        # Build the user prompt with case data
        prompt = self._build_prompt(case_data, custom_instructions)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        self._log(f"Starting PA agent with model {self.model}")
        self._log(f"Case: {case_data.get('description', 'custom')}")

        for turn in range(1, MAX_TURNS + 1):
            result.turns = turn
            self._log(f"  Turn {turn}/{MAX_TURNS} (step {navigator.state.current_step}/5)")

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=ALL_TOOLS,
                messages=messages,
            )

            # Capture assistant text
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text

            if assistant_text:
                result.transcript.append({"role": "assistant", "text": assistant_text})
                if self.verbose:
                    # Print short excerpt
                    excerpt = assistant_text[:120].replace("\n", " ")
                    self._log(f"    Agent: {excerpt}{'...' if len(assistant_text) > 120 else ''}")

            # Check for end of turn
            if response.stop_reason == "end_turn":
                self._log("  Agent finished (end_turn).")
                break

            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = dict(block.input)

                    self._log(f"    [{tool_name}] {_summarize_tool(tool_name, tool_input)}")
                    result.tool_calls.append({"name": tool_name, "input": tool_input})

                    tool_result_str = execute_tool(navigator, tool_name, tool_input)
                    tool_result_data = json.loads(tool_result_str)

                    # Track errors
                    if isinstance(tool_result_data, dict) and not tool_result_data.get("success", True):
                        error_msg = tool_result_data.get("error", "Unknown error")
                        result.errors.append(f"[{tool_name}] {error_msg}")
                        self._log(f"    ERROR: {error_msg}")

                    # Check for successful submission
                    if (tool_name == "click_button" and
                            isinstance(tool_result_data, dict) and
                            tool_result_data.get("submitted")):
                        result.success = True
                        result.reference_number = tool_result_data.get("reference_number")
                        result.form_data = tool_result_data.get("form_data", {})
                        self._log(f"  SUBMITTED! Ref: {result.reference_number}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result_str,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break

            # Early exit if submitted
            if result.success:
                break

        if not result.success:
            self._log("  Agent did not complete submission.")
            result.form_data = navigator.get_form_state()

        return result

    def _build_prompt(self, case_data: dict[str, Any], custom_instructions: str = "") -> str:
        """Build the user prompt from case data."""
        submit_instruction = (
            "After reviewing the summary on step 5, check the attestation and submit."
            if self.submit else
            "Fill out all fields through step 4 but do NOT submit. Stop on step 5 after review."
        )

        parts = [
            "Please fill out the prior authorization form with the following patient case data.",
            "",
            f"**Case**: {case_data.get('description', 'Prior authorization request')}",
            "",
            "## Patient & Provider Data",
            "```json",
            json.dumps({k: v for k, v in case_data.items()
                        if k in ("patient", "provider")}, indent=2),
            "```",
            "",
            "## Medication Data",
            "```json",
            json.dumps({k: v for k, v in case_data.items()
                        if k in ("medication",)}, indent=2),
            "```",
        ]

        # Category-specific details
        category = case_data.get("medication", {}).get("category", "")
        detail_keys = {
            "specialty": "specialty_details",
            "controlled": "controlled_details",
            "brand_nonpreferred": "brand_details",
            "compound": "compound_details",
            "gene_therapy": "gene_therapy_details",
        }
        detail_key = detail_keys.get(category)
        if detail_key and detail_key in case_data:
            parts.extend([
                "",
                f"## {category.replace('_', ' ').title()} Details",
                "```json",
                json.dumps(case_data[detail_key], indent=2),
                "```",
            ])

        parts.extend([
            "",
            "## Clinical Data",
            "```json",
            json.dumps(case_data.get("clinical", {}), indent=2),
            "```",
            "",
            "## Documents",
            "```json",
            json.dumps(case_data.get("documents", {}), indent=2),
            "```",
            "",
            f"## Instructions",
            f"- Start by reading step 1, then fill all fields systematically.",
            f"- When you set the medication category, new conditional fields will appear — fill those too.",
            f"- Upload all listed documents on step 4.",
            f"- {submit_instruction}",
        ])

        if custom_instructions:
            parts.extend(["", f"## Additional Instructions", custom_instructions])

        return "\n".join(parts)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pa-agent] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Server Submission (optional: POST to running portal)
# ---------------------------------------------------------------------------

async def submit_to_server(form_data: dict[str, Any], portal_url: str = PORTAL_URL) -> dict[str, Any]:
    """Submit the completed form data to the running portal server."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{portal_url}/api/submit",
                json=form_data,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": f"Failed to submit to server: {e}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_tool(name: str, input_dict: dict[str, Any]) -> str:
    """Short log summary for a tool call."""
    if name == "fill_field":
        val = input_dict.get("value", "")
        val_short = val[:40] + "..." if len(val) > 40 else val
        return f"{input_dict.get('field_id')} = \"{val_short}\""
    if name == "click_button":
        return input_dict.get("button", "")
    if name == "upload_file":
        return f"{input_dict.get('category')}/{input_dict.get('filename')}"
    return name


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prior Authorization Agent — AI-powered form filling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Sample cases: specialty, controlled, brand_nonpreferred, compound, gene_therapy

            Examples:
              python -m agents.prior_auth_agent --case specialty
              python -m agents.prior_auth_agent --case controlled --no-submit
              python -m agents.prior_auth_agent --case gene_therapy --json
              python -m agents.prior_auth_agent --case-file ./my_case.json
              python -m agents.prior_auth_agent --list-cases
        """),
    )
    parser.add_argument("--case", "-c", choices=list(SAMPLE_CASES.keys()),
                        default="specialty", help="Built-in sample case to use")
    parser.add_argument("--case-file", "-f", help="Path to a JSON file with custom case data")
    parser.add_argument("--list-cases", action="store_true", help="List available sample cases and exit")
    parser.add_argument("--no-submit", action="store_true", help="Fill form but don't submit")
    parser.add_argument("--submit-to-server", action="store_true",
                        help="Also POST the result to the running portal server")
    parser.add_argument("--model", "-m", default=MODEL, help=f"Model to use (default: {MODEL})")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress logging")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    if args.list_cases:
        print("Available sample cases:\n")
        for name, case in SAMPLE_CASES.items():
            cat = case["medication"]["category"]
            print(f"  {name:20s}  {case['description']}")
            print(f"  {'':20s}  Patient: {case['patient']['firstName']} {case['patient']['lastName']}")
            print(f"  {'':20s}  Medication: {case['medication']['name']}")
            print()
        return

    # Load case data
    if args.case_file:
        case_path = Path(args.case_file)
        if not case_path.exists():
            print(f"Error: Case file not found: {case_path}", file=sys.stderr)
            sys.exit(1)
        with open(case_path) as f:
            case_data = json.load(f)
    else:
        case_data = SAMPLE_CASES[args.case]

    agent = PriorAuthAgent(
        model=args.model,
        verbose=not args.quiet,
        submit=not args.no_submit,
    )

    result = asyncio.run(agent.run(case_data=case_data, case_name=args.case))

    # Optionally submit to server
    if args.submit_to_server and result.success and result.form_data:
        server_result = asyncio.run(submit_to_server(result.form_data))
        if not args.quiet:
            print(f"[pa-agent] Server response: {json.dumps(server_result)}", file=sys.stderr)

    if args.json:
        output = {
            "success": result.success,
            "reference_number": result.reference_number,
            "turns": result.turns,
            "tool_calls": len(result.tool_calls),
            "errors": result.errors,
            "form_data": result.form_data,
            "timestamp": result.timestamp,
        }
        print(json.dumps(output, indent=2))
    else:
        print()
        print("=" * 60)
        print("PRIOR AUTHORIZATION AGENT — RESULT")
        print("=" * 60)
        print()
        print(f"  Status:     {'SUBMITTED' if result.success else 'INCOMPLETE'}")
        if result.reference_number:
            print(f"  Reference:  {result.reference_number}")
        print(f"  Turns:      {result.turns}")
        print(f"  Tool Calls: {len(result.tool_calls)}")
        if result.errors:
            print(f"  Errors:     {len(result.errors)}")
            for err in result.errors[:5]:
                print(f"    - {err}")
        print()

        if result.success and result.form_data:
            fd = result.form_data
            patient = fd.get("patient", {})
            med = fd.get("medication", {})
            print(f"  Patient:    {patient.get('firstName', '')} {patient.get('lastName', '')}")
            print(f"  Member ID:  {patient.get('memberId', '')}")
            print(f"  Medication: {med.get('name', '')}")
            print(f"  Category:   {med.get('category', '')}")
            print()

        print("-" * 60)
        print(result.summary)
        print()


if __name__ == "__main__":
    main()
