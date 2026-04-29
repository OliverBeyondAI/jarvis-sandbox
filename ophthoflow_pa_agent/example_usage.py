#!/usr/bin/env python3
"""
Example usage of the OphthoFlow mock payer portal API.

Run: python -m ophthoflow_pa_agent.example_usage
"""

from .payer_portal import PayerPortalAPI, PASubmission, ClinicalInfo
from .payer_portal.models import UrgencyLevel


def main():
    api = PayerPortalAPI(seed=42)

    print("=" * 70)
    print("OphthoFlow PA Agent — Mock Payer Portal Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Check PA requirements for anti-VEGF injection
    # ------------------------------------------------------------------
    print("\n--- 1. Check PA Requirement: Ranibizumab (Lucentis) via Aetna ---\n")
    req = api.check_pa_requirement("J2778", "AETNA")
    print(f"  Procedure:      {req.procedure_name} ({req.procedure_code})")
    print(f"  Payer:          {req.payer_name}")
    print(f"  Requires PA:    {req.requires_pa}")
    print(f"  Step therapy:   {req.step_therapy_required}")
    if req.step_therapy_details:
        print(f"  Step details:   {req.step_therapy_details}")
    print(f"  Est. review:    {req.estimated_review_hours} hours")
    print(f"  Required docs:  {', '.join(req.required_documents)}")
    print(f"  Message:        {req.message}")

    # ------------------------------------------------------------------
    # 2. Check a procedure that does NOT need PA
    # ------------------------------------------------------------------
    print("\n--- 2. Check PA Requirement: Bevacizumab (Avastin) via Aetna ---\n")
    req2 = api.check_pa_requirement("J9035", "AETNA")
    print(f"  Procedure:      {req2.procedure_name}")
    print(f"  Requires PA:    {req2.requires_pa}")
    print(f"  Message:        {req2.message}")

    # ------------------------------------------------------------------
    # 3. Submit PA for cataract surgery
    # ------------------------------------------------------------------
    print("\n--- 3. Submit PA: Cataract Surgery via UnitedHealthcare ---\n")
    clinical = ClinicalInfo(
        diagnosis_codes=["H25.11"],
        visual_acuity_od="20/70",
        visual_acuity_os="20/25",
        oct_findings="Posterior subcapsular opacity",
        functional_impairment="Difficulty driving at night, glare sensitivity",
        symptoms_duration_days=180,
        additional_notes="Nuclear sclerosis grade 3+",
    )
    submission = PASubmission(
        procedure_code="66984",
        payer_id="UNITEDHEALTHCARE",
        member_id="UHC987654321",
        provider_npi="1234567890",
        provider_name="Dr. Jane Smith, MD",
        patient_name="Mary Johnson",
        patient_dob="1948-07-22",
        date_of_service="2026-05-15",
        place_of_service="22",
        diagnosis_codes=["H25.11"],
        clinical_info=clinical,
        facility_name="City Eye Surgery Center",
    )
    resp = api.submit_pa(submission)
    print(f"  Auth Reference: {resp.auth_reference_number}")
    print(f"  Tracking ID:    {resp.tracking_id}")
    print(f"  Status:         {resp.status.value}")
    print(f"  Est. Decision:  {resp.estimated_determination_date}")
    print(f"  Docs received:  {len(resp.documents_received)}")
    print(f"  Docs missing:   {len(resp.missing_documents)}")
    print(f"  Message:        {resp.message}")
    print(f"  Next steps:")
    for step in resp.next_steps:
        print(f"    - {step}")

    # ------------------------------------------------------------------
    # 4. Check PA status
    # ------------------------------------------------------------------
    print(f"\n--- 4. Check PA Status: {resp.auth_reference_number} ---\n")
    status = api.check_pa_status(resp.auth_reference_number)
    print(f"  Status:         {status.status.value}")
    print(f"  Last updated:   {status.last_updated}")
    if status.approved_units:
        print(f"  Approved units: {status.approved_units}")
        print(f"  Valid from:     {status.approved_from_date}")
        print(f"  Valid through:  {status.approved_through_date}")
    if status.denial_reason:
        print(f"  Denial reason:  {status.denial_reason}")
        print(f"  Appeal by:      {status.appeal_deadline}")
    print(f"  Message:        {status.message}")

    # ------------------------------------------------------------------
    # 5. Urgent retinal detachment repair
    # ------------------------------------------------------------------
    print("\n--- 5. Urgent PA: Retinal Detachment Repair via Medicare ---\n")
    rd_clinical = ClinicalInfo(
        diagnosis_codes=["H33.001"],
        visual_acuity_od="20/400",
        visual_acuity_os="20/20",
        oct_findings="Macula-off rhegmatogenous RD with horseshoe tear",
        symptoms_duration_days=2,
        functional_impairment="Acute vision loss, curtain-like visual field defect",
        additional_notes="B-scan confirms total RD with single horseshoe tear at 10 o'clock",
    )
    rd_submission = PASubmission(
        procedure_code="67108",
        payer_id="MEDICARE",
        member_id="1EG4-TE5-MK72",
        provider_npi="9876543210",
        provider_name="Dr. Robert Chen, MD",
        patient_name="James Williams",
        patient_dob="1962-11-03",
        date_of_service="2026-04-30",
        place_of_service="22",
        diagnosis_codes=["H33.001"],
        clinical_info=rd_clinical,
        urgency=UrgencyLevel.EMERGENT,
        facility_name="University Retina Center",
    )
    rd_resp = api.submit_pa(rd_submission)
    print(f"  Auth Reference: {rd_resp.auth_reference_number}")
    print(f"  Urgency:        {rd_resp.urgency.value}")
    print(f"  Status:         {rd_resp.status.value}")
    print(f"  Est. Decision:  {rd_resp.estimated_determination_date}")
    print(f"  Message:        {rd_resp.message}")

    # ------------------------------------------------------------------
    # 6. List all supported procedures and payers
    # ------------------------------------------------------------------
    print("\n--- 6. Supported Procedures ---\n")
    for proc in api.list_supported_procedures():
        pa_tag = "PA Required" if proc["requires_pa"] else "No PA"
        print(f"  {proc['code']:>6}  {proc['name']:<55} [{pa_tag}]")

    print("\n--- 7. Supported Payers ---\n")
    for payer in api.list_supported_payers():
        print(f"  {payer['key']:<20} {payer['name']:<30} (ID: {payer['payer_id']})")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
