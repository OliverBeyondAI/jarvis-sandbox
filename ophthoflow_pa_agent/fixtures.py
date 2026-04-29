"""
Example patient record fixtures for the OphthoFlow PA Agent.

Each fixture represents a realistic ophthalmology clinical scenario with
varying procedures, payers, urgency levels, and step therapy situations.
"""

from __future__ import annotations

import textwrap

FIXTURES: list[dict[str, str]] = [
    # -----------------------------------------------------------------
    # Fixture 1: Wet AMD — anti-VEGF switch after failed bevacizumab
    #   Payer: Aetna (step therapy applies, but bevacizumab already tried)
    #   Urgency: routine
    #   Expected: PA required, step therapy satisfied
    # -----------------------------------------------------------------
    {
        "label": "Wet AMD — Aflibercept switch (Aetna, step therapy satisfied)",
        "provider_name": "Dr. Sarah Chen, MD",
        "provider_npi": "1987654321",
        "record": textwrap.dedent("""\
            Patient: Margaret Thompson, DOB: 1951-06-14
            Visit Date: 2026-04-28

            Chief Complaint: Decreased vision OD x 3 months, progressive.

            History: 74-year-old female with known wet age-related macular
            degeneration (wAMD) in the right eye, diagnosed 6 months ago.
            Previously received 3 monthly injections of bevacizumab (Avastin)
            with partial response — subretinal fluid persists on OCT.

            Examination:
            - VA OD: 20/100 (corrected), OS: 20/25 (corrected)
            - IOP: 14 mmHg OD, 15 mmHg OS
            - Anterior segment: Normal OU
            - Fundus OD: Subfoveal CNV with subretinal hemorrhage, subretinal fluid
            - Fundus OS: Few small drusen, no CNV
            - OCT OD: Central subfield thickness 385 microns, persistent SRF
              and intraretinal fluid despite 3 prior bevacizumab injections

            Assessment & Plan:
            1. Wet AMD OD (H35.32) — incomplete response to bevacizumab
            2. Switch to aflibercept (Eylea) 2mg intravitreal injection OD
            3. Treat-and-extend protocol planned
            4. PA required — patient insured by Aetna, Member ID: AET887654321

            Treating Physician: Dr. Sarah Chen, MD
            NPI: 1987654321
            Practice: Pacific Retina Specialists
        """),
    },
    # -----------------------------------------------------------------
    # Fixture 2: Diabetic retinopathy — PRP laser (Medicare)
    #   Payer: Medicare (PA not typically required for PRP)
    #   Urgency: routine
    #   Expected: PA NOT required
    # -----------------------------------------------------------------
    {
        "label": "Diabetic Retinopathy — PRP Laser (Medicare, no PA needed)",
        "provider_name": "Dr. James Rodriguez, MD",
        "provider_npi": "1122334455",
        "record": textwrap.dedent("""\
            Patient: Robert Williams, DOB: 1958-11-22
            Visit Date: 2026-04-29

            Chief Complaint: Blurred vision OU, worse over the past 6 weeks.

            History: 67-year-old male with Type 2 diabetes mellitus x 20 years.
            HbA1c 8.9% (most recent). Known nonproliferative diabetic retinopathy
            progressing to proliferative diabetic retinopathy (PDR) bilaterally.
            No prior retinal laser treatment or intravitreal injections.

            Examination:
            - VA OD: 20/60 (corrected), OS: 20/50 (corrected)
            - IOP: 16 mmHg OD, 17 mmHg OS
            - Anterior segment: Mild nuclear sclerosis OU
            - Fundus OD: Neovascularization of disc (NVD) ~1.5 disc areas,
              scattered dot-blot hemorrhages, hard exudates
            - Fundus OS: Neovascularization elsewhere (NVE) inferior arcade,
              preretinal hemorrhage inferiorly
            - OCT OU: No clinically significant macular edema
            - FA: Extensive capillary nonperfusion and neovascularization OU

            Assessment & Plan:
            1. Proliferative diabetic retinopathy OU (E11.359)
            2. Plan: Panretinal photocoagulation (PRP) bilaterally, starting OD
            3. Follow up in 2 weeks for OS treatment
            4. Continue current diabetes management, recommend endocrine follow-up
            5. Patient insured by Medicare, Beneficiary ID: 1EG4-TE5-MK72

            Treating Physician: Dr. James Rodriguez, MD
            NPI: 1122334455
            Practice: Valley Eye Institute
        """),
    },
    # -----------------------------------------------------------------
    # Fixture 3: Retinal detachment — urgent vitrectomy (UnitedHealthcare)
    #   Payer: UnitedHealthcare
    #   Urgency: emergent
    #   Expected: PA required, expedited review
    # -----------------------------------------------------------------
    {
        "label": "Retinal Detachment — Emergent Vitrectomy (UHC)",
        "provider_name": "Dr. Anita Patel, MD",
        "provider_npi": "1567890123",
        "record": textwrap.dedent("""\
            Patient: David Kim, DOB: 1965-03-08
            Visit Date: 2026-04-29

            Chief Complaint: Sudden onset of flashing lights, floaters, and
            a "curtain" coming down over his left eye vision since yesterday morning.

            History: 61-year-old male, high myope (-8.00D OS), no prior ocular
            surgery. Presented emergently via on-call referral. No trauma.
            No history of retinal detachment in either eye. Family history
            significant for retinal detachment in his father.

            Examination:
            - VA OD: 20/20 (corrected), OS: 20/400 (corrected), count fingers
              in superior field only
            - IOP: 15 mmHg OD, 8 mmHg OS
            - Anterior segment: Quiet OU, pseudophakia OD, phakic OS
            - Fundus OD: Normal, flat retina, no breaks
            - Fundus OS: Macula-off rhegmatogenous retinal detachment with
              a large superotemporal horseshoe tear at 1 o'clock, ~2 disc
              diameters. Detachment extends from 10 o'clock to 4 o'clock
              involving the macula. Vitreous hemorrhage present.
            - B-scan OS: Confirms total macula-off detachment, no PVR noted

            Assessment & Plan:
            1. Rhegmatogenous retinal detachment OS, macula-off (H33.001)
            2. EMERGENT: Pars plana vitrectomy with membrane peel, endolaser,
               and gas tamponade OS — scheduled for tomorrow AM
            3. Patient instructed to position face-down, NPO after midnight
            4. Risks and benefits of surgery discussed extensively, informed
               consent obtained
            5. Patient insured by UnitedHealthcare, Member ID: UHC445566778

            Treating Physician: Dr. Anita Patel, MD
            NPI: 1567890123
            Practice: Eastside Retina Surgery Center
        """),
    },
]
