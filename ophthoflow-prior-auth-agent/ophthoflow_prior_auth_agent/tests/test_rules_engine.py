"""Tests for the payer rules engine."""

from ophthoflow_prior_auth_agent.payer_portal.rules_engine import (
    PayerRulesEngine,
    check_requirements,
)


def test_engine_initialization():
    """Engine loads all payer rules."""
    engine = PayerRulesEngine()
    payers = engine.list_payers()
    assert "Aetna" in payers
    assert "UnitedHealthcare" in payers
    assert "Cigna" in payers
    assert "Blue Cross Blue Shield" in payers
    assert "Medicare Part B" in payers


def test_engine_procedures():
    """Engine supports all documented procedures."""
    engine = PayerRulesEngine()
    procedures = engine.list_procedures()
    assert "J2778" in procedures
    assert "J0178" in procedures
    assert "J9035" in procedures
    assert "J2503" in procedures
    assert "66984" in procedures
    assert "66982" in procedures
    assert "67108" in procedures
    assert "67228" in procedures


def test_payer_alias_normalization():
    """Payer aliases resolve correctly."""
    engine = PayerRulesEngine()
    # UHC aliases
    assert engine.get_rule("UHC", "J2778") is not None
    assert engine.get_rule("United", "J2778") is not None
    assert engine.get_rule("United Healthcare", "J2778") is not None
    # BCBS aliases
    assert engine.get_rule("BCBS", "J2778") is not None
    assert engine.get_rule("Blue Cross", "J2778") is not None
    # Medicare aliases
    assert engine.get_rule("Medicare", "66984") is not None
    assert engine.get_rule("CMS", "66984") is not None


def test_aetna_step_therapy_required():
    """Aetna requires step-therapy for brand anti-VEGFs."""
    engine = PayerRulesEngine()
    rule = engine.get_rule("Aetna", "J2778")
    assert rule is not None
    assert rule.step_therapy is not None
    assert "bevacizumab" in rule.step_therapy.required_prior_agents


def test_step_therapy_met():
    """Step therapy is met when bevacizumab is in prior treatments."""
    engine = PayerRulesEngine()
    met, details = engine.evaluate_step_therapy(
        "Aetna", "J2778", prior_treatments=["bevacizumab x6 injections"]
    )
    assert met is True
    assert "MET" in details


def test_step_therapy_not_met():
    """Step therapy is NOT met without bevacizumab in history."""
    engine = PayerRulesEngine()
    met, details = engine.evaluate_step_therapy(
        "Aetna", "J2778", prior_treatments=["artificial tears", "vitamin supplements"]
    )
    assert met is False
    assert "NOT MET" in details


def test_step_therapy_not_applicable():
    """UHC has no step-therapy for anti-VEGFs."""
    engine = PayerRulesEngine()
    met, details = engine.evaluate_step_therapy(
        "UnitedHealthcare", "J2778", prior_treatments=[]
    )
    assert met is None
    assert "No step-therapy" in details


def test_bevacizumab_no_pa_required():
    """Bevacizumab (Avastin) does not require PA from any commercial payer."""
    result = check_requirements(payer="Aetna", cpt_code="J9035")
    assert result["pa_required"] is False

    result = check_requirements(payer="Cigna", cpt_code="J9035")
    assert result["pa_required"] is False


def test_medicare_no_pa():
    """Medicare Part B does not require formal PA for most procedures."""
    result = check_requirements(payer="Medicare Part B", cpt_code="J2778")
    assert result["pa_required"] is False

    result = check_requirements(payer="Medicare Part B", cpt_code="66984")
    assert result["pa_required"] is False


def test_icd10_coverage_matching():
    """ICD-10 codes matching approved diagnoses are recognized."""
    engine = PayerRulesEngine()
    covered, criteria = engine.check_icd10_coverage(
        "Aetna", "J2778", icd10_codes=["H35.31"]
    )
    assert covered is True
    assert len(criteria) > 0


def test_icd10_coverage_no_match():
    """Unrelated ICD-10 codes do not match."""
    engine = PayerRulesEngine()
    covered, criteria = engine.check_icd10_coverage(
        "Aetna", "J2778", icd10_codes=["Z00.00"]  # routine exam code
    )
    assert covered is False


def test_check_requirements_full_workflow():
    """Full check_requirements call returns expected structure."""
    result = check_requirements(
        payer="Aetna",
        cpt_code="J2778",
        icd10_codes=["H35.31"],
        prior_treatments=["bevacizumab x4 injections, suboptimal response"],
    )
    assert result["pa_required"] is True
    assert result["step_therapy_met"] is True
    assert len(result["required_documents"]) > 0
    assert len(result["approved_indications"]) > 0
    assert result["review_timeline_days"] == 5


def test_check_requirements_unknown_payer():
    """Unknown payer/procedure returns safe default (PA required)."""
    result = check_requirements(payer="Unknown Payer", cpt_code="99999")
    assert result["pa_required"] is True
    assert "rule found" in result["notes"].lower() or "not found" in result["notes"].lower()


def test_cataract_surgery_requirements():
    """Cataract surgery has correct documentation requirements."""
    result = check_requirements(
        payer="Aetna",
        cpt_code="66984",
        icd10_codes=["H25.10"],
    )
    assert result["pa_required"] is True
    assert any("visual acuity" in doc.lower() for doc in result["required_documents"])


def test_vitrectomy_expedited_timeline():
    """Vitrectomy has short review timeline due to urgency."""
    engine = PayerRulesEngine()
    rule = engine.get_rule("Aetna", "67108")
    assert rule is not None
    assert rule.review_timeline_days <= 3
    assert rule.urgency_expedite_days == 1
