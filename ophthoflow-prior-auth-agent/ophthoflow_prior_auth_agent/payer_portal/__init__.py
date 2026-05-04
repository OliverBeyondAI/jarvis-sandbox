"""
Payer portal mock API for PA requirement checks.

This module implements a rules engine encoding common payer prior authorization
requirements for ophthalmology procedures, including:
- Required documentation per procedure/payer
- Clinical criteria and approved indications
- Step-therapy protocols (e.g., bevacizumab-first policies)
"""

from ophthoflow_prior_auth_agent.payer_portal.rules_engine import (
    PayerRulesEngine,
    check_requirements,
)

__all__ = ["PayerRulesEngine", "check_requirements"]
