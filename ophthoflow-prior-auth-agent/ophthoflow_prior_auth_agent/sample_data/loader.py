"""
Utility for loading and validating sample patient data against schemas.

Usage:
    from ophthoflow_prior_auth_agent.sample_data.loader import load_samples

    cases = load_samples("intravitreal_injections")
    for case in cases:
        print(case.case_id, case.completeness)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from ophthoflow_prior_auth_agent.schemas import OphthalmologyPatientCase

_DATA_DIR = Path(__file__).parent

AVAILABLE_DATASETS = (
    "intravitreal_injections",
    "cataract_surgery",
    "retinal_imaging",
)


def load_samples(
    dataset: Literal[
        "intravitreal_injections",
        "cataract_surgery",
        "retinal_imaging",
    ],
) -> list[OphthalmologyPatientCase]:
    """Load and validate sample cases from a JSON dataset.

    Returns a list of validated ``OphthalmologyPatientCase`` instances.
    Raises ``FileNotFoundError`` if the dataset does not exist, or
    ``pydantic.ValidationError`` if any record fails schema validation.
    """
    path = _DATA_DIR / f"{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = json.loads(path.read_text())
    return [OphthalmologyPatientCase.model_validate(record) for record in raw]


def load_all_samples() -> dict[str, list[OphthalmologyPatientCase]]:
    """Load every available dataset, keyed by name."""
    return {name: load_samples(name) for name in AVAILABLE_DATASETS}
