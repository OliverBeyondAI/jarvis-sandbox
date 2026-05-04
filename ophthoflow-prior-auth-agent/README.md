# OphthoFlow Prior Auth Agent

An AI-powered ophthalmology prior authorization agent built on the Claude Agent SDK via AWS Bedrock. Processes structured patient JSON through a complete PA workflow: parsing patient data, checking payer requirements, analyzing missing information, assessing denial risk, and drafting PA request letters.

## Architecture

```
ophthoflow-prior-auth-agent/
├── pyproject.toml                          # Package config & dependencies
├── run_demo.py                             # Demo runner (no LLM required)
├── README.md
└── ophthoflow_prior_auth_agent/
    ├── __init__.py                         # Package metadata
    ├── config.py                           # Bedrock/model configuration
    ├── models.py                           # Pydantic models (PatientRecord, PARequirement, PALetter)
    ├── schemas.py                          # Patient case schemas (demographics, insurance, clinical)
    ├── tools.py                            # Tool schemas (MCP-compatible) + implementations
    ├── agent.py                            # Core agent loop with tool dispatch
    ├── main.py                             # CLI entry point
    ├── payer_portal/                       # Payer rules engine
    │   ├── __init__.py
    │   └── rules_engine.py                 # PA rules for 5 payers × 8 procedures
    ├── sample_data/                        # Sample patient cases
    │   ├── loader.py
    │   ├── intravitreal_injections.json
    │   ├── cataract_surgery.json
    │   └── retinal_imaging.json
    └── tests/
        ├── test_rules_engine.py            # Payer rules engine tests
        └── test_tools.py                   # Agent tool tests
```

### Pipeline

The agent follows a five-step pipeline:

1. **Parse** — `parse_patient_json` extracts structured fields (demographics, insurance, diagnoses, procedures, imaging, prior treatments) from patient JSON input.
2. **Check** — `check_pa_requirements` queries the payer rules engine for PA requirements, step therapy status, required documents, and review timelines.
3. **Analyze** — `analyze_missing_information` identifies gaps in the clinical record (missing demographics, imaging, visual acuity, treatment history) with severity ratings.
4. **Assess** — `assess_denial_risk` evaluates denial probability based on step therapy compliance, clinical criteria coverage, documentation completeness, and payer-specific factors.
5. **Draft** — `draft_pa_letter` generates a professional PA request letter with clinical justification, diagnosis codes, treatment history, and provider details.

## Setup

### Prerequisites

- Python 3.9+
- AWS credentials configured for Bedrock access (for LLM mode)
- `pydantic>=2.0.0`, `anthropic[bedrock]>=0.97.0`

### Installation

```bash
cd ophthoflow-prior-auth-agent
pip install -e .

# For development
pip install -e ".[dev]"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPHTHOFLOW_USE_BEDROCK` | `true` | Use AWS Bedrock (`true`) or direct Anthropic API (`false`) |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `ANTHROPIC_API_KEY` | — | Required only when using direct API mode |

## Usage

### Local Demo (no LLM required)

Run the full PA workflow locally against sample patient cases:

```bash
# List all available sample cases
python run_demo.py --list

# Run demo for a specific case
python run_demo.py --dataset intravitreal_injections --case 0

# Run all cases across all datasets
python run_demo.py --all
```

### CLI with Sample Data

```bash
# Process a built-in sample case
ophthoflow-pa --sample intravitreal_injections --case-index 0

# Local demo mode (no LLM)
ophthoflow-pa --demo
ophthoflow-pa --demo --sample cataract_surgery --case-index 1
```

### Process a Patient JSON File

```bash
ophthoflow-pa --file patient_case.json
ophthoflow-pa --file patient_case.json --verbose
```

### Process from stdin

```bash
cat patient_case.json | ophthoflow-pa --stdin
```

### Use direct Anthropic API instead of Bedrock

```bash
ophthoflow-pa --file patient_case.json --direct
```

## Output

For each patient case, the agent produces:

1. **Parsed Patient Record** — Structured fields extracted from the input (name, DOB, diagnosis, ICD-10 codes, procedure, CPT code, payer, prior treatments, imaging).
2. **PA Determination** — Whether PA is required, required documents, step therapy status, estimated review timeline, and approved indications.
3. **Missing Information Analysis** — Categorized gaps (critical/high/moderate) in the clinical record that could delay or prevent approval.
4. **Denial Risk Assessment** — Risk level (LOW/MODERATE/HIGH), risk score, specific risk factors, and actionable recommendations.
5. **Draft PA Letter** — A professional, clinically detailed letter ready for provider review and signature (only generated when PA is required).

## Supported Procedures

| Code | Procedure |
|------|-----------|
| J2778 | Ranibizumab (Lucentis) injection |
| J0178 | Aflibercept (Eylea) injection |
| J9035 | Bevacizumab (Avastin) injection |
| J2503 | Faricimab (Vabysmo) injection |
| 66984 | Cataract surgery (standard) |
| 66982 | Cataract surgery (complex) |
| 67108 | Vitrectomy for retinal detachment |
| 67228 | Panretinal photocoagulation (PRP) |

## Supported Payers

| Payer | Step Therapy | Notes |
|-------|-------------|-------|
| Aetna | Yes (bevacizumab first) | 5-day standard review |
| UnitedHealthcare | No | 3-day fast-track for anti-VEGFs |
| Cigna | Yes (bevacizumab first) | Similar to Aetna |
| Blue Cross Blue Shield | No | Standard timelines |
| Medicare Part B | No | No formal PA; medical necessity documented for audit |

## Running Tests

```bash
pytest ophthoflow_prior_auth_agent/tests/ -v
```

27 tests covering the payer rules engine (15 tests) and agent tools (12 tests).
