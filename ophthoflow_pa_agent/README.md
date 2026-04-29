# OphthoFlow PA Agent

An AI-powered ophthalmology prior authorization agent built on the Anthropic Python SDK. OphthoFlow automates the PA workflow: parsing clinical notes, checking payer requirements, and drafting PA request letters using Claude's tool-use capabilities.

## Architecture

```
ophthoflow_pa_agent/
├── main.py              # CLI entry point — runs the full PA pipeline
├── agent.py             # Anthropic SDK agentic tool-use loop + tool definitions
├── fixtures.py          # Example patient records for testing
├── example_usage.py     # Standalone demo of the payer portal mock API
├── payer_portal/
│   ├── mock_api.py      # Simulated payer portal (check, submit, status)
│   ├── models.py        # Data models (enums, dataclasses)
│   └── procedure_data.py# Procedure catalog, ICD-10 codes, payer profiles
└── tests/
    └── test_mock_api.py # pytest suite for the mock payer portal
```

### Workflow

1. **Parse** — `parse_patient_record` uses Claude to extract structured fields (diagnosis, treatment, payer, clinical findings) from free-text clinical notes.
2. **Check** — `check_pa_requirements` queries the mock payer portal for PA requirements, step therapy, required documents, and review timelines.
3. **Draft** — `draft_pa_letter` generates a professional PA request letter with clinical justification, diagnosis codes, and payer-specific documentation.

## Setup

### Prerequisites

- Python 3.11+
- An Anthropic API key

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd jarvis-sandbox

# Install dependencies
pip install anthropic

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Verify the mock API works (no API key needed)

```bash
python -m ophthoflow_pa_agent.example_usage
python -m pytest ophthoflow_pa_agent/tests/ -v
```

## Usage

### Run an example fixture

The agent ships with three built-in patient scenarios:

| # | Scenario | Payer | Expected Outcome |
|---|----------|-------|------------------|
| 1 | Wet AMD — aflibercept switch after failed bevacizumab | Aetna | PA required, step therapy satisfied |
| 2 | Proliferative diabetic retinopathy — PRP laser | Medicare | PA **not** required |
| 3 | Macula-off retinal detachment — emergent vitrectomy | UnitedHealthcare | PA required, expedited review |

```bash
# Run fixture 1 (wet AMD case)
python -m ophthoflow_pa_agent.main --example 1

# Run fixture 2 (PRP laser — no PA needed)
python -m ophthoflow_pa_agent.main --example 2

# Run fixture 3 (emergent retinal detachment)
python -m ophthoflow_pa_agent.main --example 3

# Run all three fixtures
python -m ophthoflow_pa_agent.main --all

# List available fixtures
python -m ophthoflow_pa_agent.main --list
```

### Run with a custom patient record

```bash
# From a file
python -m ophthoflow_pa_agent.main --file patient_note.txt

# From stdin
cat patient_note.txt | python -m ophthoflow_pa_agent.main --stdin
```

### Interactive agent mode

Launch a conversational session where you can type patient records and ask follow-up questions:

```bash
python -m ophthoflow_pa_agent.main --interactive
```

## Output

For each patient record, the agent outputs:

1. **Parsed Patient Record** — Structured fields extracted from the clinical note (name, DOB, diagnosis, ICD-10 codes, procedure code, payer, etc.)
2. **PA Determination** — Whether PA is required, required documents, step therapy status, estimated review timeline, and approved indications.
3. **Draft PA Letter** — A professional, clinically detailed letter ready for provider review and signature (only generated when PA is required).

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
| 67113 | Complex vitrectomy |
| 67101 | Scleral buckle |
| 65855 | Laser trabeculoplasty (SLT) |
| 67228 | Panretinal photocoagulation (PRP) |
| 67040 | Epiretinal membrane peel |

## Supported Payers

| Key | Payer | Step Therapy |
|-----|-------|-------------|
| AETNA | Aetna | Yes (bevacizumab first for anti-VEGF) |
| UNITEDHEALTHCARE | UnitedHealthcare | No |
| CIGNA | Cigna | Yes (bevacizumab first for anti-VEGF) |
| BCBS | Blue Cross Blue Shield | No |
| MEDICARE | Medicare Part B | No |

## Running Tests

```bash
python -m pytest ophthoflow_pa_agent/tests/ -v
```

The test suite covers the mock payer portal API including PA requirement checks, submission workflows, status tracking, step therapy enforcement, and error handling.
