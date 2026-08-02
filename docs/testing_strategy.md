Testing Strategy

Purpose

The test suite protects fraud-analysis rules, workflow controls, auditability, and machine-learning data quality.

The documented baseline contains 128 passing tests.

Test Framework

The project uses:

pytest for automated tests;

pandas DataFrames for test fixtures and expected outputs;

isolated temporary paths where file operations require them.

Configuration is stored in pyproject.toml.

Run the Full Suite

python -m pytest -q

Test Categories

Risk-scoring tests

Validate:

score calculations;

allowed ranges;

missing inputs;

risk-level assignment;

recommended actions.

Digital-fraud tests

Validate:

device signals;

network signals;

authentication signals;

transaction signals;

total digital score;

triggered-signal lists.

Taxonomy and enrichment tests

Validate:

allowed fraud types;

typology descriptions;

expected evidence sources;

recommended controls.

Consolidation and reporting tests

Validate:

highest-risk consolidation;

output schemas;

generated metrics;

chart and report creation.

Professional pipeline tests

Validate:

assessment creation;

alert-generation conditions;

assessment and alert separation;

required identifiers and timestamps.

Prioritization tests

Validate:

priority calculations;

selection capacity;

selected and waiting queues;

invalid capacity values.

Workflow tests

Validate:

permitted transitions;

rejected transitions;

analyst and supervisor assignment;

audit-event creation;

case availability.

Evidence tests

Validate:

note registration;

evidence file existence;

rejection of empty evidence;

SHA-256 creation;

integrity verification;

chain-of-custody events.

Governance tests

Validate:

source-system registration;

lineage generation;

file hashes;

RACI requirements;

quality-report output.

Dashboard tests

Validate:

assignment times;

investigation-start times;

SLA calculations;

analyst workload;

KPI aggregation;

rejection of invalid temporal differences.

Closure tests

Validate:

analyst-only proposal;

supervisor-only approval;

allowed outcomes;

amount validation;

controlled closure;

audit-event sequence;

feedback generation.

Reopening tests

Validate:

supervisor authority;

reopening reason;

target status;

preservation of prior resolution;

feedback invalidation;

reopening audit events.

Training-dataset tests

Validate:

active-feedback inclusion;

invalidated-feedback exclusion;

temporal leakage prevention;

target-leakage prevention;

duplicate-case rejection;

binary-label consistency;

lifecycle resolution;

exclusion reporting.

Test Naming

Test names should describe observable behavior.

Preferred:

def test_invalidated_feedback_is_excluded() -> None:
    ...

Avoid:

def test_1() -> None:
    ...

Documentation in Tests

A test does not require comments for every assertion.

Docstrings should be added when the business reason is not obvious, especially for:

segregation of duties;

feedback invalidation;

temporal leakage;

evidence preservation;

audit-history requirements.

Regression Rule

After documentation or refactoring changes, the full test suite must pass before committing:

python -m pytest -q

Documentation-only changes must not alter functional results.

Future Quality Controls

Planned additions include:

coverage measurement;

branch coverage for critical workflow rules;

Ruff linting;

static type checking;

GitHub Actions;

integration tests;

API contract tests;

database migration tests;

end-to-end tests;

performance and load tests.

Minimum Acceptance Criteria

A change is acceptable when:

relevant tests are added or updated;

the full suite passes;

no new warnings appear;

generated artifacts remain reproducible;

business rules remain documented;

synthetic-data restrictions are preserved.