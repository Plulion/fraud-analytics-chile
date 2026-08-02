Fraud Analytics Chile

Fraud Analytics Chile is an educational and portfolio-oriented platform for fraud risk assessment, alert management, investigation workflows, audit trails, evidence handling, feedback governance, and machine-learning preparation.

The project is being developed incrementally as a professional fraud analytics MVP adapted to Chilean business scenarios. All cases, organizations, identifiers, amounts, and investigation records included in the repository are synthetic.

Project Status

Current functional baseline:

128 automated tests passing

0 known warnings

risk scoring implemented

digital fraud assessment implemented

alert generation and prioritization implemented

case workflow and audit trail implemented

evidence registration and integrity verification implemented

operational dashboard implemented

supervised closure and controlled reopening implemented

investigation feedback lifecycle implemented

clean training dataset generation implemented

The machine-learning model, APIs, distributed architecture, frontend, and containerized deployment are still under development.

Business Problem

Organizations receive many potentially suspicious events, but not every event should automatically become a fraud investigation.

This project separates the main operational concepts:

event
→ assessment
→ alert
→ prioritization
→ investigation case
→ evidence
→ resolution
→ feedback
→ training dataset

This separation improves traceability, supports human review, and reduces the risk of treating a score as proof of fraud.

Important Interpretation

The current scores are educational prioritization scores.

They are not:

probabilities of guilt;

legal conclusions;

substitutes for evidence;

substitutes for human investigation;

production-ready decision systems.

A fraud alert indicates that a case deserves review. It does not prove that fraud occurred.

Main Capabilities

Risk assessment

The project evaluates synthetic cases using opportunity, pressure, rationalization, aggravating factors, and digital fraud signals.

Alert and case management

Assessments may generate professional alerts. Alerts are prioritized according to operational capacity and may be converted into investigation cases.

Auditability

Relevant workflow changes generate append-only audit events. This supports traceability of assignments, status changes, resolutions, closures, and reopenings.

Evidence handling

Evidence files are copied into a controlled evidence store. SHA-256 hashes are recorded to detect later changes.

A cryptographic hash can detect file modification, but it does not prove that the original evidence was truthful or authentic.

Resolution and feedback

An assigned analyst proposes a resolution and an assigned supervisor approves it. Closed cases generate supervised feedback labels.

When a case is reopened:

the previous resolution is preserved;

the previous feedback is invalidated;

historical records are not deleted;

invalidated feedback is excluded from model training.

Machine-learning preparation

The project currently prepares a clean supervised dataset by:

accepting only active feedback;

checking binary labels;

excluding invalidated resolutions;

preventing temporal leakage;

rejecting forbidden post-investigation variables;

recording exclusions and quality metrics.

Repository Structure

fraud-analytics-chile/
├── data/                  # Synthetic and reference input data
├── outputs/               # Locally generated artifacts
├── scripts/               # Demonstration and orchestration scripts
├── src/                   # Application and analytical modules
├── tests/                 # Automated test suite
├── docs/                  # Architecture, process, data, testing, and security documentation
├── README.md
├── CHANGELOG.md
├── pyproject.toml
└── requirements.txt

Generated CSV, JSON, chart, dashboard, audit, and evidence artifacts are excluded from version control. They can be reproduced by running the documented scripts.

Requirements

Python 3.11 or later

pip

Git

VS Code or another Python-compatible IDE

Installation

Clone the repository:

git clone https://github.com/Plulion/fraud-analytics-chile.git
cd fraud-analytics-chile

Create and activate a virtual environment on macOS or Linux:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Run the Main Risk Assessment

python -m src.main

Custom paths may also be supplied:

python -m src.main \
  --input data/casos_sinteticos_chile.csv \
  --output outputs/evaluacion_riesgo.csv

Run Key Demonstrations

python -m scripts.run_professional_pipeline
python -m scripts.run_investigation_prioritization
python -m scripts.run_investigation_demo
python -m scripts.run_evidence_demo
python -m scripts.run_operational_dashboard
python -m scripts.run_case_closure_demo
python -m scripts.run_case_reopening_demo
python -m scripts.run_training_dataset

Each script writes reproducible artifacts under outputs/.

Run Tests

python -m pytest -q

Current documented baseline:

128 passed

Documentation

Architecture

Business Process

Data Dictionary

Testing Strategy

Security and Privacy

Data Policy

All repository data is synthetic and intended only for education, demonstrations, and software testing.

The project must not include:

real customer information;

real personal identifiers;

real investigation evidence;

passwords or credentials;

API keys;

confidential company data.

Ethical Use

The system must be used with human oversight.

A high score or alert must not independently trigger an adverse action against a person or organization. Any operational use would require validated data, documented governance, legal review, security controls, bias assessment, monitoring, and an appropriate appeal or review mechanism.

Roadmap

Completed

expert risk scoring

digital fraud assessment

fraud typology enrichment

professional alert pipeline

investigation prioritization

case workflow

audit trail

evidence registration

operational dashboard

closure and reopening governance

feedback lifecycle

clean training dataset generation

In progress

temporal train, validation, and test partitioning

documentation regularization

portfolio-quality code comments and docstrings

Planned

baseline machine-learning model

model evaluation and threshold selection

explainability and monitoring

FastAPI inference service

PostgreSQL persistence

RabbitMQ asynchronous processing

Spring Boot business services

Next.js frontend

Docker Compose integration

final portfolio demonstration

Disclaimer

This repository is an educational project. It is not a production fraud detection system and must not be used as legal, financial, compliance, or investigative advice.