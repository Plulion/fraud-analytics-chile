Architecture

Purpose

This document describes the current logical architecture of Fraud Analytics Chile.

The project is currently implemented as a modular Python application. Future phases will introduce API, persistence, messaging, frontend, and containerized deployment components.

Architectural Principles

The project follows these principles:

Separate assessment from investigation.

Preserve historical records instead of overwriting them.

Keep audit events append-only.

Require human review for case outcomes.

Treat scores as prioritization aids, not proof.

Exclude invalidated feedback from training.

Prevent temporal and target leakage.

Keep synthetic data clearly identified.

Make outputs reproducible.

Validate each functional increment with automated tests.

Current Logical Flow

flowchart LR
    A[Synthetic Cases and Events] --> B[Risk Assessment]
    B --> C[Professional Assessment]
    C --> D[Alert Generation]
    D --> E[Investigation Prioritization]
    E --> F[Case Workflow]
    F --> G[Evidence and Notes]
    G --> H[Resolution Proposal]
    H --> I[Supervisor Approval]
    I --> J[Case Closure]
    J --> K[Investigation Feedback]
    K --> L[Clean Training Dataset]
    J --> M[Controlled Reopening]
    M --> N[Feedback Invalidation]
    N --> L

Main Modules

Risk scoring

Primary modules:

src/risk_scoring.py

src/main.py

Responsibilities:

validate synthetic case inputs;

calculate educational risk dimensions;

calculate total prioritization scores;

assign risk levels and recommended actions;

export risk-assessment results.

Digital fraud assessment

Primary modules:

src/digital_fraud.py

src/digital_batch.py

src/digital_reporting.py

Responsibilities:

evaluate device, network, authentication, and transaction signals;

calculate digital risk scores;

identify triggered signals;

produce batch outputs and digital reports.

Fraud taxonomy and consolidation

Primary modules:

src/fraud_taxonomy.py

src/typology_enrichment.py

src/case_consolidation.py

src/consolidated_reporting.py

Responsibilities:

classify fraud typologies;

enrich cases with descriptions and expected evidence;

consolidate general and digital assessments;

retain the highest relevant risk level.

Professional alert pipeline

Primary modules:

src/professional_pipeline.py

src/investigation_prioritization.py

Responsibilities:

separate assessment records from alerts;

generate alerts only when configured criteria are met;

calculate operational priority;

split alerts into selected and waiting queues according to capacity.

Investigation workflow

Primary modules:

src/investigation_workflow.py

src/case_metrics.py

Responsibilities:

maintain case status;

assign analysts and supervisors;

validate allowed transitions;

record append-only audit events;

calculate operational case metrics.

Evidence and investigation records

Primary module:

src/investigation_records.py

Responsibilities:

register investigation notes;

copy evidence into a controlled local store;

calculate SHA-256 hashes;

maintain evidence metadata and chain-of-custody events;

reject missing or empty evidence files.

Data governance

Primary module:

src/data_governance.py

Responsibilities:

validate source-system registrations;

maintain lineage records;

calculate file hashes;

validate RACI ownership;

report data-quality issues.

Operational dashboard

Primary module:

src/operational_dashboard.py

Responsibilities:

calculate assignment and investigation-start times;

calculate SLA compliance;

aggregate analyst workload;

generate operational KPIs.

Resolution, closure, and reopening

Primary modules:

src/case_closure.py

src/case_reopening.py

Responsibilities:

allow analysts to propose outcomes;

require supervisor approval;

close cases using controlled transitions;

preserve resolutions;

reopen cases under supervisor control;

supersede previous resolutions;

invalidate previous feedback.

Feedback and machine-learning preparation

Primary modules:

src/model_feedback.py

src/training_dataset.py

Responsibilities:

transform approved outcomes into supervised labels;

perform initial backtesting;

resolve feedback lifecycle;

exclude inactive labels;

prevent target and temporal leakage;

generate training datasets and exclusion reports.

Current Deployment Model

The current project runs locally as Python modules and command-line scripts.

CSV input
→ Python module
→ CSV, JSON, PNG, or evidence artifact output

There is no external database or API in the current baseline.

Planned Target Architecture

flowchart LR
    UI[Next.js Frontend] --> JAVA[Spring Boot Business API]
    JAVA --> DB[(PostgreSQL)]
    JAVA --> MQ[RabbitMQ]
    MQ --> PY[FastAPI Fraud Analytics Service]
    PY --> MODEL[Versioned ML Model]
    PY --> DB
    JAVA --> AUDIT[Audit and Case Services]

Planned responsibilities:

Next.js: dashboards, case review, evidence views, model explanations;

Spring Boot: users, roles, cases, audit, workflow, orchestration;

FastAPI: feature validation, scoring, inference, explanations;

PostgreSQL: persistent application, audit, model, and prediction data;

RabbitMQ: asynchronous scoring and investigation events;

Docker Compose: reproducible local deployment.

Architectural Risks

Current risks include:

synthetic and very small datasets;

in-memory and CSV-based processing;

no authentication or authorization;

no persistent database;

no production-grade evidence storage;

no encryption at rest;

no model registry;

no production monitoring;

no deployment automation.

These limitations are intentional at the current educational stage and must be resolved before any production use.