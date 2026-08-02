Data Dictionary

Purpose

This document describes the most important fields currently used by Fraud Analytics Chile.

The repository contains multiple CSV outputs. Not every field appears in every table.

Core Identifiers

Field

Type

Description

Typical source

case_id

string

Unique synthetic investigation-case identifier

Case input or workflow

event_id

string

Unique synthetic digital-event identifier

Digital event input

customer_id

string

Synthetic customer identifier

Digital event input

assessment_id

string

Unique professional assessment identifier

Professional pipeline

alert_id

string

Unique alert identifier

Alert pipeline

resolution_id

string

Unique case-resolution identifier

Case closure

audit_id

string

Unique audit-event identifier

Workflow

evidence_id

string

Unique evidence identifier

Investigation records

Risk Assessment Fields

Field

Type

Description

opportunity_score

float

Score for opportunity-related indicators

pressure_score

float

Score for pressure-related indicators

rationalization_score

float

Score for rationalization-related indicators

aggravating_score

float

Score for aggravating factors such as collusion

risk_score

float

Total educational prioritization score

assessment_coverage

float

Percentage of expected assessment data available

risk_level

string

Assigned risk category

recommended_action

string

Suggested operational review action

detection_timeliness

string

Educational detection-timing category

Digital Fraud Fields

Field

Type

Description

event_timestamp

datetime

Timestamp of the digital event

channel

string

Event channel such as web or mobile

new_device

integer

Indicates whether a new device was observed

unusual_ip

integer

Indicates unusual network activity

geolocation_mismatch

integer

Indicates a location inconsistency

failed_mfa_attempts

integer

Number of failed MFA attempts

recent_password_reset

integer

Indicates a recent password reset

new_beneficiary

integer

Indicates a newly added beneficiary

amount_spike_ratio

float

Ratio of amount to expected historical level

rapid_transaction_count_10m

integer

Transactions observed in ten minutes

digital_risk_score

float

Total digital risk score

digital_alert_level

string

Digital alert category

digital_triggered_signals

string

Delimited list of activated digital signals

Alert and Prioritization Fields

Field

Type

Description

alert_status

string

Current operational alert status

priority_score

float

Educational operational-priority score

investigation_priority

string

Priority category assigned to the alert

selection_status

string

Indicates whether the alert was selected or is waiting

selection_reason

string

Explanation for selection or deferral

Workflow Fields

Field

Type

Description

investigation_status

string

Current case workflow status

assigned_analyst_id

string

Identifier of the assigned analyst

assigned_supervisor_id

string

Identifier of the assigned supervisor

created_at

datetime

Case creation timestamp

updated_at

datetime

Last case update timestamp

closed_at

datetime

Effective closure timestamp

reopened_at

datetime

Effective reopening timestamp

Typical workflow statuses include:

NUEVO

ASIGNADO

EN_INVESTIGACION

REQUIERE_ANTECEDENTES

ESCALADO

CONFIRMADO

DESCARTADO

CERRADO

Audit Fields

Field

Type

Description

audit_id

string

Unique audit-event identifier

case_id

string

Related case

event_timestamp

datetime

Audit-event timestamp

actor_id

string

Actor who performed the action

action_type

string

Type of audited action

field_name

string

Field affected by the action

previous_value

string

Previous field value

new_value

string

New field value

comment

string

Human-readable audit context

Evidence Fields

Field

Type

Description

evidence_id

string

Unique evidence identifier

case_id

string

Related case

original_filename

string

Original evidence filename

stored_path

string

Controlled local storage path

file_size_bytes

integer

File size in bytes

sha256_hash

string

SHA-256 digest used for change detection

registered_at

datetime

Evidence registration timestamp

registered_by

string

Actor who registered the evidence

integrity_status

string

Result of later integrity verification

Resolution and Feedback Fields

Field

Type

Description

investigation_outcome

string

Final approved outcome

outcome_label

integer

Supervised label: 0 or 1

resolution_status

string

Resolution approval status

resolution_lifecycle_status

string

Current validity of the resolution

feedback_status

string

Current validity of the supervised label

feedback_source

string

Origin of the feedback record

confirmed_loss_clp

float

Confirmed loss in Chilean pesos

recovered_amount_clp

float

Recovered amount in Chilean pesos

invalidated_at

datetime

Feedback invalidation timestamp

Allowed supervised outcomes:

Outcome

Label

DESCARTADO

0

CONFIRMADO

1

Machine-Learning Preparation Fields

Field

Type

Description

feature_snapshot_at

datetime

Time when predictor values were captured

dataset_partition

string

Future train, validation, or test assignment

exclusion_reason_code

string

Machine-readable exclusion reason

exclusion_reason_detail

string

Human-readable exclusion explanation

Important Data Rules

case_id must be unique in one-row-per-case datasets.

outcome_label must contain only 0 or 1.

CONFIRMADO must map to 1.

DESCARTADO must map to 0.

invalidated feedback must not be used for training.

feature_snapshot_at must not be later than closed_at.

identifiers may be retained as metadata but should not automatically be used as model predictors.

post-investigation fields must not be selected as model features.

all public repository data must remain synthetic.