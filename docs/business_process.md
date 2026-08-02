Business Process

Purpose

This document explains the fraud-analysis and investigation process implemented by the project.

Process Overview

event
→ assessment
→ alert
→ prioritization
→ case
→ investigation
→ evidence
→ resolution
→ supervisor approval
→ closure
→ feedback
→ model dataset

A controlled reopening path may invalidate previous feedback:

closed case
→ supervisor-approved reopening
→ previous resolution superseded
→ previous feedback invalidated
→ investigation resumes

1. Event or Case Input

The process starts with synthetic case or event data.

Examples include:

suspicious transactions;

unusual authentication activity;

operational fraud indicators;

pressure or opportunity indicators;

delayed detection;

potential collusion.

Input data must be validated before scoring.

2. Assessment

An assessment evaluates available signals and calculates an educational prioritization score.

The assessment record represents analytical output. It is not yet an alert or investigation.

Important rule:

An assessment score does not establish fraud.

3. Alert Generation

An alert is created only when configured criteria justify operational review.

The alert contains:

source assessment;

risk level;

recommended action;

triggering signals;

creation timestamp;

alert status.

Assessment and alert records remain separate for traceability.

4. Prioritization

Alerts are ordered according to operational priority.

Prioritization may consider:

risk level;

amount involved;

signal severity;

available evidence;

investigation capacity;

age of the alert.

Capacity constraints determine whether an alert is selected immediately or placed in a waiting queue.

5. Case Creation and Assignment

A selected alert may become an investigation case.

The workflow records:

assigned analyst;

assigned supervisor;

investigation status;

created and updated timestamps;

audit events.

6. Investigation

The assigned analyst reviews the case and may:

inspect source data;

add investigation notes;

request additional information;

register evidence;

escalate the case;

prepare a resolution proposal.

7. Evidence

Evidence registration includes:

evidence identifier;

case identifier;

original filename;

controlled storage path;

file size;

SHA-256 hash;

registration timestamp;

actor;

chain-of-custody event.

A SHA-256 hash supports change detection. It does not independently establish authenticity or truthfulness.

8. Resolution Proposal

The assigned analyst may propose one of the permitted final outcomes:

CONFIRMADO

DESCARTADO

The proposal may include:

confirmed loss;

recovered amount;

rationale;

supporting references.

9. Supervisor Approval

Only the assigned supervisor may approve the resolution.

This segregation of duties prevents the analyst from unilaterally closing the case.

10. Closure

After approval, the workflow records:

resolution approval;

final investigation outcome;

transition to CERRADO;

audit events;

supervised feedback.

Historical information is preserved.

11. Feedback

Closed, approved cases generate a supervised label:

Investigation outcome

Label

DESCARTADO

0

CONFIRMADO

1

Feedback is valid only while its lifecycle status is active.

12. Controlled Reopening

A closed case may be reopened only by the assigned supervisor and with a documented reason.

Reopening:

preserves the previous resolution;

marks it as superseded;

invalidates the previous feedback;

returns the case to investigation;

records audit events.

13. Training Dataset

Only active, consistent feedback may enter the supervised training dataset.

Records are excluded when:

feedback is invalidated;

variables are missing;

labels are inconsistent;

multiple active labels exist;

feature snapshots occur after closure;

forbidden post-outcome variables are selected.

Roles

Analyst

Responsible for:

reviewing assigned cases;

documenting findings;

registering evidence;

proposing resolutions.

Supervisor

Accountable for:

approving resolutions;

controlling reopening;

reviewing exceptional decisions;

preserving segregation of duties.

Data owner or steward

Responsible for:

data definitions;

source quality;

lineage;

access and governance.

Model or analytics team

Responsible for:

feature preparation;

model development;

evaluation;

monitoring;

documentation.

Key Control Principles

no automatic presumption of guilt;

human review;

append-only audit events;

evidence integrity checks;

separation of duties;

controlled feedback lifecycle;

temporal leakage prevention;

synthetic-data transparency.