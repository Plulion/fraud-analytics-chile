Security and Privacy

Purpose

This document defines the current security and privacy expectations for Fraud Analytics Chile.

The current repository is educational and does not contain production security controls.

Data Classification

The public repository may contain only:

synthetic cases;

synthetic events;

fictitious organizations;

fictitious identifiers;

demonstration evidence;

reference classifications;

generated educational reports.

The repository must not contain:

real personal data;

real customer data;

real RUT values;

real account numbers;

real fraud investigations;

confidential evidence;

production credentials;

passwords;

API keys;

private certificates;

access tokens.

Secrets Management

Real secrets must never be committed.

The following files or patterns are excluded through .gitignore:

.env

.env.*

virtual environments;

IDE-local configuration;

caches;

generated outputs.

A future .env.example may document required variables using placeholder values only.

Evidence Handling

The current evidence store is a local educational implementation.

It provides:

controlled copying;

unique identifiers;

file-size validation;

SHA-256 hashing;

chain-of-custody metadata.

It does not yet provide:

encryption at rest;

immutable object storage;

legal evidence certification;

malware scanning;

key management;

retention enforcement;

secure deletion;

production access control.

Audit Trail

Audit records are designed to be append-only.

Relevant actions include:

assignment;

status changes;

resolution proposals;

approvals;

closure;

reopening;

feedback invalidation.

A production system should store audit records in a protected database or immutable log service.

Access Control

The current Python implementation identifies actors by synthetic IDs but does not authenticate them.

The planned architecture should implement role-based access control.

Suggested roles:

analyst;

supervisor;

data steward;

model developer;

auditor;

administrator.

The principle of least privilege should be applied.

Segregation of Duties

The current business logic separates:

analyst resolution proposal;

supervisor resolution approval;

supervisor-controlled reopening.

This control reduces unilateral decision risk.

Privacy Principles

A production system should follow:

data minimization;

purpose limitation;

access limitation;

retention control;

traceability;

secure deletion;

pseudonymization where appropriate;

documented lawful basis and governance.

Logging

Logs must not expose:

passwords;

tokens;

full personal identifiers;

confidential evidence contents;

private account data.

Identifiers should be masked or pseudonymized when possible.

Machine-Learning Security

Future model controls should include:

validated feature schemas;

dataset hashes;

versioned models;

access-controlled artifacts;

drift monitoring;

prediction logging;

rollback capability;

adversarial-input considerations;

human review of adverse outcomes.

Public Repository Review

Before each public push, verify:

git status

Search for likely secret terms:

grep -RniE \
'password|passwd|secret|token|api[_-]?key|authorization|private[_-]?key' \
README.md docs data scripts src tests \
2>/dev/null

Search results must be reviewed manually because legitimate field names may also match.

Current Limitations

The project currently lacks:

authentication;

authorization;

encrypted persistence;

production secret storage;

secure database;

TLS configuration;

threat modeling;

dependency vulnerability scanning;

security testing;

production incident response.

These limitations are explicitly documented and must be addressed before any real deployment.