from __future__ import annotations

import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


CHILE_TIME_ZONE = ZoneInfo(
    "America/Santiago"
)


VALID_NOTE_TYPES = {
    "OBSERVACION",
    "ANALISIS",
    "SOLICITUD",
    "ENTREVISTA",
    "DECISION_SUPPORT",
}


VALID_EVIDENCE_STATUSES = {
    "REGISTRADA",
    "PRESERVADA",
    "EN_REVISION",
    "ARCHIVADA",
}


VALID_CUSTODY_ACTIONS = {
    "REGISTERED",
    "HASH_VERIFIED",
    "ACCESSED",
    "STATUS_CHANGED",
    "ARCHIVED",
    "INTEGRITY_MISMATCH",
}


NOTE_COLUMNS = [
    "note_id",
    "case_id",
    "created_at",
    "author_id",
    "note_type",
    "content",
    "is_private",
    "previous_note_id",
]


EVIDENCE_COLUMNS = [
    "evidence_id",
    "case_id",
    "original_filename",
    "stored_filename",
    "storage_path",
    "source_description",
    "collected_by",
    "collected_at",
    "sha256",
    "size_bytes",
    "content_type",
    "evidence_status",
]


CUSTODY_COLUMNS = [
    "custody_id",
    "evidence_id",
    "case_id",
    "event_timestamp",
    "actor_id",
    "action_type",
    "location_reference",
    "comment",
    "sha256",
]


@dataclass(frozen=True)
class EvidenceRegistrationResult:
    evidence: pd.DataFrame
    custody: pd.DataFrame
    stored_path: Path


@dataclass(frozen=True)
class IntegrityVerificationResult:
    custody: pd.DataFrame
    evidence_id: str
    expected_hash: str
    current_hash: str
    integrity_ok: bool


def current_chile_timestamp() -> str:
    return datetime.now(
        CHILE_TIME_ZONE
    ).isoformat(
        timespec="seconds"
    )


def normalize_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_upper_text(
    value: Any,
) -> str:
    return normalize_text(
        value
    ).upper()


def create_empty_notes() -> pd.DataFrame:
    return pd.DataFrame(
        columns=NOTE_COLUMNS
    )


def create_empty_evidence_registry() -> pd.DataFrame:
    return pd.DataFrame(
        columns=EVIDENCE_COLUMNS
    )


def create_empty_custody_log() -> pd.DataFrame:
    return pd.DataFrame(
        columns=CUSTODY_COLUMNS
    )


def next_sequential_id(
    df: pd.DataFrame,
    *,
    column: str,
    prefix: str,
) -> str:
    if df.empty:
        return f"{prefix}-000001"

    if column not in df.columns:
        raise ValueError(
            f"No existe la columna {column!r}."
        )

    maximum = 0

    for value in df[column]:
        text = normalize_text(
            value
        )

        expected_prefix = (
            f"{prefix}-"
        )

        if not text.startswith(
            expected_prefix
        ):
            continue

        numeric_part = text[
            len(expected_prefix):
        ]

        if numeric_part.isdigit():
            maximum = max(
                maximum,
                int(numeric_part),
            )

    return (
        f"{prefix}-{maximum + 1:06d}"
    )


def validate_case_available(
    cases_df: pd.DataFrame,
    case_id: str,
) -> pd.Series:
    """
    Comprueba que el caso exista y que no esté cerrado.

    Acepta distintos nombres de columna para el estado
    del caso, incluyendo investigation_status.
    """

    if "case_id" not in cases_df.columns:
        raise ValueError(
            "La tabla de casos no contiene "
            "la columna requerida 'case_id'."
        )

    status_candidates = [
        "investigation_status",
        "case_status",
        "workflow_status",
        "current_status",
        "status",
    ]

    status_column = next(
        (
            column
            for column in status_candidates
            if column in cases_df.columns
        ),
        None,
    )

    if status_column is None:
        raise ValueError(
            "La tabla de casos no contiene una "
            "columna reconocida para el estado. "
            "Se esperaba alguna de estas: "
            f"{status_candidates}. "
            "Columnas encontradas: "
            f"{cases_df.columns.tolist()}"
        )

    normalized_case_id = normalize_text(
        case_id
    )

    matches = cases_df.loc[
        cases_df["case_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        == normalized_case_id
    ]

    if matches.empty:
        raise ValueError(
            "No existe el caso: "
            f"{normalized_case_id}"
        )

    if len(matches) > 1:
        raise ValueError(
            "El caso aparece más de una vez: "
            f"{normalized_case_id}"
        )

    case_row = matches.iloc[0]

    case_status = normalize_upper_text(
        case_row[status_column]
    )

    if not case_status:
        raise ValueError(
            "El caso no tiene un estado válido: "
            f"{normalized_case_id}"
        )

    if case_status == "CERRADO":
        raise ValueError(
            "No se pueden agregar registros "
            "a un caso cerrado."
        )

    return case_row


def add_investigation_note(
    cases_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    *,
    case_id: str,
    author_id: str,
    note_type: str,
    content: str,
    is_private: bool = False,
    previous_note_id: str = "",
    created_at: str | None = None,
) -> pd.DataFrame:
    validate_case_available(
        cases_df,
        case_id,
    )

    normalized_author = normalize_text(
        author_id
    )

    normalized_type = normalize_upper_text(
        note_type
    )

    normalized_content = normalize_text(
        content
    )

    normalized_previous = normalize_text(
        previous_note_id
    )

    if not normalized_author:
        raise ValueError(
            "author_id es obligatorio."
        )

    if normalized_type not in (
        VALID_NOTE_TYPES
    ):
        raise ValueError(
            "Tipo de nota inválido: "
            f"{note_type!r}"
        )

    if not normalized_content:
        raise ValueError(
            "La nota no puede estar vacía."
        )

    if normalized_previous:
        if notes_df.empty:
            raise ValueError(
                "No existe la nota anterior."
            )

        previous_matches = notes_df.loc[
            notes_df["note_id"]
            .astype(str)
            == normalized_previous
        ]

        if previous_matches.empty:
            raise ValueError(
                "No existe la nota anterior: "
                f"{normalized_previous}"
            )

        previous_case_id = normalize_text(
            previous_matches.iloc[0][
                "case_id"
            ]
        )

        if previous_case_id != normalize_text(
            case_id
        ):
            raise ValueError(
                "La nota anterior pertenece "
                "a otro caso."
            )

    note_id = next_sequential_id(
        notes_df,
        column="note_id",
        prefix="NOTE",
    )

    row = {
        "note_id": note_id,
        "case_id": normalize_text(
            case_id
        ),
        "created_at": (
            created_at
            if created_at is not None
            else current_chile_timestamp()
        ),
        "author_id": normalized_author,
        "note_type": normalized_type,
        "content": normalized_content,
        "is_private": bool(
            is_private
        ),
        "previous_note_id": (
            normalized_previous
        ),
    }

    return pd.concat(
        [
            notes_df,
            pd.DataFrame(
                [row],
                columns=NOTE_COLUMNS,
            ),
        ],
        ignore_index=True,
    )


def hash_file_sha256(
    file_path: Path,
) -> str:
    path = Path(
        file_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"La ruta no es un archivo: {path}"
        )

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def append_custody_event(
    custody_df: pd.DataFrame,
    *,
    evidence_id: str,
    case_id: str,
    actor_id: str,
    action_type: str,
    location_reference: str,
    comment: str,
    sha256: str,
    event_timestamp: str | None = None,
) -> pd.DataFrame:
    normalized_action = (
        normalize_upper_text(
            action_type
        )
    )

    if normalized_action not in (
        VALID_CUSTODY_ACTIONS
    ):
        raise ValueError(
            "Acción de custodia inválida: "
            f"{action_type!r}"
        )

    normalized_actor = normalize_text(
        actor_id
    )

    if not normalized_actor:
        raise ValueError(
            "actor_id es obligatorio."
        )

    custody_id = next_sequential_id(
        custody_df,
        column="custody_id",
        prefix="CUST",
    )

    row = {
        "custody_id": custody_id,
        "evidence_id": normalize_text(
            evidence_id
        ),
        "case_id": normalize_text(
            case_id
        ),
        "event_timestamp": (
            event_timestamp
            if event_timestamp is not None
            else current_chile_timestamp()
        ),
        "actor_id": normalized_actor,
        "action_type": normalized_action,
        "location_reference": (
            normalize_text(
                location_reference
            )
        ),
        "comment": normalize_text(
            comment
        ),
        "sha256": normalize_text(
            sha256
        ),
    }

    return pd.concat(
        [
            custody_df,
            pd.DataFrame(
                [row],
                columns=CUSTODY_COLUMNS,
            ),
        ],
        ignore_index=True,
    )


def register_evidence(
    cases_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    custody_df: pd.DataFrame,
    *,
    case_id: str,
    source_path: Path,
    storage_root: Path,
    source_description: str,
    collected_by: str,
    collected_at: str | None = None,
) -> EvidenceRegistrationResult:
    validate_case_available(
        cases_df,
        case_id,
    )

    source = Path(
        source_path
    )

    if not source.exists():
        raise FileNotFoundError(
            f"No existe la evidencia: {source}"
        )

    if not source.is_file():
        raise ValueError(
            "La evidencia debe ser un archivo."
        )

    if source.stat().st_size == 0:
        raise ValueError(
            "La evidencia no puede ser un archivo vacío."
        )

    normalized_collector = normalize_text(
        collected_by
    )

    if not normalized_collector:
        raise ValueError(
            "collected_by es obligatorio."
        )

    normalized_description = normalize_text(
        source_description
    )

    if not normalized_description:
        raise ValueError(
            "source_description es obligatorio."
        )

    evidence_id = next_sequential_id(
        evidence_df,
        column="evidence_id",
        prefix="EVD",
    )

    evidence_directory = (
        Path(storage_root)
        / normalize_text(case_id)
        / evidence_id
    )

    evidence_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    original_filename = (
        source.name
    )

    stored_filename = (
        original_filename
    )

    stored_path = (
        evidence_directory
        / stored_filename
    )

    source_hash = hash_file_sha256(
        source
    )

    shutil.copy2(
        source,
        stored_path,
    )

    stored_hash = hash_file_sha256(
        stored_path
    )

    if source_hash != stored_hash:
        shutil.rmtree(
            evidence_directory,
            ignore_errors=True,
        )

        raise RuntimeError(
            "El hash de la copia no coincide "
            "con el archivo de origen."
        )

    content_type = (
        mimetypes.guess_type(
            original_filename
        )[0]
        or "application/octet-stream"
    )

    timestamp = (
        collected_at
        if collected_at is not None
        else current_chile_timestamp()
    )

    evidence_row = {
        "evidence_id": evidence_id,
        "case_id": normalize_text(
            case_id
        ),
        "original_filename": (
            original_filename
        ),
        "stored_filename": (
            stored_filename
        ),
        "storage_path": str(
            stored_path
        ),
        "source_description": (
            normalized_description
        ),
        "collected_by": (
            normalized_collector
        ),
        "collected_at": timestamp,
        "sha256": stored_hash,
        "size_bytes": int(
            stored_path.stat().st_size
        ),
        "content_type": (
            content_type
        ),
        "evidence_status": (
            "PRESERVADA"
        ),
    }

    updated_evidence = pd.concat(
        [
            evidence_df,
            pd.DataFrame(
                [evidence_row],
                columns=EVIDENCE_COLUMNS,
            ),
        ],
        ignore_index=True,
    )

    updated_custody = (
        append_custody_event(
            custody_df,
            evidence_id=evidence_id,
            case_id=case_id,
            actor_id=normalized_collector,
            action_type="REGISTERED",
            location_reference=str(
                stored_path
            ),
            comment=(
                "Evidencia registrada "
                "desde el archivo de origen."
            ),
            sha256=source_hash,
            event_timestamp=timestamp,
        )
    )

    updated_custody = (
        append_custody_event(
            updated_custody,
            evidence_id=evidence_id,
            case_id=case_id,
            actor_id="SYSTEM",
            action_type="HASH_VERIFIED",
            location_reference=str(
                stored_path
            ),
            comment=(
                "El hash SHA-256 de la copia "
                "coincide con el origen."
            ),
            sha256=stored_hash,
            event_timestamp=timestamp,
        )
    )

    return EvidenceRegistrationResult(
        evidence=updated_evidence,
        custody=updated_custody,
        stored_path=stored_path,
    )


def verify_evidence_integrity(
    evidence_df: pd.DataFrame,
    custody_df: pd.DataFrame,
    *,
    evidence_id: str,
    actor_id: str,
    comment: str = "",
    event_timestamp: str | None = None,
) -> IntegrityVerificationResult:
    matches = evidence_df.loc[
        evidence_df["evidence_id"]
        .astype(str)
        == normalize_text(
            evidence_id
        )
    ]

    if matches.empty:
        raise ValueError(
            "No existe la evidencia: "
            f"{evidence_id}"
        )

    if len(matches) > 1:
        raise ValueError(
            "El identificador de evidencia "
            "está duplicado."
        )

    evidence_row = matches.iloc[0]

    stored_path = Path(
        normalize_text(
            evidence_row[
                "storage_path"
            ]
        )
    )

    expected_hash = normalize_text(
        evidence_row["sha256"]
    )

    current_hash = hash_file_sha256(
        stored_path
    )

    integrity_ok = (
        current_hash
        == expected_hash
    )

    action_type = (
        "HASH_VERIFIED"
        if integrity_ok
        else "INTEGRITY_MISMATCH"
    )

    default_comment = (
        "La integridad de la evidencia "
        "fue verificada."
        if integrity_ok
        else
        "El hash actual no coincide "
        "con el hash registrado."
    )

    updated_custody = (
        append_custody_event(
            custody_df,
            evidence_id=(
                evidence_row[
                    "evidence_id"
                ]
            ),
            case_id=(
                evidence_row[
                    "case_id"
                ]
            ),
            actor_id=actor_id,
            action_type=action_type,
            location_reference=str(
                stored_path
            ),
            comment=(
                normalize_text(comment)
                or default_comment
            ),
            sha256=current_hash,
            event_timestamp=(
                event_timestamp
            ),
        )
    )

    return IntegrityVerificationResult(
        custody=updated_custody,
        evidence_id=normalize_text(
            evidence_id
        ),
        expected_hash=expected_hash,
        current_hash=current_hash,
        integrity_ok=integrity_ok,
    )