from pathlib import Path

import pandas as pd
import pytest

from src.investigation_records import (
    add_investigation_note,
    create_empty_custody_log,
    create_empty_evidence_registry,
    create_empty_notes,
    hash_file_sha256,
    register_evidence,
    verify_evidence_integrity,
)


def build_cases_dataframe() -> pd.DataFrame:
    """
    Construye casos sintéticos para las pruebas.

    CL-001 está disponible para investigación.
    CL-002 está cerrado y no debe aceptar
    nuevas notas ni evidencias.
    """

    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
            ],
            "case_status": [
                "EN_INVESTIGACION",
                "CERRADO",
            ],
        }
    )


def test_add_investigation_note() -> None:
    notes_df = add_investigation_note(
        build_cases_dataframe(),
        create_empty_notes(),
        case_id="CL-001",
        author_id="ANA-001",
        note_type="OBSERVACION",
        content="Se revisaron las operaciones.",
        created_at=(
            "2026-08-01T20:00:00-04:00"
        ),
    )

    assert len(notes_df) == 1

    assert (
        notes_df.iloc[0]["note_id"]
        == "NOTE-000001"
    )

    assert (
        notes_df.iloc[0]["case_id"]
        == "CL-001"
    )

    assert (
        notes_df.iloc[0]["author_id"]
        == "ANA-001"
    )

    assert (
        notes_df.iloc[0]["note_type"]
        == "OBSERVACION"
    )


def test_empty_note_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="no puede estar vacía",
    ):
        add_investigation_note(
            build_cases_dataframe(),
            create_empty_notes(),
            case_id="CL-001",
            author_id="ANA-001",
            note_type="OBSERVACION",
            content="",
        )


def test_invalid_note_type_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Tipo de nota inválido",
    ):
        add_investigation_note(
            build_cases_dataframe(),
            create_empty_notes(),
            case_id="CL-001",
            author_id="ANA-001",
            note_type="TIPO_INEXISTENTE",
            content="Contenido válido.",
        )


def test_note_correction_references_previous_note() -> None:
    notes_df = add_investigation_note(
        build_cases_dataframe(),
        create_empty_notes(),
        case_id="CL-001",
        author_id="ANA-001",
        note_type="OBSERVACION",
        content="Nota original.",
    )

    notes_df = add_investigation_note(
        build_cases_dataframe(),
        notes_df,
        case_id="CL-001",
        author_id="ANA-001",
        note_type="ANALISIS",
        content="Nota corregida.",
        previous_note_id="NOTE-000001",
    )

    assert len(notes_df) == 2

    assert (
        notes_df.iloc[1]["note_id"]
        == "NOTE-000002"
    )

    assert (
        notes_df.iloc[1][
            "previous_note_id"
        ]
        == "NOTE-000001"
    )


def test_note_correction_rejects_unknown_previous_note() -> None:
    with pytest.raises(
        ValueError,
        match="No existe la nota anterior",
    ):
        add_investigation_note(
            build_cases_dataframe(),
            create_empty_notes(),
            case_id="CL-001",
            author_id="ANA-001",
            note_type="ANALISIS",
            content="Corrección sin nota original.",
            previous_note_id="NOTE-999999",
        )


def test_hash_file_sha256_is_reproducible(
    tmp_path: Path,
) -> None:
    file_path = (
        tmp_path
        / "sample.txt"
    )

    file_path.write_text(
        "contenido de prueba",
        encoding="utf-8",
    )

    first_hash = hash_file_sha256(
        file_path
    )

    second_hash = hash_file_sha256(
        file_path
    )

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_hash_changes_when_file_content_changes(
    tmp_path: Path,
) -> None:
    file_path = (
        tmp_path
        / "sample.txt"
    )

    file_path.write_text(
        "contenido original",
        encoding="utf-8",
    )

    original_hash = hash_file_sha256(
        file_path
    )

    file_path.write_text(
        "contenido modificado",
        encoding="utf-8",
    )

    modified_hash = hash_file_sha256(
        file_path
    )

    assert original_hash != modified_hash


def test_register_evidence_copies_and_hashes(
    tmp_path: Path,
) -> None:
    source_file = (
        tmp_path
        / "evidence.txt"
    )

    source_file.write_text(
        "evidencia sintética",
        encoding="utf-8",
    )

    result = register_evidence(
        build_cases_dataframe(),
        create_empty_evidence_registry(),
        create_empty_custody_log(),
        case_id="CL-001",
        source_path=source_file,
        storage_root=(
            tmp_path
            / "store"
        ),
        source_description=(
            "Archivo de prueba."
        ),
        collected_by="ANA-001",
        collected_at=(
            "2026-08-01T20:00:00-04:00"
        ),
    )

    assert result.stored_path.exists()
    assert result.stored_path.is_file()

    assert len(result.evidence) == 1
    assert len(result.custody) == 2

    evidence_row = result.evidence.iloc[0]

    assert (
        evidence_row["evidence_id"]
        == "EVD-000001"
    )

    assert (
        evidence_row["case_id"]
        == "CL-001"
    )

    assert (
        evidence_row["evidence_status"]
        == "PRESERVADA"
    )

    assert (
        int(evidence_row["size_bytes"])
        > 0
    )

    assert (
        evidence_row["sha256"]
        == hash_file_sha256(
            result.stored_path
        )
    )

    assert (
        result.custody.iloc[0][
            "action_type"
        ]
        == "REGISTERED"
    )

    assert (
        result.custody.iloc[1][
            "action_type"
        ]
        == "HASH_VERIFIED"
    )


def test_verify_evidence_integrity_when_unchanged(
    tmp_path: Path,
) -> None:
    source_file = (
        tmp_path
        / "evidence.txt"
    )

    source_file.write_text(
        "contenido original",
        encoding="utf-8",
    )

    registration = register_evidence(
        build_cases_dataframe(),
        create_empty_evidence_registry(),
        create_empty_custody_log(),
        case_id="CL-001",
        source_path=source_file,
        storage_root=(
            tmp_path
            / "store"
        ),
        source_description=(
            "Archivo de prueba."
        ),
        collected_by="ANA-001",
    )

    evidence_id = (
        registration.evidence.iloc[0][
            "evidence_id"
        ]
    )

    verification = verify_evidence_integrity(
        registration.evidence,
        registration.custody,
        evidence_id=evidence_id,
        actor_id="SUP-001",
    )

    assert verification.integrity_ok

    assert (
        verification.expected_hash
        == verification.current_hash
    )

    assert (
        verification.custody.iloc[-1][
            "action_type"
        ]
        == "HASH_VERIFIED"
    )


def test_verify_evidence_detects_change(
    tmp_path: Path,
) -> None:
    source_file = (
        tmp_path
        / "evidence.txt"
    )

    source_file.write_text(
        "contenido original",
        encoding="utf-8",
    )

    registration = register_evidence(
        build_cases_dataframe(),
        create_empty_evidence_registry(),
        create_empty_custody_log(),
        case_id="CL-001",
        source_path=source_file,
        storage_root=(
            tmp_path
            / "store"
        ),
        source_description=(
            "Archivo de prueba."
        ),
        collected_by="ANA-001",
    )

    registration.stored_path.write_text(
        "contenido modificado",
        encoding="utf-8",
    )

    evidence_id = (
        registration.evidence.iloc[0][
            "evidence_id"
        ]
    )

    verification = verify_evidence_integrity(
        registration.evidence,
        registration.custody,
        evidence_id=evidence_id,
        actor_id="SUP-001",
    )

    assert not verification.integrity_ok

    assert (
        verification.expected_hash
        != verification.current_hash
    )

    assert (
        verification.custody.iloc[-1][
            "action_type"
        ]
        == "INTEGRITY_MISMATCH"
    )


def test_closed_case_rejects_new_evidence(
    tmp_path: Path,
) -> None:
    source_file = (
        tmp_path
        / "evidence.txt"
    )

    source_file.write_text(
        "evidencia",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="caso cerrado",
    ):
        register_evidence(
            build_cases_dataframe(),
            create_empty_evidence_registry(),
            create_empty_custody_log(),
            case_id="CL-002",
            source_path=source_file,
            storage_root=(
                tmp_path
                / "store"
            ),
            source_description=(
                "Archivo de prueba."
            ),
            collected_by="ANA-001",
        )


def test_empty_evidence_file_is_rejected(
    tmp_path: Path,
) -> None:
    source_file = (
        tmp_path
        / "empty.txt"
    )

    source_file.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="archivo vacío",
    ):
        register_evidence(
            build_cases_dataframe(),
            create_empty_evidence_registry(),
            create_empty_custody_log(),
            case_id="CL-001",
            source_path=source_file,
            storage_root=(
                tmp_path
                / "store"
            ),
            source_description=(
                "Archivo vacío de prueba."
            ),
            collected_by="ANA-001",
        )


def test_nonexistent_evidence_file_is_rejected(
    tmp_path: Path,
) -> None:
    nonexistent_file = (
        tmp_path
        / "does_not_exist.txt"
    )

    with pytest.raises(
        FileNotFoundError,
        match="No existe la evidencia",
    ):
        register_evidence(
            build_cases_dataframe(),
            create_empty_evidence_registry(),
            create_empty_custody_log(),
            case_id="CL-001",
            source_path=nonexistent_file,
            storage_root=(
                tmp_path
                / "store"
            ),
            source_description=(
                "Archivo inexistente."
            ),
            collected_by="ANA-001",
        )


def test_accepts_status_column_alias() -> None:
    cases_df = pd.DataFrame(
        {
            "case_id": [
                "CL-001",
            ],
            "status": [
                "EN_INVESTIGACION",
            ],
        }
    )

    notes_df = add_investigation_note(
        cases_df,
        create_empty_notes(),
        case_id="CL-001",
        author_id="ANA-001",
        note_type="OBSERVACION",
        content=(
            "Nota compatible con columna status."
        ),
    )

    assert len(notes_df) == 1

    assert (
        notes_df.iloc[0]["case_id"]
        == "CL-001"
    )


def test_accepts_investigation_status_column() -> None:
    cases_df = pd.DataFrame(
        {
            "case_id": [
                "CL-001",
            ],
            "investigation_status": [
                "EN_INVESTIGACION",
            ],
        }
    )

    notes_df = add_investigation_note(
        cases_df,
        create_empty_notes(),
        case_id="CL-001",
        author_id="ANA-001",
        note_type="OBSERVACION",
        content=(
            "Nota compatible con "
            "investigation_status."
        ),
    )

    assert len(notes_df) == 1


def test_unknown_case_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="No existe el caso",
    ):
        add_investigation_note(
            build_cases_dataframe(),
            create_empty_notes(),
            case_id="CL-999",
            author_id="ANA-001",
            note_type="OBSERVACION",
            content="Caso inexistente.",
        )