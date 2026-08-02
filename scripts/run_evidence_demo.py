from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.investigation_records import (
    add_investigation_note,
    create_empty_custody_log,
    create_empty_evidence_registry,
    create_empty_notes,
    register_evidence,
    verify_evidence_integrity,
)


CASES_INPUT = Path(
    "outputs/gestion_casos.csv"
)

SAMPLE_EVIDENCE = Path(
    "data/evidence_samples/"
    "transacciones_CL002.txt"
)

NOTES_OUTPUT = Path(
    "outputs/notas_investigacion.csv"
)

EVIDENCE_OUTPUT = Path(
    "outputs/evidencias_casos.csv"
)

CUSTODY_OUTPUT = Path(
    "outputs/cadena_custodia.csv"
)

STORAGE_ROOT = Path(
    "outputs/evidence_store"
)


def main() -> None:
    if not CASES_INPUT.exists():
        raise FileNotFoundError(
            "No existe outputs/gestion_casos.csv. "
            "Ejecute primero el demo del workflow."
        )

    if not SAMPLE_EVIDENCE.exists():
        raise FileNotFoundError(
            "No existe la evidencia sintética: "
            f"{SAMPLE_EVIDENCE}"
        )

    cases_df = pd.read_csv(
        CASES_INPUT
    )

    if STORAGE_ROOT.exists():
        shutil.rmtree(
            STORAGE_ROOT
        )

    notes_df = create_empty_notes()
    evidence_df = (
        create_empty_evidence_registry()
    )
    custody_df = (
        create_empty_custody_log()
    )

    target_case_id = "CL-002"

    notes_df = add_investigation_note(
        cases_df,
        notes_df,
        case_id=target_case_id,
        author_id="ANA-001",
        note_type="OBSERVACION",
        content=(
            "Se identificaron tres operaciones "
            "dentro de un intervalo breve."
        ),
        created_at=(
            "2026-08-01T21:10:00-04:00"
        ),
    )

    original_note_id = (
        notes_df.iloc[0]["note_id"]
    )

    notes_df = add_investigation_note(
        cases_df,
        notes_df,
        case_id=target_case_id,
        author_id="ANA-001",
        note_type="ANALISIS",
        content=(
            "Corrección: las operaciones deben "
            "ser contrastadas con el comportamiento "
            "histórico antes de concluir."
        ),
        previous_note_id=(
            original_note_id
        ),
        created_at=(
            "2026-08-01T21:12:00-04:00"
        ),
    )

    registration = register_evidence(
        cases_df,
        evidence_df,
        custody_df,
        case_id=target_case_id,
        source_path=SAMPLE_EVIDENCE,
        storage_root=STORAGE_ROOT,
        source_description=(
            "Extracto sintético preparado para "
            "el laboratorio de investigación."
        ),
        collected_by="ANA-001",
        collected_at=(
            "2026-08-01T21:15:00-04:00"
        ),
    )

    evidence_df = registration.evidence
    custody_df = registration.custody

    evidence_id = (
        evidence_df.iloc[0][
            "evidence_id"
        ]
    )

    verification = (
        verify_evidence_integrity(
            evidence_df,
            custody_df,
            evidence_id=evidence_id,
            actor_id="SUP-001",
            comment=(
                "Verificación previa a revisión "
                "del supervisor."
            ),
            event_timestamp=(
                "2026-08-01T21:20:00-04:00"
            ),
        )
    )

    custody_df = verification.custody

    NOTES_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    notes_df.to_csv(
        NOTES_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    evidence_df.to_csv(
        EVIDENCE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    custody_df.to_csv(
        CUSTODY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nNOTAS DE INVESTIGACIÓN\n"
    )

    print(
        notes_df[
            [
                "note_id",
                "case_id",
                "author_id",
                "note_type",
                "previous_note_id",
                "content",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nEVIDENCIAS\n"
    )

    evidence_display = evidence_df[
        [
            "evidence_id",
            "case_id",
            "original_filename",
            "evidence_status",
            "size_bytes",
            "sha256",
        ]
    ].copy()

    evidence_display["sha256"] = (
        evidence_display[
            "sha256"
        ].str.slice(
            0,
            16,
        )
        + "..."
    )

    print(
        evidence_display.to_string(
            index=False
        )
    )

    print(
        "\nCADENA DE CUSTODIA\n"
    )

    print(
        custody_df[
            [
                "custody_id",
                "evidence_id",
                "actor_id",
                "action_type",
                "event_timestamp",
                "comment",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nVERIFICACIÓN DE INTEGRIDAD\n"
    )

    print(
        "Integridad correcta: "
        f"{verification.integrity_ok}"
    )

    print(
        "Hash esperado: "
        f"{verification.expected_hash}"
    )

    print(
        "Hash actual:   "
        f"{verification.current_hash}"
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        f"- {NOTES_OUTPUT.resolve()}"
    )

    print(
        f"- {EVIDENCE_OUTPUT.resolve()}"
    )

    print(
        f"- {CUSTODY_OUTPUT.resolve()}"
    )

    print(
        f"- {registration.stored_path.resolve()}"
    )


if __name__ == "__main__":
    main()