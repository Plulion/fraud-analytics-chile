from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Iterable
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd


CHILE_TIME_ZONE = ZoneInfo(
    "America/Santiago"
)


REQUIRED_CONSOLIDATED_COLUMNS = {
    "case_id",
    "risk_score",
    "assessment_coverage",
    "alert_level",
    "decision_status",
    "consolidated_alert_level",
    "consolidated_data_status",
    "consolidated_recommended_action",
    "estimated_loss_clp",
}


VALID_ALERT_LEVELS = {
    "BAJO",
    "MEDIO",
    "ALTO",
    "CRITICO",
    "INCOMPLETO",
    "SIN_DATOS",
}


VALID_ALERT_PRIORITIES = {
    "BAJA",
    "MEDIA",
    "ALTA",
    "CRITICA",
}


@dataclass(frozen=True)
class AssessmentRecord:
    """
    Resultado reproducible de un motor analítico.

    No representa culpabilidad ni una decisión
    judicial o administrativa.
    """

    assessment_id: str
    case_id: str
    engine_type: str
    engine_version: str
    policy_id: str
    policy_version: str
    input_schema_version: str
    evaluated_at: str
    risk_score: float
    assessment_coverage: float
    calculated_alert_level: str
    recommended_action: str
    data_quality_status: str


@dataclass(frozen=True)
class AlertRecord:
    """
    Registro creado cuando una evaluación cumple
    una política de selección.
    """

    alert_id: str
    assessment_id: str
    case_id: str
    selection_policy_id: str
    selection_policy_version: str
    created_at: str
    alert_status: str
    alert_priority: str
    alert_reason: str
    risk_level: str
    estimated_loss_clp: float


@dataclass(frozen=True)
class AlertSelectionDecision:
    """
    Resultado interno de la política de selección.
    """

    should_create_alert: bool
    alert_priority: str | None
    alert_reason: str


@dataclass(frozen=True)
class ProfessionalPipelineResult:
    """
    Contiene las evaluaciones y alertas producidas.
    """

    assessments: pd.DataFrame
    alerts: pd.DataFrame


def current_chile_timestamp() -> str:
    """
    Devuelve fecha y hora ISO 8601 para Chile.
    """

    return datetime.now(
        CHILE_TIME_ZONE
    ).isoformat(
        timespec="seconds"
    )


def generate_assessment_id() -> str:
    """
    Genera un identificador técnico único.
    """

    return (
        "ASM-"
        f"{uuid4().hex.upper()}"
    )


def generate_alert_id() -> str:
    """
    Genera un identificador único de alerta.
    """

    return (
        "ALT-"
        f"{uuid4().hex.upper()}"
    )


def normalize_text(
    value: Any,
) -> str:
    """
    Convierte un valor a texto normalizado.
    """

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


def validate_consolidated_input(
    df: pd.DataFrame,
) -> None:
    """
    Valida la vista consolidada antes de crear
    evaluaciones profesionales.
    """

    missing_columns = (
        REQUIRED_CONSOLIDATED_COLUMNS
        .difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "La vista consolidada no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "La vista consolidada está vacía."
        )

    if df["case_id"].isna().any():
        raise ValueError(
            "Existen registros sin case_id."
        )

    empty_case_ids = (
        df["case_id"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_case_ids.any():
        raise ValueError(
            "Existen registros con case_id vacío."
        )

    if df["case_id"].duplicated().any():
        duplicated_cases = (
            df.loc[
                df["case_id"].duplicated(
                    keep=False
                ),
                "case_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Existen case_id duplicados: "
            f"{sorted(duplicated_cases)}"
        )

    numeric_columns = [
        "risk_score",
        "assessment_coverage",
        "estimated_loss_clp",
    ]

    for column in numeric_columns:
        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric_values.isna().any():
            affected_cases = (
                df.loc[
                    numeric_values.isna(),
                    "case_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} contiene "
                "valores no numéricos. "
                f"Casos afectados: "
                f"{affected_cases}"
            )

        if (
            numeric_values < 0
        ).any():
            affected_cases = (
                df.loc[
                    numeric_values < 0,
                    "case_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} contiene "
                "valores negativos. "
                f"Casos afectados: "
                f"{affected_cases}"
            )

    for value in df[
        "consolidated_alert_level"
    ]:
        normalized = normalize_upper_text(
            value
        )

        if normalized not in (
            VALID_ALERT_LEVELS
        ):
            raise ValueError(
                "Nivel consolidado desconocido: "
                f"{value!r}"
            )


def create_assessment_record(
    row: pd.Series,
    *,
    assessment_id: str,
    evaluated_at: str,
    engine_type: str = (
        "CONSOLIDATED_RULE_ENGINE"
    ),
    engine_version: str = "1.0.0",
    policy_id: str = (
        "CONSOLIDATED_RISK_POLICY"
    ),
    policy_version: str = "1.0.0",
    input_schema_version: str = "1.0.0",
) -> AssessmentRecord:
    """
    Transforma una fila consolidada en una
    evaluación versionada.
    """

    case_id = normalize_text(
        row["case_id"]
    )

    alert_level = normalize_upper_text(
        row[
            "consolidated_alert_level"
        ]
    )

    if alert_level not in (
        VALID_ALERT_LEVELS
    ):
        raise ValueError(
            "Nivel consolidado desconocido: "
            f"{alert_level!r}"
        )

    return AssessmentRecord(
        assessment_id=assessment_id,
        case_id=case_id,
        engine_type=engine_type,
        engine_version=engine_version,
        policy_id=policy_id,
        policy_version=policy_version,
        input_schema_version=(
            input_schema_version
        ),
        evaluated_at=evaluated_at,
        risk_score=float(
            row["risk_score"]
        ),
        assessment_coverage=float(
            row["assessment_coverage"]
        ),
        calculated_alert_level=(
            alert_level
        ),
        recommended_action=(
            normalize_upper_text(
                row[
                    "consolidated_recommended_action"
                ]
            )
        ),
        data_quality_status=(
            normalize_upper_text(
                row[
                    "consolidated_data_status"
                ]
            )
        ),
    )


def select_alert(
    assessment: AssessmentRecord,
) -> AlertSelectionDecision:
    """
    Decide si una evaluación debe generar
    una alerta.

    Esta política inicial considera el nivel
    consolidado y la calidad de los datos.

    La pérdida económica se utilizará en la
    siguiente revisión para priorización avanzada.
    """

    risk_level = (
        assessment.calculated_alert_level
    )

    data_status = (
        assessment.data_quality_status
    )

    if data_status == (
        "REQUIERE_REVISION_DE_DATOS"
    ):
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="ALTA",
            alert_reason=(
                "La evaluación requiere revisar "
                "la calidad o integridad de los datos."
            ),
        )

    if risk_level == "CRITICO":
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="CRITICA",
            alert_reason=(
                "La evaluación alcanzó nivel "
                "de riesgo CRITICO."
            ),
        )

    if risk_level == "ALTO":
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="ALTA",
            alert_reason=(
                "La evaluación alcanzó nivel "
                "de riesgo ALTO."
            ),
        )

    if risk_level == "MEDIO":
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="MEDIA",
            alert_reason=(
                "La evaluación alcanzó nivel "
                "de riesgo MEDIO."
            ),
        )

    if risk_level == "INCOMPLETO":
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="MEDIA",
            alert_reason=(
                "La evaluación quedó incompleta "
                "y necesita antecedentes."
            ),
        )

    return AlertSelectionDecision(
        should_create_alert=False,
        alert_priority=None,
        alert_reason=(
            "La evaluación no cumple la política "
            "actual de selección de alertas."
        ),
    )


def create_alert_record(
    assessment: AssessmentRecord,
    decision: AlertSelectionDecision,
    *,
    alert_id: str,
    created_at: str,
    estimated_loss_clp: float,
    selection_policy_id: str = (
        "ALERT_SELECTION_POLICY"
    ),
    selection_policy_version: str = "1.0.0",
) -> AlertRecord:
    """
    Crea una alerta desde una evaluación.

    Solo debe llamarse cuando la política haya
    indicado que corresponde crearla.
    """

    if not decision.should_create_alert:
        raise ValueError(
            "No se puede crear una alerta "
            "para una decisión negativa."
        )

    if decision.alert_priority not in (
        VALID_ALERT_PRIORITIES
    ):
        raise ValueError(
            "Prioridad de alerta inválida: "
            f"{decision.alert_priority!r}"
        )

    return AlertRecord(
        alert_id=alert_id,
        assessment_id=(
            assessment.assessment_id
        ),
        case_id=assessment.case_id,
        selection_policy_id=(
            selection_policy_id
        ),
        selection_policy_version=(
            selection_policy_version
        ),
        created_at=created_at,
        alert_status="ABIERTA",
        alert_priority=(
            decision.alert_priority
        ),
        alert_reason=(
            decision.alert_reason
        ),
        risk_level=(
            assessment.calculated_alert_level
        ),
        estimated_loss_clp=float(
            estimated_loss_clp
        ),
    )


def records_to_dataframe(
    records: Iterable[Any],
    *,
    columns: list[str],
) -> pd.DataFrame:
    """
    Convierte dataclasses en DataFrame.

    columns garantiza una estructura estable
    incluso cuando no existen alertas.
    """

    rows = [
        asdict(record)
        for record in records
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def process_professional_pipeline(
    consolidated_df: pd.DataFrame,
    *,
    timestamp: str | None = None,
    assessment_id_factory: Callable[
        [],
        str,
    ] = generate_assessment_id,
    alert_id_factory: Callable[
        [],
        str,
    ] = generate_alert_id,
) -> ProfessionalPipelineResult:
    """
    Crea una evaluación para cada caso y una
    alerta solo cuando la política lo determine.
    """

    validate_consolidated_input(
        consolidated_df
    )

    processing_timestamp = (
        timestamp
        if timestamp is not None
        else current_chile_timestamp()
    )

    assessments: list[
        AssessmentRecord
    ] = []

    alerts: list[
        AlertRecord
    ] = []

    for _, row in consolidated_df.iterrows():
        assessment = (
            create_assessment_record(
                row,
                assessment_id=(
                    assessment_id_factory()
                ),
                evaluated_at=(
                    processing_timestamp
                ),
            )
        )

        assessments.append(
            assessment
        )

        decision = select_alert(
            assessment
        )

        if decision.should_create_alert:
            alert = create_alert_record(
                assessment,
                decision,
                alert_id=(
                    alert_id_factory()
                ),
                created_at=(
                    processing_timestamp
                ),
                estimated_loss_clp=float(
                    row[
                        "estimated_loss_clp"
                    ]
                ),
            )

            alerts.append(
                alert
            )

    assessment_columns = [
        field_name
        for field_name in (
            AssessmentRecord
            .__dataclass_fields__
            .keys()
        )
    ]

    alert_columns = [
        field_name
        for field_name in (
            AlertRecord
            .__dataclass_fields__
            .keys()
        )
    ]

    assessments_df = records_to_dataframe(
        assessments,
        columns=assessment_columns,
    )

    alerts_df = records_to_dataframe(
        alerts,
        columns=alert_columns,
    )

    priority_rank = {
        "CRITICA": 4,
        "ALTA": 3,
        "MEDIA": 2,
        "BAJA": 1,
    }

    if not alerts_df.empty:
        alerts_df[
            "_priority_rank"
        ] = (
            alerts_df[
                "alert_priority"
            ]
            .map(priority_rank)
            .fillna(0)
        )

        alerts_df = alerts_df.sort_values(
            by=[
                "_priority_rank",
                "estimated_loss_clp",
            ],
            ascending=[
                False,
                False,
            ],
        ).drop(
            columns=[
                "_priority_rank"
            ]
        ).reset_index(
            drop=True
        )

    return ProfessionalPipelineResult(
        assessments=assessments_df,
        alerts=alerts_df,
    )