from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FraudTypology:
    code: str
    category: str
    display_name: str
    description: str
    common_tactics: tuple[str, ...]
    recommended_data: tuple[str, ...]
    recommended_controls: tuple[str, ...]


FRAUD_TYPOLOGIES: dict[str, FraudTypology] = {
    "FINANCIAL_STATEMENT_FRAUD": FraudTypology(
        code="FINANCIAL_STATEMENT_FRAUD",
        category="FRAUDE_FINANCIERO",
        display_name="Manipulación de estados financieros",
        description=(
            "Alteración deliberada de registros o informes "
            "financieros para presentar una situación falsa."
        ),
        common_tactics=(
            "manipulacion_contable",
            "ocultamiento",
            "documentacion_falsa",
        ),
        recommended_data=(
            "asientos_contables",
            "facturas",
            "notas_de_credito",
            "cierres_mensuales",
            "usuarios_que_modificaron_registros",
        ),
        recommended_controls=(
            "segregacion_de_funciones",
            "conciliaciones",
            "auditoria_continua",
            "registro_de_cambios",
        ),
    ),
    "EMBEZZLEMENT": FraudTypology(
        code="EMBEZZLEMENT",
        category="FRAUDE_FINANCIERO",
        display_name="Apropiación indebida de activos",
        description=(
            "Uso o apropiación no autorizada de dinero, "
            "bienes o recursos confiados a una persona."
        ),
        common_tactics=(
            "abuso_de_acceso",
            "ocultamiento",
            "proveedor_ficticio",
        ),
        recommended_data=(
            "pagos",
            "proveedores",
            "cuentas_bancarias",
            "aprobaciones",
            "logs_de_usuario",
        ),
        recommended_controls=(
            "doble_aprobacion",
            "segregacion_de_funciones",
            "confirmacion_de_cuentas_bancarias",
            "revision_de_proveedores",
        ),
    ),
    "IDENTITY_THEFT": FraudTypology(
        code="IDENTITY_THEFT",
        category="FRAUDE_IDENTIDAD",
        display_name="Robo de identidad",
        description=(
            "Uso no autorizado de información personal "
            "para obtener productos, servicios o acceso."
        ),
        common_tactics=(
            "suplantacion",
            "documentos_falsos",
            "ingenieria_social",
        ),
        recommended_data=(
            "documentos_de_identidad",
            "datos_de_contacto",
            "dispositivos",
            "direcciones_ip",
            "solicitudes_de_productos",
        ),
        recommended_controls=(
            "verificacion_de_identidad",
            "validacion_documental",
            "mfa",
            "analisis_de_dispositivo",
        ),
    ),
    "SYNTHETIC_IDENTITY": FraudTypology(
        code="SYNTHETIC_IDENTITY",
        category="FRAUDE_IDENTIDAD",
        display_name="Identidad sintética",
        description=(
            "Creación de una identidad ficticia mediante "
            "la combinación de información real y falsa."
        ),
        common_tactics=(
            "suplantacion",
            "datos_combinados",
            "cuentas_multiples",
        ),
        recommended_data=(
            "identificadores_personales",
            "telefonos",
            "correos",
            "direcciones",
            "dispositivos_compartidos",
        ),
        recommended_controls=(
            "verificacion_cruzada",
            "analisis_de_redes",
            "validacion_documental",
            "deteccion_de_identidades_relacionadas",
        ),
    ),
    "ACCOUNT_TAKEOVER": FraudTypology(
        code="ACCOUNT_TAKEOVER",
        category="FRAUDE_DIGITAL",
        display_name="Toma de cuenta",
        description=(
            "Acceso no autorizado a una cuenta legítima "
            "utilizando credenciales robadas o engaño."
        ),
        common_tactics=(
            "suplantacion",
            "phishing",
            "robo_de_credenciales",
        ),
        recommended_data=(
            "inicios_de_sesion",
            "direcciones_ip",
            "dispositivos",
            "cambios_de_clave",
            "eventos_mfa",
        ),
        recommended_controls=(
            "mfa",
            "analisis_de_comportamiento",
            "deteccion_de_dispositivo_nuevo",
            "autenticacion_adaptativa",
        ),
    ),
    "PHISHING": FraudTypology(
        code="PHISHING",
        category="FRAUDE_DIGITAL",
        display_name="Phishing",
        description=(
            "Engaño mediante comunicaciones que simulan "
            "provenir de una fuente confiable."
        ),
        common_tactics=(
            "ingenieria_social",
            "suplantacion",
            "enlaces_maliciosos",
        ),
        recommended_data=(
            "correos",
            "dominios",
            "urls",
            "encabezados_de_mensaje",
            "reportes_de_usuarios",
        ),
        recommended_controls=(
            "filtro_de_correo",
            "capacitacion",
            "verificacion_de_dominios",
            "mfa",
        ),
    ),
    "PAYMENT_CARD_FRAUD": FraudTypology(
        code="PAYMENT_CARD_FRAUD",
        category="FRAUDE_PAGOS",
        display_name="Fraude con tarjeta de pago",
        description=(
            "Uso no autorizado de los datos de una tarjeta "
            "para realizar transacciones."
        ),
        common_tactics=(
            "tarjeta_robada",
            "skimming",
            "phishing",
            "compra_no_presencial",
        ),
        recommended_data=(
            "transacciones",
            "comercio",
            "ubicacion",
            "dispositivo",
            "historial_del_cliente",
        ),
        recommended_controls=(
            "tokenizacion",
            "autenticacion_reforzada",
            "limites_dinamicos",
            "analisis_de_comportamiento",
        ),
    ),
    "INSURANCE_HARD_FRAUD": FraudTypology(
        code="INSURANCE_HARD_FRAUD",
        category="FRAUDE_SEGUROS",
        display_name="Fraude de seguros duro",
        description=(
            "Invención deliberada de una pérdida, daño "
            "o siniestro para obtener una indemnización."
        ),
        common_tactics=(
            "siniestro_inventado",
            "documentacion_falsa",
            "colusion",
        ),
        recommended_data=(
            "declaracion_del_siniestro",
            "fotografias",
            "peritajes",
            "geolocalizacion",
            "relaciones_entre_participantes",
        ),
        recommended_controls=(
            "peritaje_independiente",
            "verificacion_documental",
            "analisis_de_redes",
            "comparacion_historica",
        ),
    ),
    "INSURANCE_SOFT_FRAUD": FraudTypology(
        code="INSURANCE_SOFT_FRAUD",
        category="FRAUDE_SEGUROS",
        display_name="Fraude de seguros blando",
        description=(
            "Exageración de una pérdida o daño que "
            "realmente ocurrió."
        ),
        common_tactics=(
            "inflacion_del_monto",
            "daños_exagerados",
            "documentacion_alterada",
        ),
        recommended_data=(
            "monto_reclamado",
            "peritajes",
            "cotizaciones",
            "historial_de_reclamos",
            "evidencia_del_daño",
        ),
        recommended_controls=(
            "comparacion_de_costos",
            "peritaje",
            "deteccion_de_outliers",
            "revision_de_reincidencia",
        ),
    ),
    "CORRUPTION": FraudTypology(
        code="CORRUPTION",
        category="CORRUPCION",
        display_name="Corrupción",
        description=(
            "Abuso de una posición de poder para obtener "
            "un beneficio personal o para terceros."
        ),
        common_tactics=(
            "soborno",
            "comision_ilegal",
            "conflicto_de_interes",
            "colusion",
        ),
        recommended_data=(
            "licitaciones",
            "proveedores",
            "aprobaciones",
            "beneficiarios_finales",
            "relaciones_entre_personas",
        ),
        recommended_controls=(
            "declaracion_de_conflictos",
            "segregacion_de_funciones",
            "analisis_de_redes",
            "auditoria_de_proveedores",
        ),
    ),
}


def normalize_fraud_code(value: str) -> str:
    """
    Normaliza el código ingresado por un usuario o archivo.
    """

    if not isinstance(value, str):
        raise TypeError(
            "El código de tipología debe ser texto."
        )

    normalized = (
        value.strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if not normalized:
        raise ValueError(
            "El código de tipología no puede estar vacío."
        )

    return normalized


def get_fraud_typology(
    fraud_code: str,
) -> FraudTypology:
    """
    Devuelve la tipología solicitada.

    Genera un error cuando el código no pertenece
    al catálogo oficial del sistema.
    """

    normalized_code = normalize_fraud_code(
        fraud_code
    )

    try:
        return FRAUD_TYPOLOGIES[
            normalized_code
        ]
    except KeyError as exc:
        available_codes = ", ".join(
            sorted(FRAUD_TYPOLOGIES)
        )

        raise ValueError(
            "Tipología de fraude desconocida: "
            f"{fraud_code!r}. "
            "Códigos disponibles: "
            f"{available_codes}"
        ) from exc


def list_typologies_by_category(
    category: str,
) -> list[FraudTypology]:
    """
    Devuelve todas las tipologías de una categoría.
    """

    normalized_category = normalize_fraud_code(
        category
    )

    return sorted(
        [
            typology
            for typology in FRAUD_TYPOLOGIES.values()
            if typology.category
            == normalized_category
        ],
        key=lambda item: item.code,
    )