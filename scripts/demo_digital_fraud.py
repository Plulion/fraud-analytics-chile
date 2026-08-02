from __future__ import annotations

from src.digital_fraud import assess_digital_event


def main() -> None:
    event = {
        "new_device": 1,
        "unusual_ip": 1,
        "geolocation_mismatch": 0,
        "failed_mfa_attempts": 3,
        "recent_password_reset": 1,
        "new_beneficiary": 1,
        "amount_spike_ratio": 4.0,
        "rapid_transaction_count_10m": 5,
    }

    result = assess_digital_event(event)

    print("\nEVALUACIÓN DE FRAUDE DIGITAL\n")

    print(
        "Puntaje por dispositivo: "
        f"{result.device_score}"
    )

    print(
        "Puntaje por red e IP: "
        f"{result.network_score}"
    )

    print(
        "Puntaje de autenticación: "
        f"{result.authentication_score}"
    )

    print(
        "Puntaje transaccional: "
        f"{result.transaction_score}"
    )

    print(
        "\nPuntaje digital total: "
        f"{result.digital_risk_score}"
    )

    print(
        "Cobertura de la evaluación: "
        f"{result.assessment_coverage}%"
    )

    print(
        "Nivel de alerta: "
        f"{result.alert_level}"
    )

    print(
        "Acción recomendada: "
        f"{result.recommended_action}"
    )

    print("\nSEÑALES ACTIVADAS\n")

    if result.triggered_signals:
        for signal in result.triggered_signals:
            print(f"- {signal}")
    else:
        print("- No se activaron señales")

    print(
        "\nIMPORTANTE: este puntaje no confirma fraude."
    )


if __name__ == "__main__":
    main()