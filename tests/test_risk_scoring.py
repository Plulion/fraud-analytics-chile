from src.risk_scoring import assess_case


def test_high_risk_case() -> None:
    case = {
        "weak_controls": 1,
        "privileged_access": 1,
        "financial_pressure": 1,
        "performance_pressure": 1,
        "rationalization_signal": 1,
        "collusion_signal": 1,
        "high_competence": 1,
    }

    result = assess_case(case)

    assert result.opportunity == 50.0
    assert result.pressure == 30.0
    assert result.rationalization == 15.0
    assert result.aggravating_factors == 5.0
    assert result.risk_score == 100.0
    assert result.assessment_coverage == 100.0
    assert result.alert_level == "CRITICO"
    assert result.decision_status == "REVISION_PRIORITARIA"


def test_low_risk_case() -> None:
    case = {
        "weak_controls": 0,
        "privileged_access": 0,
        "financial_pressure": 0,
        "performance_pressure": 0,
        "rationalization_signal": 0,
        "collusion_signal": 0,
        "high_competence": 0,
    }

    result = assess_case(case)

    assert result.opportunity == 0.0
    assert result.pressure == 0.0
    assert result.rationalization == 0.0
    assert result.aggravating_factors == 0.0
    assert result.risk_score == 0.0
    assert result.assessment_coverage == 100.0
    assert result.alert_level == "BAJO"
    assert result.decision_status == "REVISION_ESTANDAR"


def test_municipality_case() -> None:
    case = {
        "weak_controls": 1,
        "privileged_access": 1,
        "financial_pressure": 0,
        "performance_pressure": 1,
        "rationalization_signal": 1,
        "collusion_signal": 1,
        "high_competence": 1,
    }

    result = assess_case(case)

    assert result.opportunity == 50.0
    assert result.pressure == 15.0
    assert result.rationalization == 15.0
    assert result.aggravating_factors == 5.0
    assert result.risk_score == 85.0
    assert result.assessment_coverage == 100.0
    assert result.alert_level == "CRITICO"
    assert result.decision_status == "REVISION_PRIORITARIA"


def test_unknown_information_is_not_zero() -> None:
    case = {
        "weak_controls": 1,
        "privileged_access": 1,
        "financial_pressure": None,
        "performance_pressure": None,
        "rationalization_signal": None,
        "collusion_signal": None,
        "high_competence": None,
    }

    result = assess_case(case)

    assert result.opportunity == 50.0
    assert result.pressure == 0.0
    assert result.rationalization == 0.0
    assert result.aggravating_factors == 0.0
    assert result.risk_score == 50.0
    assert result.assessment_coverage == 50.0
    assert result.alert_level == "INCOMPLETO"
    assert result.decision_status == "REQUIERE_MAS_ANTECEDENTES"