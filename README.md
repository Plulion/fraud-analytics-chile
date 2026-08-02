# Capítulo 1 — Triángulo del Fraude aplicado a Chile

Proyecto educativo basado en los conceptos del Capítulo 1 de *Fraud Analytics in Action*.

## Propósito

Priorizar casos ficticios mediante cuatro dimensiones:

- oportunidad;
- presión;
- racionalización;
- factores agravantes de colusión y competencia.

El puntaje **no es una probabilidad de fraude**, no reemplaza una investigación y no debe utilizarse para acusar a una persona. Es una herramienta didáctica de priorización.

Todos los datos son sintéticos. Las organizaciones y situaciones son ficticias.

## Requisitos

- Python 3.11 o superior
- VS Code
- Extensión oficial de Python para VS Code

## Instalación en macOS

```bash
cd fraud_analytics_chile_capitulo_01
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución

```bash
python -m src.main
```

También puede indicar rutas distintas:

```bash
python -m src.main   --input data/casos_sinteticos_chile.csv   --output outputs/evaluacion_riesgo.csv
```

## Pruebas

```bash
pytest -q
```

## Desafíos sugeridos

1. Cambiar los pesos del puntaje y justificar el cambio.
2. Agregar la variable `anonymous_reporting_available`.
3. Penalizar más los casos con demora prolongada de detección.
4. Crear una visualización por sector y región.
5. Sustituir el puntaje experto por un modelo aprendido cuando exista una variable objetivo válida.
