"""Contratos de los artefactos de evaluación y reducción.

Comprueba rangos válidos de las métricas, el orden de los intervalos y que la
diferencia puntual de cada comparación pareada quede dentro de su IC bootstrap.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
EVAL = RES / "evaluacion_familias"
RED = RES / "reduccion"

METRICAS = ("AP", "ROC_AUC", "precision", "recall", "F1", "balanced_accuracy", "accuracy")


def _csv(ruta):
    if not ruta.is_file():
        pytest.skip(f"No existe {ruta.relative_to(RAIZ)}")
    return pd.read_csv(ruta)


def test_metricas_de_evaluacion_en_rango():
    tabla = _csv(EVAL / "metricas.csv")
    for columna in METRICAS:
        assert columna in tabla.columns
        valores = pd.to_numeric(tabla[columna], errors="coerce").dropna()
        assert valores.between(0.0, 1.0).all(), f"{columna} fuera de [0, 1]"


def test_intervalos_de_evaluacion_ordenados_y_con_el_valor_dentro():
    tabla = _csv(EVAL / "intervalos.csv")
    assert {"modelo", "metrica", "ic_inferior", "ic_superior"}.issubset(tabla.columns)
    assert (tabla["ic_inferior"] <= tabla["ic_superior"]).all()
    if "valor" in tabla.columns:
        assert tabla["valor"].between(tabla["ic_inferior"], tabla["ic_superior"]).all()


def test_comparaciones_pareadas_con_diferencia_dentro_del_ic():
    tabla = _csv(EVAL / "comparaciones_test.csv")
    assert {"diferencia_puntual", "ic_inferior", "ic_superior"}.issubset(tabla.columns)
    assert (tabla["ic_inferior"] <= tabla["ic_superior"]).all()
    dentro = tabla["diferencia_puntual"].between(tabla["ic_inferior"], tabla["ic_superior"])
    assert dentro.all(), tabla.loc[~dentro, ["modelo_a", "modelo_b", "metrica"]]


def test_reduccion_test_tiene_ic_por_metrica_y_en_rango():
    tabla = _csv(RED / "reduccion_test.csv")
    for metrica in ("AP", "ROC_AUC"):
        assert metrica in tabla.columns
        assert f"ic_{metrica}_inferior" in tabla.columns
        assert f"ic_{metrica}_superior" in tabla.columns
        valores = pd.to_numeric(tabla[metrica], errors="raise")
        assert valores.between(0.0, 1.0).all()
        inferior = tabla[f"ic_{metrica}_inferior"]
        superior = tabla[f"ic_{metrica}_superior"]
        assert (inferior <= superior).all()
        assert valores.between(inferior, superior).all()


def test_reduccion_comparacion_con_diferencia_dentro_del_ic():
    tabla = _csv(RED / "reduccion_comparacion.csv")
    assert {"diferencia_puntual", "ic_inferior", "ic_superior"}.issubset(tabla.columns)
    assert (tabla["ic_inferior"] <= tabla["ic_superior"]).all()
    dentro = tabla["diferencia_puntual"].between(tabla["ic_inferior"], tabla["ic_superior"])
    assert dentro.all()
    assert set(tabla["metrica"]) == {"AP", "ROC_AUC"}


def test_cada_familia_reporta_base_pca_umap_y_umap2d():
    tabla = _csv(RED / "reduccion_test.csv")
    for modelo, grupo in tabla.groupby("modelo"):
        metodos = set(grupo["metodo"])
        assert {"Base", "PCA", "UMAP", "UMAP2D"}.issubset(metodos), modelo
        assert np.isfinite(grupo.loc[grupo["metodo"] == "Base", "AP"]).all()
