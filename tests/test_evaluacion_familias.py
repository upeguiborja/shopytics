"""Pruebas de evaluacion sin ejecutar la busqueda ni modelos costosos."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluacion  # noqa: E402
from proyecto_ml import bootstrap_diferencia  # noqa: E402


def test_umbral_f1_usa_solo_puntos_con_umbral_y_desempata():
    umbral, f1 = evaluacion.umbral_f1_validacion([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])
    assert umbral == pytest.approx(0.2)
    assert f1 == pytest.approx(0.8)


def test_bootstrap_ap_pareado_reproducible_y_signo():
    y = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    b = 1 - a
    resultado = bootstrap_diferencia(y, a, b, average_precision_score, n_boot=30, semilla=7)
    assert resultado == bootstrap_diferencia(y, a, b, average_precision_score, n_boot=30, semilla=7)
    assert resultado[0] == pytest.approx(average_precision_score(y, a) - average_precision_score(y, b))
    assert resultado[2] > 0
    assert bootstrap_diferencia(y, a, a, average_precision_score, n_boot=5) == (0, 0, 0, 0)


def test_configuraciones_rechaza_familias_duplicadas(tmp_path):
    ruta = tmp_path / "config.csv"
    pd.DataFrame({"modelo": ["MLP", "MLP"], "params": ["{}", "{}"]}).to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="exactamente una"):
        evaluacion.cargar_configuraciones(ruta)


def test_evaluacion_sin_fuga_y_salida_configurable(tmp_path, monkeypatch):
    nombres = ["RandomForest", "MLP", "SVM", "kNN", "Regresion_logistica"]
    ruta = tmp_path / "config.csv"
    pd.DataFrame({"modelo": nombres, "params": ["{'parametro': 3}"] * 5}).to_csv(ruta, index=False)
    X = pd.DataFrame({"score": [0.1, 0.9, 0.2, 0.8, 0.1, 0.7, 0.4, 0.6, 0.3, 0.9, 0.2, 0.8]})
    y = pd.Series([0, 1] * 6)
    indices = (np.arange(4), np.arange(4, 8), np.arange(8, 12))
    ajustes = []

    def ajustar(nombre, params, x, etiquetas, **opciones):
        assert list(x.index) == list(indices[0])
        assert list(etiquetas.index) == list(indices[0])
        ajustes.append((nombre, params, opciones))
        return nombre

    monkeypatch.setattr(evaluacion, "cargar_datos", lambda ruta: None)
    monkeypatch.setattr(evaluacion, "particionar", lambda datos: (None, X, y, *indices))
    monkeypatch.setattr(evaluacion, "ajustar", ajustar)
    monkeypatch.setattr(evaluacion, "puntuacion", lambda modelo, x: x["score"].to_numpy())
    monkeypatch.setattr(evaluacion, "predecir", lambda modelo, x: (x["score"].to_numpy() >= 0.5).astype(int))
    salida = tmp_path / "salida"
    resultado = evaluacion.evaluar_familias("datos.csv", ruta, salida, n_boot=3, rapido=True)
    assert len(resultado["metricas"]) == 15
    assert len(resultado["intervalos"]) == 105
    assert len(resultado["comparaciones_ap_test"]) == 10
    assert len(ajustes) == 7
    assert all(params == {"parametro": 3} for _, params, _ in ajustes)
    assert ajustes[-2][2] == {"incluir_pagevalues": False}
    assert ajustes[-1][2] == {"incluir_specialday": False}
    assert ((resultado["comparaciones_ap_test"].modelo_a == "RandomForest") &
            (resultado["comparaciones_ap_test"].modelo_b == "MLP")).any()
    assert json.loads((salida / "resumen.json").read_text(encoding="utf-8"))["n_boot"] == 3
    umbrales = resultado["umbrales"].copy()
    y.iloc[indices[2]] = 1 - y.iloc[indices[2]]
    X.loc[indices[2], "score"] = 1 - X.loc[indices[2], "score"]
    resultado_2 = evaluacion.evaluar_familias("datos.csv", ruta, tmp_path / "otra", n_boot=2,
                                               ejecutar_ablaciones=False)
    pd.testing.assert_frame_equal(umbrales, resultado_2["umbrales"])
    assert resultado["resumen"]["ganador_validacion"] == resultado_2["resumen"]["ganador_validacion"]
