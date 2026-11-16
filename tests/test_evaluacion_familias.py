"""Pruebas de evaluacion sin ejecutar la busqueda ni modelos costosos."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluacion  # noqa: E402
import busqueda  # noqa: E402
from proyecto_ml import bootstrap_diferencia  # noqa: E402


def test_umbral_f1_usa_solo_puntos_con_umbral_y_desempata():
    umbral, f1 = evaluacion.umbral_f1_validacion([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])
    assert umbral == pytest.approx(0.2)
    assert f1 == pytest.approx(0.8)


def test_intervalo_t_usa_desviacion_muestral():
    valores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    resultado = evaluacion.intervalo_t(valores)
    desviacion = np.std(valores, ddof=1)
    error = desviacion / np.sqrt(len(valores))
    t_critico = t_dist.ppf(0.975, len(valores) - 1)
    assert resultado["media"] == pytest.approx(0.7)
    assert resultado["desviacion"] == pytest.approx(desviacion)
    assert resultado["error_estandar"] == pytest.approx(error)
    assert resultado["ic_inferior"] == pytest.approx(0.7 - t_critico * error)
    assert resultado["ic_superior"] == pytest.approx(0.7 + t_critico * error)
    assert resultado["n_folds"] == 5
    with pytest.raises(ValueError, match="al menos dos"):
        evaluacion.intervalo_t([0.5])


def test_validacion_cruzada_usa_cinco_folds_estratificados(monkeypatch):
    capturado = {}
    valores = {
        "AP": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        "ROC_AUC": np.array([0.80, 0.82, 0.84, 0.86, 0.88]),
        "F1": np.array([0.40, 0.45, 0.50, 0.55, 0.60]),
    }

    def fake_cross_validate(modelo, X, y, **kwargs):
        capturado.update(kwargs)
        return {f"test_{metrica}": valores[metrica] for metrica in evaluacion.METRICAS_CV}

    monkeypatch.setattr(evaluacion, "cross_validate", fake_cross_validate)
    monkeypatch.setattr(
        evaluacion, "construir_modelos",
        lambda: ({"RandomForest": LogisticRegression()}, {}),
    )
    tabla = evaluacion.validacion_cruzada(
        {"RandomForest": {}}, pd.DataFrame({"x": [0, 1, 2, 3, 4]}),
        pd.Series([0, 1, 0, 1, 0]),
    )
    cv = capturado["cv"]
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == evaluacion.N_FOLDS_CV == busqueda.N_SPLITS == 5
    assert cv.shuffle is True
    assert cv.random_state == evaluacion.SEMILLA
    assert capturado["scoring"] == evaluacion.METRICAS_CV
    assert capturado["n_jobs"] == 1
    assert capturado["error_score"] == "raise"
    assert list(tabla.metrica) == ["AP", "ROC_AUC", "F1"]
    assert set(tabla.n_folds) == {5}
    assert tabla.loc[tabla.metrica == "AP", "media"].iloc[0] == pytest.approx(0.7)


def test_validacion_cruzada_real_produce_intervalos(monkeypatch):
    monkeypatch.setattr(
        evaluacion, "construir_modelos",
        lambda: ({"RandomForest": LogisticRegression(max_iter=1000)}, {}),
    )
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"x": rng.normal(size=60)})
    y = pd.Series(np.tile([0, 1], 30))
    tabla = evaluacion.validacion_cruzada({"RandomForest": {}}, X, y)
    assert len(tabla) == 3
    assert set(tabla.metrica) == {"AP", "ROC_AUC", "F1"}
    assert (tabla.n_folds == 5).all()
    assert (tabla.ic_inferior <= tabla.media).all()
    assert (tabla.media <= tabla.ic_superior).all()


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
    llamadas_cv = []

    def ajustar(nombre, params, x, etiquetas, **opciones):
        assert list(x.index) == list(indices[0])
        assert list(etiquetas.index) == list(indices[0])
        ajustes.append((nombre, params, opciones))
        return nombre

    def validacion_cruzada(configuraciones, X_cv, y_cv, **kwargs):
        assert list(X_cv.index) == list(indices[0])
        assert list(y_cv.index) == list(indices[0])
        llamadas_cv.append((list(configuraciones), kwargs))
        return pd.DataFrame([
            {"modelo": nombre, "metrica": metrica, "n_folds": 5, "media": 0.5,
             "desviacion": 0.01, "error_estandar": 0.004, "ic_inferior": 0.49,
             "ic_superior": 0.51}
            for nombre in nombres for metrica in ("AP", "ROC_AUC", "F1")
        ])

    monkeypatch.setattr(evaluacion, "cargar_datos", lambda ruta: None)
    monkeypatch.setattr(evaluacion, "particionar", lambda datos: (None, X, y, *indices))
    monkeypatch.setattr(evaluacion, "ajustar", ajustar)
    monkeypatch.setattr(evaluacion, "validacion_cruzada", validacion_cruzada)
    monkeypatch.setattr(evaluacion, "puntuacion", lambda modelo, x: x["score"].to_numpy())
    monkeypatch.setattr(evaluacion, "predecir", lambda modelo, x: (x["score"].to_numpy() >= 0.5).astype(int))
    salida = tmp_path / "salida"
    resultado = evaluacion.evaluar_familias("datos.csv", ruta, salida, n_boot=3, rapido=True)
    assert len(resultado["metricas"]) == 15
    assert len(resultado["intervalos"]) == 105
    assert len(resultado["cv_intervalos"]) == 15
    assert len(resultado["comparaciones_test"]) == 20
    assert len(ajustes) == 7
    assert len(llamadas_cv) == 1
    assert llamadas_cv[0][0] == nombres
    assert llamadas_cv[0][1]["semilla"] == evaluacion.SEMILLA
    assert all(params == {"parametro": 3} for _, params, _ in ajustes)
    assert ajustes[-2][2] == {"incluir_pagevalues": False}
    assert ajustes[-1][2] == {"incluir_specialday": False}
    assert set(resultado["comparaciones_test"].metrica) == {"AP", "ROC_AUC"}
    assert ((resultado["comparaciones_test"].modelo_a == "RandomForest") &
            (resultado["comparaciones_test"].modelo_b == "MLP")).any()
    assert json.loads((salida / "resumen.json").read_text(encoding="utf-8"))["n_boot"] == 3
    assert list(pd.read_csv(salida / "cv_intervalos.csv").columns) == [
        "modelo", "metrica", "n_folds", "media", "desviacion", "error_estandar",
        "ic_inferior", "ic_superior",
    ]
    assert list(pd.read_csv(salida / "comparaciones_test.csv").columns) == [
        "modelo_a", "modelo_b", "metrica", "diferencia_puntual", "media_bootstrap",
        "ic_inferior", "ic_superior", "n_boot",
    ]
    umbrales = resultado["umbrales"].copy()
    y.iloc[indices[2]] = 1 - y.iloc[indices[2]]
    X.loc[indices[2], "score"] = 1 - X.loc[indices[2], "score"]
    resultado_2 = evaluacion.evaluar_familias("datos.csv", ruta, tmp_path / "otra", n_boot=2,
                                               ejecutar_ablaciones=False)
    assert len(llamadas_cv) == 2
    assert (tmp_path / "otra" / "cv_intervalos.csv").is_file()
    assert (tmp_path / "otra" / "comparaciones_test.csv").is_file()
    pd.testing.assert_frame_equal(umbrales, resultado_2["umbrales"])
    assert resultado["resumen"]["ganador_validacion"] == resultado_2["resumen"]["ganador_validacion"]
