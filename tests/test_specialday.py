"""Pruebas focalizadas del experimento de SpecialDay."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import specialday  # noqa: E402
from proyecto_ml import BINARIAS, ESPECIALES, NOMINALES, NUMERICAS  # noqa: E402

CONFIGURACIONES = {
    "Regresion_logistica": {"clf__max_iter": 200},
    "kNN": {"clf__n_neighbors": 3},
    "RandomForest": {"clf__n_estimators": 8, "clf__max_depth": 2, "clf__n_jobs": 1},
    "MLP": {
        "clf__hidden_layer_sizes": (4,),
        "clf__max_iter": 10,
        "clf__early_stopping": False,
    },
    "SVM": {"clf__kernel": "linear", "clf__C": 0.1},
}


def _datos(n=160):
    rng = np.random.default_rng(7)
    meses = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    X = pd.DataFrame(index=range(n))
    for columna in NUMERICAS:
        X[columna] = rng.gamma(2.0, 1.0, n)
    X["BounceRates"] = rng.uniform(0.0, 0.2, n)
    X["ExitRates"] = rng.uniform(0.0, 0.2, n)
    X["Month"] = [meses[i % len(meses)] for i in range(n)]
    X["OperatingSystems"] = rng.integers(1, 4, n)
    X["Browser"] = rng.integers(1, 4, n)
    X["Region"] = rng.integers(1, 4, n)
    X["TrafficType"] = rng.integers(1, 5, n)
    X["VisitorType"] = rng.choice(["Returning_Visitor", "New_Visitor"], n)
    X["Weekend"] = rng.integers(0, 2, n)
    X["SpecialDay"] = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n)
    # Cada mes recibe ambas clases en entrenamiento y validacion.
    y = pd.Series(((np.arange(n) // len(meses)) % 2).astype(int), index=X.index)
    return X, y


def _columnas_esperadas():
    return NUMERICAS + NOMINALES + BINARIAS + ESPECIALES


def test_columnas_sinteticas_cubren_las_predictoras():
    X, _ = _datos()
    assert list(X.columns) == _columnas_esperadas()


def test_ejecutar_specialday_genera_tablas_y_deltas(tmp_path, monkeypatch):
    X, y = _datos()
    X_tr, y_tr = X.iloc[:120], y.iloc[:120]
    X_va, y_va = X.iloc[120:], y.iloc[120:]
    longitudes = set()
    puntuacion_original = specialday.puntuacion

    def puntuacion_espia(modelo, X_particion):
        longitudes.add(len(X_particion))
        return puntuacion_original(modelo, X_particion)

    monkeypatch.setattr(specialday, "puntuacion", puntuacion_espia)
    resultado = specialday.ejecutar_specialday(
        X_tr, y_tr, X_va, y_va, CONFIGURACIONES, tmp_path,
        rapido=True, por_mes=True, n_jobs=1,
    )

    ablacion = pd.read_csv(tmp_path / "specialday_ablacion.csv")
    codificacion = pd.read_csv(tmp_path / "specialday_codificacion.csv")
    por_mes = pd.read_csv(tmp_path / "specialday_por_mes.csv")

    assert len(ablacion) == 5
    assert set(ablacion["modelo"]) == set(specialday.FAMILIAS)
    assert np.allclose(
        ablacion["delta_AP_con_menos_sin"],
        ablacion["AP_con_SpecialDay"] - ablacion["AP_sin_SpecialDay"],
    )
    assert len(codificacion) == 5
    assert np.allclose(
        codificacion["delta_AP_CV_onehot_menos_numerico"],
        codificacion["AP_CV_onehot"] - codificacion["AP_CV_numerico"],
    )
    assert np.allclose(
        codificacion["delta_AP_validacion_onehot_menos_numerico"],
        codificacion["AP_validacion_onehot"] - codificacion["AP_validacion_numerico"],
    )
    assert (
        codificacion["n_columnas_onehot"] - codificacion["n_columnas_numerico"] == 5
    ).all()
    assert set(por_mes["mes"]) == set(X_va["Month"])
    assert por_mes["AP_validacion"].notna().all()
    assert longitudes == {len(X_va)}
    assert resultado["resumen"]["n_folds"] == 2
    assert resultado["resumen"]["n_entrenamiento"] == 120


def test_por_mes_desactivable(tmp_path, monkeypatch):
    X, y = _datos()
    tabla_ablacion = pd.DataFrame(
        [{"modelo": "RandomForest", "AP_con_SpecialDay": 0.5}]
    )
    monkeypatch.setattr(specialday, "calcular_ablacion", lambda *a, **k: tabla_ablacion)
    monkeypatch.setattr(
        specialday, "calcular_codificacion", lambda *a, **k: pd.DataFrame()
    )
    monkeypatch.setattr(
        specialday, "calcular_por_mes", lambda *a, **k: pytest.fail("no debe ejecutarse")
    )
    resultado = specialday.ejecutar_specialday(
        X, y, X, y, CONFIGURACIONES, tmp_path, rapido=True, por_mes=False
    )
    assert resultado["por_mes"] is None
    assert not (tmp_path / "specialday_por_mes.csv").exists()


def test_ejecucion_normal_usa_cinco_folds(tmp_path, monkeypatch):
    X, y = _datos()
    capturado = {}
    tabla_ablacion = pd.DataFrame(
        [{"modelo": "RandomForest", "AP_con_SpecialDay": 0.5}]
    )

    def codificacion(configuraciones, X_tr, y_tr, X_va, y_va, n_folds, n_jobs, semilla):
        capturado["n_folds"] = n_folds
        return pd.DataFrame()

    monkeypatch.setattr(specialday, "calcular_ablacion", lambda *a, **k: tabla_ablacion)
    monkeypatch.setattr(specialday, "calcular_codificacion", codificacion)
    resultado = specialday.ejecutar_specialday(
        X, y, X, y, CONFIGURACIONES, tmp_path, rapido=False, por_mes=False
    )
    assert capturado["n_folds"] == 5
    assert resultado["resumen"]["n_folds"] == 5


def test_cargar_configuraciones_desde_csv(tmp_path):
    filas = [
        {"modelo": nombre, "params": repr(params)}
        for nombre, params in CONFIGURACIONES.items()
    ]
    ruta = tmp_path / "config.csv"
    pd.DataFrame(filas).to_csv(ruta, index=False)
    cargadas = specialday.cargar_configuraciones(ruta)
    assert tuple(cargadas) == specialday.FAMILIAS
    assert cargadas["RandomForest"]["clf__n_estimators"] == 8

    pd.DataFrame(filas[:-1]).to_csv(ruta, index=False)
    with pytest.raises(ValueError):
        specialday.cargar_configuraciones(ruta)
