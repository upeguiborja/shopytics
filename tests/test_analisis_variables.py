"""Pruebas del analisis individual y del experimento de eliminacion (fase A.3)."""

import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import analisis_variables as av  # noqa: E402
from proyecto_ml import (  # noqa: E402
    BINARIAS,
    ESPECIALES,
    NOMINALES,
    NUMERICAS,
    PREDICTORAS,
)


def _datos(n=240, semilla=42):
    """DataFrame sintetico con las 17 predictoras y Revenue."""
    rng = np.random.RandomState(semilla)
    y = (rng.random(n) < 0.3).astype(int)
    datos = {}
    for i, var in enumerate(NUMERICAS):
        datos[var] = rng.gamma(2.0, 1.0, n) + 0.5 * y
    datos["SpecialDay"] = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n)
    for i, var in enumerate(NOMINALES):
        datos[var] = np.where(
            rng.random(n) < 0.5, "A", np.where(rng.random(n) < 0.5, "B", "C")
        )
    datos["Weekend"] = rng.randint(0, 2, n)
    datos["Revenue"] = y
    return pd.DataFrame(datos)


def _Xy():
    datos = _datos()
    return datos[PREDICTORAS].copy(), datos["Revenue"].astype(int).copy()


def _tabla_analisis(variables, aucs):
    return pd.DataFrame({"variable": variables, "auc_univariado": aucs})


def _csv_configuraciones(ruta, parametros=None):
    parametros = parametros or {"RandomForest": "{}", "MLP": "{}"}
    pd.DataFrame(
        [
            {"modelo": modelo, "AP": 0.7, "params": params}
            for modelo, params in parametros.items()
        ]
    ).to_csv(ruta, index=False)


class QuitarColumnas(BaseEstimator, TransformerMixin):
    """Simula el contrato de construir_modelos(excluir=...)."""

    def __init__(self, excluir=()):
        self.excluir = tuple(excluir)

    def fit(self, X, y=None):
        self.columnas_ = [c for c in X.columns if c not in self.excluir]
        return self

    def transform(self, X):
        return X[self.columnas_].to_numpy(dtype=float)


def test_analisis_genera_tabla_de_las_17_variables():
    X, y = _Xy()
    tabla = av.analizar_variables(X, y)

    assert len(tabla) == len(PREDICTORAS) == 17
    assert sorted(tabla["variable"]) == sorted(PREDICTORAS)
    assert tabla["ranking"].tolist() == list(range(1, 18))
    assert tabla["auc_univariado"].between(0.5, 1.0).all()
    assert tabla["auc_univariado"].is_monotonic_decreasing

    tipos = dict(zip(tabla["variable"], tabla["tipo"]))
    assert all(tipos[v] == "numérica" for v in NUMERICAS + ESPECIALES)
    assert all(tipos[v] == "categórica" for v in NOMINALES)
    assert all(tipos[v] == "binaria" for v in BINARIAS)

    for fila in tabla.itertuples():
        esperado = (
            "mantener"
            if fila.auc_univariado >= 0.6
            else ("revisar" if fila.auc_univariado >= 0.55 else "eliminar o combinar")
        )
        assert fila.recomendacion == esperado
    assert tabla["p_valor"].isna().sum() == len(NOMINALES + BINARIAS)

    numericas = tabla[tabla["tipo"] == "numérica"]["indice_fisher"]
    categoricas = tabla[tabla["tipo"] == "categórica"]["indice_fisher"]
    assert numericas.notna().all()
    assert (numericas >= 0).all()
    assert categoricas.isna().all()


def test_indice_fisher_crece_con_la_separacion_de_clases():
    rng = np.random.RandomState(7)
    y = np.array([0] * 50 + [1] * 50)
    separadas = np.concatenate([rng.normal(0, 0.1, 50), rng.normal(3, 0.1, 50)])
    mezcladas = np.concatenate([rng.normal(0, 0.1, 50), rng.normal(0.01, 0.1, 50)])
    assert av.indice_fisher(separadas, y) > av.indice_fisher(mezcladas, y)
    assert np.isnan(av.indice_fisher(np.ones(100), y))
    assert np.isnan(av.indice_fisher(separadas, np.zeros(100, dtype=int)))


def test_auc_categorica_usa_cross_val_predict_estratificado(monkeypatch):
    proba = np.array([0.2, 0.8] * 50)
    capturado = {}

    def falso_cross_val_predict(estimator, X, y, cv=None, method=None, n_jobs=None):
        capturado.update(
            estimator=estimator, X=X, y=y, cv=cv, method=method, n_jobs=n_jobs
        )
        return np.column_stack([1 - proba, proba])

    monkeypatch.setattr(av, "cross_val_predict", falso_cross_val_predict)
    x = np.array(["a", "b"] * 50)
    y = np.array([0, 1] * 50)

    auc, direccion = av.auc_univariado_categorica(x, y, n_splits=5, semilla=7)

    assert isinstance(capturado["cv"], StratifiedKFold)
    assert capturado["cv"].get_n_splits() == 5
    assert capturado["cv"].shuffle is True
    assert capturado["cv"].random_state == 7
    assert capturado["method"] == "predict_proba"
    assert capturado["X"].shape == (100, 1)
    assert capturado["estimator"].named_steps["onehot"].handle_unknown == "ignore"
    assert auc == pytest.approx(1.0)
    assert direccion == "positiva"


def test_auc_categorica_invierte_direccion(monkeypatch):
    proba = np.array([0.8, 0.2] * 50)
    monkeypatch.setattr(
        av,
        "cross_val_predict",
        lambda *args, **kwargs: np.column_stack([1 - proba, proba]),
    )
    x = np.array(["a", "b"] * 50)
    y = np.array([0, 1] * 50)

    auc, direccion = av.auc_univariado_categorica(x, y)

    assert auc == pytest.approx(1.0)
    assert direccion == "negativa"


def test_redundancia_numericas_detecta_el_par_conocido():
    X, _ = _Xy()
    X = X.copy()
    X["ExitRates"] = X["BounceRates"] * 2

    tabla = av.redundancia_numericas(X)

    assert list(tabla.columns) == ["variable_1", "variable_2", "correlacion"]
    pares = set(zip(tabla["variable_1"], tabla["variable_2"]))
    assert ("BounceRates", "ExitRates") in pares
    fila = tabla[
        (tabla["variable_1"] == "BounceRates") & (tabla["variable_2"] == "ExitRates")
    ].iloc[0]
    assert fila["correlacion"] == pytest.approx(1.0)


def test_candidatas_eliminacion_por_umbral_de_auc():
    tabla = _tabla_analisis(["a", "b", "c", "d"], [0.50, 0.54, 0.55, 0.70])
    assert av.candidatas_eliminacion(tabla) == ["a", "b"]

    empate = _tabla_analisis(["b", "a", "c"], [0.5, 0.5, 0.7])
    assert av.candidatas_eliminacion(empate) == ["a", "b"]

    with pytest.raises(ValueError):
        av.candidatas_eliminacion(pd.DataFrame({"variable": ["a"]}))


def test_cargar_configuraciones_elige_la_mejor_fila_de_rf_y_mlp(tmp_path):
    ruta = tmp_path / "mejores.csv"
    pd.DataFrame(
        [
            {"modelo": "RandomForest", "AP": 0.70, "params": "{'clf__C': 0.1}"},
            {"modelo": "RandomForest", "AP": 0.75, "params": "{'clf__C': 0.2}"},
            {"modelo": "MLP", "AP": 0.72, "params": "{'clf__alpha': 0.01}"},
            {"modelo": "SVM", "AP": 0.90, "params": "{'clf__C': 1}"},
        ]
    ).to_csv(ruta, index=False)

    configuraciones = av.cargar_configuraciones(ruta)

    assert [nombre for nombre, _ in configuraciones] == ["RandomForest", "MLP"]
    assert configuraciones[0][1] == {"clf__C": 0.2}
    assert configuraciones[1][1] == {"clf__alpha": 0.01}


def test_cargar_configuraciones_rechaza_familias_ausentes(tmp_path):
    ruta = tmp_path / "mejores.csv"
    _csv_configuraciones(ruta, {"RandomForest": "{}"})

    with pytest.raises(ValueError, match="MLP"):
        av.cargar_configuraciones(ruta)


def test_cargar_configuraciones_rechaza_parametros_invalidos(tmp_path):
    ruta = tmp_path / "mejores.csv"
    _csv_configuraciones(
        ruta, {"RandomForest": "{'clf__C': 0.2}", "MLP": "no-es-un-diccionario"}
    )

    with pytest.raises(ValueError, match="Parametros invalidos"):
        av.cargar_configuraciones(ruta)


def test_ejecutar_eliminacion_pasa_excluir_y_escribe_tablas(tmp_path, monkeypatch):
    rng = np.random.RandomState(42)
    columnas = NUMERICAS + ["Weekend", "Region", "Browser", "VisitorType", "SpecialDay"]
    X = pd.DataFrame(rng.normal(size=(200, len(columnas))), columns=columnas)
    y = pd.Series([0, 1] * 100)
    tabla = _tabla_analisis(
        NUMERICAS, [0.51, 0.52, 0.53, 0.54, 0.55, 0.90, 0.80, 0.70, 0.60]
    )
    candidatas = list(NUMERICAS[:4])
    llamadas = []

    def constructor(excluir=None):
        excluidas = list(excluir or [])
        llamadas.append(excluidas)
        return (
            {
                nombre: Pipeline(
                    [
                        ("quitar", QuitarColumnas(excluidas)),
                        ("clf", LogisticRegression(max_iter=500, random_state=42)),
                    ]
                )
                for nombre in ("RandomForest", "MLP")
            },
            {},
        )

    monkeypatch.setattr(av, "construir_modelos", constructor)
    ruta = tmp_path / "mejores.csv"
    _csv_configuraciones(ruta)

    resultado = av.ejecutar_eliminacion(
        X.iloc[:140],
        y.iloc[:140],
        X.iloc[140:],
        y.iloc[140:],
        tabla,
        ruta,
        tmp_path,
        n_splits=5,
        semilla=42,
    )

    esperado = [[], candidatas, ["ProductRelated_Duration", "BounceRates"]]
    assert llamadas == esperado * 2

    cv = pd.read_csv(tmp_path / "eliminacion_cv.csv")
    assert len(cv) == 6
    assert set(cv["variante"]) == {"completo", "sin_candidatas", "sin_redundantes"}
    assert cv["AP_CV"].between(0.0, 1.0).all()
    assert cv["AP_folds"].map(lambda s: len(json.loads(s))).eq(5).all()

    validacion = pd.read_csv(tmp_path / "eliminacion_validacion.csv")
    assert len(validacion) == 6
    assert validacion["AP"].between(0.0, 1.0).all()
    assert validacion["excluidas"].isna().sum() == 2

    resumen = json.loads(
        (tmp_path / "eliminacion_resumen.json").read_text(encoding="utf-8")
    )
    assert resumen["candidatas"] == candidatas
    assert resumen["n_folds"] == 5
    assert resumen["variantes"][1]["excluidas"] == candidatas
    assert set(resumen["delta_AP_validacion_respecto_completo"]) == {
        "RandomForest",
        "MLP",
    }
    assert resumen["delta_AP_validacion_respecto_completo"]["RandomForest"][
        "completo"
    ] == pytest.approx(0.0)
    assert resultado["resumen"] == resumen


def test_eliminacion_rechaza_variable_ausente(tmp_path, monkeypatch):
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(60, 2)), columns=["Administrative", "PageValues"])
    y = pd.Series([0, 1] * 30)
    tabla = _tabla_analisis(["NoExiste", "Administrative"], [0.1, 0.7])
    ruta = tmp_path / "mejores.csv"
    _csv_configuraciones(ruta)
    monkeypatch.setattr(av, "construir_modelos", lambda excluir=None: ({}, {}))

    with pytest.raises(ValueError, match="NoExiste"):
        av.ejecutar_eliminacion(X, y, X, y, tabla, ruta, tmp_path)


def test_ejecutar_analisis_usa_solo_entrenamiento(tmp_path, monkeypatch):
    datos = _datos()
    ruta = tmp_path / "datos.csv"
    datos.to_csv(ruta, index=False)
    _, X, y, idx_tr, idx_va, idx_te = av.particionar(av.cargar_datos(ruta))
    capturado = {}

    def espia(X_tr, y_tr, n_splits=av.N_FOLDS, semilla=av.SEMILLA):
        capturado["indices"] = list(X_tr.index)
        return pd.DataFrame(
            {
                "variable": ["Administrative"],
                "tipo": ["numérica"],
                "metrica": ["correlación punto-biserial"],
                "valor": [0.1],
                "p_valor": [0.5],
                "auc_univariado": [0.6],
                "direccion": ["positiva"],
                "informacion_mutua": [0.01],
                "ranking": [1],
                "recomendacion": ["mantener"],
            }
        )

    monkeypatch.setattr(av, "analizar_variables", espia)
    salida = tmp_path / "resultados" / "variables"
    resultado = av.ejecutar_analisis(
        X.iloc[idx_tr],
        y.iloc[idx_tr].astype(int),
        X.iloc[idx_va],
        y.iloc[idx_va].astype(int),
        tmp_path / "no-se-lee.csv",
        salida=salida,
        ejecutar_eliminacion_experimento=False,
    )

    assert capturado["indices"] == list(X.iloc[idx_tr].index)
    assert not set(capturado["indices"]) & set(idx_te)
    assert resultado["salida"] == salida
    assert resultado["eliminacion"] is None
    assert resultado["resumen"]["rapido"] is False
    assert resultado["resumen"]["analisis"]["n_entrenamiento"] == len(idx_tr)
    assert (salida / "analisis_variables.csv").exists()


def test_ejecutar_analisis_delega_eliminacion_con_validacion(tmp_path, monkeypatch):
    X, y = _Xy()
    X_tr, y_tr = X.iloc[:168], y.iloc[:168]
    X_va, y_va = X.iloc[168:204], y.iloc[168:204]
    ruta = tmp_path / "mejores.csv"
    capturado = {}

    def espia(*args, **kwargs):
        capturado["args"] = args
        capturado["kwargs"] = kwargs
        return {"cv": None, "validacion": None, "resumen": None}

    monkeypatch.setattr(av, "ejecutar_eliminacion", espia)
    salida = tmp_path / "variables"
    av.ejecutar_analisis(
        X_tr,
        y_tr,
        X_va,
        y_va,
        ruta,
        salida,
        ejecutar_eliminacion_experimento=True,
    )

    X_cv, _, X_recibida_va, _ = capturado["args"][:4]
    assert len(X_cv) == len(X_tr)
    assert len(X_recibida_va) == len(X_va)
    assert capturado["args"][5] == ruta
    assert capturado["args"][6] == salida
    assert "n_candidatas" not in capturado["kwargs"]


def test_ejecutar_analisis_sin_eliminacion_genera_las_dos_tablas(tmp_path):
    X, y = _Xy()
    salida = tmp_path / "variables"

    resultado = av.ejecutar_analisis(
        X.iloc[:168],
        y.iloc[:168],
        X.iloc[168:204],
        y.iloc[168:204],
        tmp_path / "no-se-lee.csv",
        salida,
        ejecutar_eliminacion_experimento=False,
    )

    tabla = pd.read_csv(salida / "analisis_variables.csv")
    assert len(tabla) == len(PREDICTORAS)
    assert (salida / "redundancia_numericas.csv").exists()
    assert resultado["eliminacion"] is None


def test_ejecutar_analisis_rapido_usa_dos_folds_y_submuestra(tmp_path, monkeypatch):
    rng = np.random.RandomState(42)
    columnas = NUMERICAS + ["Weekend", "Region", "Browser", "VisitorType", "SpecialDay"]
    X = pd.DataFrame(rng.normal(size=(800, len(columnas))), columns=columnas)
    y = pd.Series([0, 1] * 400)
    X_tr, y_tr = X.iloc[:700], y.iloc[:700]
    X_va, y_va = X.iloc[700:], y.iloc[700:]
    tabla = _tabla_analisis(
        NUMERICAS, [0.51, 0.52, 0.53, 0.54, 0.55, 0.90, 0.80, 0.70, 0.60]
    )
    monkeypatch.setattr(
        av,
        "analizar_variables",
        lambda X, y, n_splits=av.N_FOLDS, semilla=av.SEMILLA: tabla,
    )
    monkeypatch.setattr(
        av,
        "redundancia_numericas",
        lambda X: pd.DataFrame(columns=["variable_1", "variable_2", "correlacion"]),
    )

    def constructor(excluir=None):
        return (
            {
                nombre: Pipeline(
                    [
                        ("quitar", QuitarColumnas(excluir)),
                        ("clf", LogisticRegression(max_iter=300, random_state=42)),
                    ]
                )
                for nombre in ("RandomForest", "MLP")
            },
            {},
        )

    monkeypatch.setattr(av, "construir_modelos", constructor)
    _csv_configuraciones(tmp_path / "mejores.csv")
    resultado = av.ejecutar_analisis(
        X_tr,
        y_tr,
        X_va,
        y_va,
        tmp_path / "mejores.csv",
        tmp_path / "variables",
        rapido=True,
    )

    cv = pd.read_csv(tmp_path / "variables" / "eliminacion_cv.csv")
    assert cv["AP_folds"].map(lambda s: len(json.loads(s))).eq(2).all()
    assert resultado["resumen"]["rapido"] is True
    assert resultado["eliminacion"]["resumen"]["n_entrenamiento"] == 600
    assert resultado["eliminacion"]["resumen"]["n_validacion"] == 100


def test_validar_configuraciones_acepta_diccionario():
    configuraciones = av._validar_configuraciones(
        {"RandomForest": {"clf__C": 1}, "MLP": {}}
    )
    assert [nombre for nombre, _ in configuraciones] == ["RandomForest", "MLP"]
    assert configuraciones[0][1] == {"clf__C": 1}

    with pytest.raises(ValueError, match="MLP"):
        av._validar_configuraciones({"RandomForest": {}})
    with pytest.raises(TypeError):
        av._validar_configuraciones(42)


def test_importar_no_crea_el_directorio_de_salida(tmp_path):
    destino = tmp_path / "src"
    destino.mkdir()
    shutil.copy(Path(av.__file__), destino / "analisis_variables.py")
    insertada = str(destino.resolve())

    try:
        espacio = runpy.run_path(str(destino / "analisis_variables.py"))
    finally:
        if insertada in sys.path:
            sys.path.remove(insertada)

    assert espacio["SALIDA"] == (tmp_path / "resultados" / "variables").resolve()
    assert not (tmp_path / "resultados" / "variables").exists()
