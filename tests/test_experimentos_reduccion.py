"""Pruebas focalizadas del protocolo de reduccion."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import experimentos_reduccion as reduccion  # noqa: E402


def test_familias_y_parametros_proceden_del_csv(tmp_path):
    ruta = tmp_path / "seleccion.csv"
    pd.DataFrame([
        {"modelo": "RandomForest", "AP": 0.7, "params": "{'clf__max_depth': 3}"},
        {"modelo": "MLP", "AP": 0.8, "params": "{'clf__alpha': 0.23, 'pre__num__log__usar_log': False}"},
        {"modelo": "SVM", "AP": 0.9, "params": "{'clf__C': 7}"},
    ]).to_csv(ruta, index=False)
    seleccion = reduccion.cargar_configuraciones(ruta)
    assert [r[0] for r in seleccion] == ["SVM", "MLP"]
    assert seleccion[0][2].get_params()["clf__C"] == 7
    pipe = reduccion.crear_pipeline(seleccion[1][2], "PCA")
    assert pipe.get_params()["clf__alpha"] == 0.23
    assert pipe.get_params()["pre__num__log__usar_log"] is False
    assert pipe.named_steps["reduccion"].n_components == 0.95


def test_ajustes_aislados_y_seleccion_independiente_de_test(tmp_path, monkeypatch):
    ajustes = []

    class Registro(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            ajustes.append(set(X.index))
            return self

        def transform(self, X):
            return X.to_numpy()

    def pipeline(modelo, metodo, dimensiones=None):
        return Pipeline([("registro", Registro()), ("escalar", StandardScaler()),
                         ("clf", LogisticRegression(random_state=42))])

    # Sustituimos PCA/UMAP por un espia para verificar el protocolo sin coste UMAP.
    # PCA debe exponer el numero de componentes aprendido.
    class RegistroPCA(Registro):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.n_components_ = 2
            return self

    def pipeline_espia(modelo, metodo, dimensiones=None):
        pipe = pipeline(modelo, metodo, dimensiones)
        pipe.steps[0] = ("reduccion", RegistroPCA() if metodo == "PCA" else Registro())
        return pipe

    monkeypatch.setattr(reduccion, "crear_pipeline", pipeline_espia)
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(120, 3)))
    y = pd.Series(np.tile([0, 1], 60))
    configuraciones = [("A", {}, None), ("B", {}, None)]
    args = (X.iloc[:80], y.iloc[:80], X.iloc[80:100], y.iloc[80:100],
            X.iloc[100:], y.iloc[100:])
    primero = reduccion.ejecutar_experimentos(*args, configuraciones, tmp_path / "a", True)
    args_cambiados = (*args[:4], args[4] * -100, 1 - args[5])
    segundo = reduccion.ejecutar_experimentos(*args_cambiados, configuraciones, tmp_path / "b", True)
    assert primero["seleccion"] == segundo["seleccion"]
    assert all(indices <= set(range(80)) for indices in ajustes)
    assert any(len(indices) == 40 for indices in ajustes)
    assert any(len(indices) == 80 for indices in ajustes)
    test = pd.read_csv(tmp_path / "a" / "reduccion_test.csv")
    assert len(test) == 4
    val = pd.read_csv(tmp_path / "a" / "reduccion_validacion.csv")
    assert val["seleccionado"].sum() == 4
    protocolo = json.loads((tmp_path / "a" / "reduccion_protocolo.json").read_text(encoding="utf-8"))
    assert protocolo["n_folds"] == 2
