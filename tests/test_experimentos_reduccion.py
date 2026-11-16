"""Pruebas focalizadas del protocolo de reduccion (Fase A.1)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import experimentos_reduccion as reduccion  # noqa: E402


def test_familias_parametros_y_rejillas(tmp_path):
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
    assert reduccion.crear_pipeline(seleccion[1][2], "PCA", 5).named_steps[
        "reduccion"
    ].n_components == 5
    umap = reduccion.crear_pipeline(seleccion[1][2], "UMAP", 10, 30, 7)
    assert umap.named_steps["reduccion"].n_components == 10
    assert umap.named_steps["reduccion"].n_neighbors == 30
    assert umap.named_steps["reduccion"].random_state == 7


def test_candidatos_pca_explicitos_y_rejilla_umap_de_quince():
    assert reduccion.COMPONENTES_PCA == (5, 10, 15, 20, 28, 35, 45, 0.95)
    assert reduccion.DIMENSIONES_UMAP == (2, 5, 10, 15, 20)
    assert reduccion.VECINOS_UMAP == (15, 30, 50)
    assert reduccion.UMAP2D_VECINOS == 15
    rejilla = [(d, v) for d in reduccion.DIMENSIONES_UMAP
               for v in reduccion.VECINOS_UMAP]
    assert len(rejilla) == 15
    assert len(set(rejilla)) == 15


def _fila(componentes, vecinos, ap, std, solicitada=None):
    return {
        "componentes": componentes,
        "vecinos": vecinos,
        "AP": ap,
        "std_CV": std,
        "dimensiones_solicitadas": componentes if solicitada is None else solicitada,
    }


def test_simplicidad_umap_y_frontera_inclusiva():
    ganador = _fila(10, 15, 0.70, 0.02)
    simple = _fila(5, 50, 0.685, 0.03)
    elegido, detalle = reduccion.seleccionar_por_validacion([ganador, simple], "UMAP")
    assert elegido is simple
    assert detalle["motivo_seleccion"] == "simplicidad"
    assert detalle["diferencia_con_max"] == pytest.approx(0.015)
    assert detalle["umbral_simplicidad"] == pytest.approx(0.68)

    simple["AP"] = 0.68  # diferencia exactamente igual a std_CV
    elegido, detalle = reduccion.seleccionar_por_validacion([ganador, simple], "UMAP")
    assert elegido is simple
    assert detalle["motivo_seleccion"] == "simplicidad"

    simple["AP"] = 0.679  # diferencia apenas mayor que std_CV
    elegido, detalle = reduccion.seleccionar_por_validacion([ganador, simple], "UMAP")
    assert elegido is ganador
    assert detalle["motivo_seleccion"] == "mayor_AP_validacion"


def test_simplicidad_umap_prefiere_menos_dimensiones_y_mas_vecinos():
    ganador = _fila(10, 15, 0.70, 0.02)
    cinco = _fila(5, 15, 0.69, 0.02)
    cinco_suave = _fila(5, 50, 0.685, 0.02)
    dos = _fila(2, 15, 0.681, 0.02)
    elegido, _ = reduccion.seleccionar_por_validacion(
        [ganador, cinco, cinco_suave, dos], "UMAP"
    )
    assert elegido is dos
    elegido, _ = reduccion.seleccionar_por_validacion(
        [ganador, cinco, cinco_suave], "UMAP"
    )
    assert elegido is cinco_suave


def test_simplicidad_pca_usa_componentes_ajustados_y_fijo_antes_de_95():
    ganador = _fila(20, None, 0.70, 0.02, solicitada=20)
    quince = _fila(15, None, 0.685, 0.02, solicitada=15)
    pca95 = _fila(18, None, 0.69, 0.02, solicitada=0.95)
    elegido, _ = reduccion.seleccionar_por_validacion(
        [ganador, quince, pca95], "PCA"
    )
    assert elegido is quince

    fijo = _fila(10, None, 0.685, 0.02, solicitada=10)
    pca95_igual = _fila(10, None, 0.685, 0.02, solicitada=0.95)
    elegido, _ = reduccion.seleccionar_por_validacion(
        [ganador, fijo, pca95_igual], "PCA"
    )
    assert elegido is fijo


def test_simplicidad_con_std_cero_o_no_finita():
    ganador = _fila(5, 15, 0.60, np.nan, solicitada=5)
    simple = _fila(2, 15, 0.60, np.nan, solicitada=2)
    elegido, detalle = reduccion.seleccionar_por_validacion([ganador, simple], "UMAP")
    assert elegido is simple
    assert detalle["desviacion_std_CV"] == 0.0

    simple["AP"] = 0.599
    elegido, _ = reduccion.seleccionar_por_validacion([ganador, simple], "UMAP")
    assert elegido is ganador


_AJUSTES = []


class _Registro(BaseEstimator, TransformerMixin):
    """Transformador espia: registra los indices con los que se ajusta.

    El registro es un atributo de clase porque sklearn clona los estimadores
    dentro de la CV y el clonado copia los parametros de instancia.
    """

    def fit(self, X, y=None):
        if hasattr(X, "index"):
            _AJUSTES.append(set(X.index))
        return self

    def transform(self, X):
        return np.asarray(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"f{i}" for i in range(100)], dtype=object)


class _RegistroPCA(_Registro):
    def fit(self, X, y=None):
        self.n_components_ = 2
        self.explained_variance_ratio_ = np.array([0.6, 0.3])
        return self


def _modelo_familia():
    return Pipeline([
        ("pre", _Registro()),
        ("clf", LogisticRegression(random_state=42)),
    ])


def _espias(monkeypatch):
    def pipeline_espia(modelo, metodo, dimensiones=None, vecinos=None, semilla=42):
        reductor = _RegistroPCA() if metodo == "PCA" else _Registro()
        return Pipeline([
            ("pre", _Registro()),
            ("reduccion", reductor),
            ("clf", LogisticRegression(random_state=42)),
        ])

    monkeypatch.setattr(reduccion, "crear_pipeline", pipeline_espia)


def _datos():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(120, 3)))
    y = pd.Series(np.tile([0, 1], 60))
    return X, y


def test_protocolo_salidas_y_ausencia_de_fuga(tmp_path, monkeypatch):
    _AJUSTES.clear()
    _espias(monkeypatch)
    X, y = _datos()
    configuraciones = [("A", {}, _modelo_familia()),
                        ("B", {}, _modelo_familia())]
    args = (X.iloc[:80], y.iloc[:80], X.iloc[80:100], y.iloc[80:100],
            X.iloc[100:], y.iloc[100:])
    primero = reduccion.ejecutar_experimentos(
        *args, configuraciones, tmp_path / "a", True, 2000
    )
    args_cambiados = (*args[:4], args[4] * -100, 1 - args[5])
    segundo = reduccion.ejecutar_experimentos(
        *args_cambiados, configuraciones, tmp_path / "b", True, 2000
    )

    assert primero["seleccion"] == segundo["seleccion"]
    assert all(indices <= set(range(80)) for indices in _AJUSTES)
    assert any(len(indices) == 40 for indices in _AJUSTES)
    assert any(len(indices) == 80 for indices in _AJUSTES)

    test = pd.read_csv(tmp_path / "a" / "reduccion_test.csv")
    assert len(test) == 8
    assert set(test["metodo"]) == {"Base", "PCA", "UMAP", "UMAP2D"}
    assert set(test["modelo"]) == {"A", "B"}
    for metrica in ("AP", "ROC_AUC", "F1"):
        assert f"ic_{metrica}_inferior" in test.columns
        assert f"ic_{metrica}_superior" in test.columns
        assert f"media_bootstrap_{metrica}" in test.columns
    assert test["ic_AP_inferior"].notna().all()
    assert (test["ic_AP_inferior"] <= test["ic_AP_superior"]).all()

    val = pd.read_csv(tmp_path / "a" / "reduccion_validacion.csv")
    assert len(val) == 12
    assert val["seleccionado"].sum() == 4
    assert set(val["metodo"]) == {"Base", "PCA", "UMAP", "UMAP2D"}
    cv = pd.read_csv(tmp_path / "a" / "reduccion_cv.csv")
    assert len(cv) == 8
    assert cv["finalista"].all()
    assert len(cv[cv.metodo == "PCA"]) == 6
    assert len(cv[cv.metodo == "UMAP"]) == 2

    comparacion = pd.read_csv(tmp_path / "a" / "reduccion_comparacion.csv")
    assert len(comparacion) == 12
    assert set(comparacion["metrica"]) == {"AP", "ROC_AUC"}
    assert set(comparacion["metodo"]) == {"PCA", "UMAP", "UMAP2D"}
    assert "Base" not in set(comparacion["metodo"])
    assert comparacion["diferencia_puntual"].notna().all()
    for metrica in ("AP", "ROC_AUC"):
        sub = comparacion[comparacion.metrica == metrica]
        assert np.allclose(sub["valor"] - sub["valor_base"], sub["diferencia_puntual"])
    assert (comparacion["n_boot"] == 20).all()

    barrido_pca = pd.read_csv(tmp_path / "a" / "pca_barrido.csv")
    assert len(barrido_pca) == 6
    assert "AP_validacion" in barrido_pca.columns
    assert "AP" not in barrido_pca.columns
    barrido_umap = pd.read_csv(tmp_path / "a" / "umap_barrido.csv")
    assert len(barrido_umap) == 2
    assert {"dimensiones", "vecinos", "AP_validacion"} <= set(barrido_umap.columns)

    varianza = pd.read_csv(tmp_path / "a" / "pca_varianza.csv")
    assert len(varianza) == 6
    assert set(varianza["modelo"]) == {"A", "B"}
    assert np.allclose(
        varianza.groupby("modelo")["varianza_acumulada"].max().to_numpy(), 1.0
    )
    varianza_familias = pd.read_csv(tmp_path / "a" / "pca_varianza_familias.csv")
    assert len(varianza_familias) == 6
    assert set(varianza_familias["modelo"]) == {"A", "B"}

    coords_train = pd.read_csv(tmp_path / "a" / "umap2d_train.csv")
    coords_test = pd.read_csv(tmp_path / "a" / "umap2d_test.csv")
    assert len(coords_train) == 160
    assert len(coords_test) == 40
    assert set(coords_train["indice"]) == set(X.iloc[:80].index)
    assert set(coords_test["indice"]) == set(X.iloc[100:].index)
    assert (coords_train["origen"] == "umap2d").all()
    assert np.isfinite(coords_train[["x", "y"]]).all().all()
    assert np.isfinite(coords_test[["x", "y"]]).all().all()

    coordenadas_json = json.loads(
        (tmp_path / "a" / "umap_2d.json").read_text(encoding="utf-8")
    )
    assert len(coordenadas_json["train"]) == 80
    assert len(coordenadas_json["test"]) == 20
    assert len(coordenadas_json["y_test"]) == 20
    assert coordenadas_json["modelo"] in {"A", "B"}
    assert set(coordenadas_json["por_familia"]) == {"A", "B"}

    protocolo = json.loads(
        (tmp_path / "a" / "reduccion_protocolo.json").read_text(encoding="utf-8")
    )
    assert protocolo["rapido"] is True
    assert protocolo["n_folds"] == 2
    assert protocolo["n_boot"] == 20
    assert len(protocolo["pca_95"]) == 2
    assert len(protocolo["seleccion"]) == 4
    assert "<=" in protocolo["regla_simplicidad"]["frontera"]
    assert "1x1" in protocolo["computo"]["rejilla_umap"]
    assert "reduccion_comparacion.csv" in protocolo["salidas"]
    assert "umap_2d.json" in protocolo["salidas"]
    assert set(protocolo["variantes_test"]) == {"Base", "PCA", "UMAP", "UMAP2D"}
    assert protocolo["candidatos_pca_por_familia"][0]["descartados"] == []
    assert protocolo["candidatos_pca_por_familia"][0]["n_features_preprocesadas"] == 100


def test_todas_las_configuraciones_umap_pasan_validacion(tmp_path, monkeypatch):
    _AJUSTES.clear()
    _espias(monkeypatch)
    X, y = _datos()
    configuraciones = [("A", {}, _modelo_familia()),
                        ("B", {}, _modelo_familia())]
    args = (X.iloc[:60], y.iloc[:60], X.iloc[60:80], y.iloc[60:80],
            X.iloc[80:], y.iloc[80:])
    protocolo = reduccion.ejecutar_experimentos(
        *args, configuraciones, tmp_path / "a", False, 20
    )
    assert protocolo["n_folds"] == 5
    assert len(protocolo["rejilla_umap"]) == 15

    cv = pd.read_csv(tmp_path / "a" / "reduccion_cv.csv")
    umap_cv = cv[cv.metodo == "UMAP"]
    assert len(umap_cv) == 30  # 15 configuraciones por familia
    assert umap_cv["finalista"].all()
    assert set(umap_cv["vecinos"]) == {15, 30, 50}
    assert set(umap_cv["dimensiones_solicitadas"]) == {2, 5, 10, 15, 20}

    val = pd.read_csv(tmp_path / "a" / "reduccion_validacion.csv")
    umap_val = val[val.metodo == "UMAP"]
    assert len(umap_val) == 30  # las quince pasan a validacion
    assert umap_val["AP"].notna().all()
    assert umap_val["seleccionado"].sum() == 2  # una por familia
    pca_val = val[val.metodo == "PCA"]
    assert len(pca_val) == 16  # ocho candidatos por familia
    assert pca_val["seleccionado"].sum() == 2
    assert len(val[val.metodo == "Base"]) == 2
    assert len(val[val.metodo == "UMAP2D"]) == 2

    test = pd.read_csv(tmp_path / "a" / "reduccion_test.csv")
    assert len(test) == 8
    comparacion = pd.read_csv(tmp_path / "a" / "reduccion_comparacion.csv")
    assert len(comparacion) == 12
    assert (comparacion["n_boot"] == 20).all()

    barrido_umap = pd.read_csv(tmp_path / "a" / "umap_barrido.csv")
    assert len(barrido_umap) == 30
    assert barrido_umap["AP_validacion"].notna().all()
    barrido_pca = pd.read_csv(tmp_path / "a" / "pca_barrido.csv")
    assert len(barrido_pca) == 16
    assert barrido_pca["AP_validacion"].notna().all()


def test_candidatos_pca_se_acotan_a_las_variables_disponibles(tmp_path, monkeypatch):
    _AJUSTES.clear()
    _espias(monkeypatch)
    X, y = _datos()
    # El espia declara 100 variables, por lo que ningun candidato se descarta.
    configuraciones = [("A", {}, _modelo_familia())]
    args = (X.iloc[:60], y.iloc[:60], X.iloc[60:80], y.iloc[60:80],
            X.iloc[80:], y.iloc[80:])
    protocolo = reduccion.ejecutar_experimentos(
        *args, configuraciones, tmp_path / "a", False, 20
    )
    info = protocolo["candidatos_pca_por_familia"][0]
    assert info["candidatos_pca"] == ["5", "10", "15", "20", "28", "35", "45", "0.95"]
    assert info["descartados"] == []
    assert info["n_features_preprocesadas"] == 100
