"""Comprueba que los óptimos elegidos no queden en bordes artificiales de malla."""

import ast
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"

# Bordes que sí son naturales: un árbol sin límite de profundidad y una hoja de
# una observación no indican que la malla se haya quedado corta.
EXCEPCIONES = {
    ("RandomForest", "clf__min_samples_leaf"): {1},
    ("RandomForest", "clf__max_depth"): {None},
}

MALLAS = {
    "RandomForest": {
        "clf__max_depth": [4, 6, 8, 10, 12, 20, None],
        "clf__max_features": ["log2", "sqrt", 0.3, 0.5, 0.7],
        "clf__min_samples_leaf": [1, 5, 10, 20],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
        "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "clf__learning_rate_init": [0.001, 0.01, 0.03, 0.1, 0.3],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10, 30, 100],
        "clf__gamma": ["scale", 0.001, 0.01, 0.03, 0.1],
    },
    "kNN": {
        "clf__n_neighbors": [5, 15, 31, 45, 61, 81, 101],
    },
}


def _candidatos():
    ruta = RES / "cv_todos_los_candidatos.csv"
    if not ruta.is_file():
        pytest.skip("No existe cv_todos_los_candidatos.csv")
    return pd.read_csv(ruta)


def test_el_optimo_de_cv_no_queda_en_un_borde_no_natural():
    tabla = _candidatos()
    for modelo, grupo in tabla.groupby("modelo"):
        if modelo not in MALLAS:
            continue
        mejor = grupo.loc[grupo["mean_test_score"].idxmax()]
        params = ast.literal_eval(mejor["params"])
        for clave, valores in MALLAS[modelo].items():
            if clave not in params:
                continue
            elegido = params[clave]
            permitidos = EXCEPCIONES.get((modelo, clave), set())
            if elegido in permitidos:
                continue
            posicion = valores.index(elegido)
            assert 0 < posicion < len(valores) - 1, (
                f"{modelo} elige {clave}={elegido} en el borde de {valores}"
            )


def test_la_dimension_elegida_de_umap_no_es_la_maxima_de_la_rejilla():
    ruta = RES / "reduccion" / "umap_barrido.csv"
    if not ruta.is_file():
        pytest.skip("No existe umap_barrido.csv")
    tabla = pd.read_csv(ruta)
    maxima = tabla["dimensiones"].max()
    elegidas = tabla[tabla["seleccionado"].astype(str).str.lower().isin({"true", "1"})]
    assert not elegidas.empty
    assert (elegidas["dimensiones"] < maxima).all()


def test_la_particion_de_prueba_no_participa_en_la_seleccion():
    ruta = RES / "reduccion" / "reduccion_protocolo.json"
    if not ruta.is_file():
        pytest.skip("No existe reduccion_protocolo.json")
    import json

    protocolo = json.loads(ruta.read_text(encoding="utf-8"))
    assert "prueba" in protocolo
    assert "seleccion" in protocolo
    for fila in protocolo["seleccion"]:
        assert "AP_CV" in fila or "AP_CV" in fila.keys()
