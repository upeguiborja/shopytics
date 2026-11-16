"""Especificaciones de las mallas de hiperparametros."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from busqueda import mallas_ampliadas  # noqa: E402
from proyecto_ml import RAIZ, construir_modelos, contar_combinaciones  # noqa: E402

BASE = {
    "Regresion_logistica": 20,
    "kNN": 56,
    "RandomForest": 64,
    "MLP": 48,
    "SVM": 80,
}
AMPLIADAS = {
    "Regresion_logistica": 20,
    "kNN": 56,
    "RandomForest": 280,
    "MLP": 250,
    "SVM": 124,
}
RAPIDA = {
    "Regresion_logistica": 1,
    "kNN": 1,
    "RandomForest": 1,
    "MLP": 1,
    "SVM": 2,
}


def _conteos(mallas):
    return contar_combinaciones(mallas).set_index("modelo")["combinaciones"].to_dict()


def test_cinco_familias_con_mallas_del_plan():
    modelos, mallas = construir_modelos()
    assert list(modelos) == list(BASE)
    assert list(mallas) == list(BASE)
    assert _conteos(mallas) == BASE
    assert int(contar_combinaciones(mallas)["combinaciones"].sum()) == 268


def test_mallas_ampliadas_coinciden_con_el_archivo_generado():
    ruta = RAIZ / "resultados" / "combinaciones_malla.csv"
    if not ruta.exists():
        pytest.skip("No existe el archivo de combinaciones generado")
    archivo = pd.read_csv(ruta).set_index("modelo")["combinaciones"].to_dict()
    _, mallas = mallas_ampliadas(construir_modelos)
    assert archivo == AMPLIADAS
    assert _conteos(mallas) == archivo


def test_mallas_ampliadas_del_flujo_actual():
    _, mallas = mallas_ampliadas(construir_modelos)
    assert _conteos(mallas) == AMPLIADAS
    assert int(contar_combinaciones(mallas)["combinaciones"].sum()) == 730


def test_malla_rapida_reduce_a_seis_combinaciones():
    _, mallas = mallas_ampliadas(construir_modelos, rapido=True)
    assert _conteos(mallas) == RAPIDA
    assert int(contar_combinaciones(mallas)["combinaciones"].sum()) == 6


def test_svm_separa_kernels_sin_duplicar_gamma():
    _, base = construir_modelos()
    _, ampliadas = mallas_ampliadas(construir_modelos)
    for mallas, esperado in ((base, {"linear": 20, "rbf": 60}),
                             (ampliadas, {"linear": 24, "rbf": 100})):
        busquedas = mallas["SVM"]
        assert isinstance(busquedas, list)
        assert len(busquedas) == 2
        assert {g["clf__kernel"][0] for g in busquedas} == {"linear", "rbf"}
        por_kernel = {
            g["clf__kernel"][0]: int(np.prod([len(v) for v in g.values()]))
            for g in busquedas
        }
        assert por_kernel == esperado
        lineal = next(g for g in busquedas if g["clf__kernel"] == ["linear"])
        assert "clf__gamma" not in lineal


def test_parametros_de_las_mallas_son_validos():
    modelos, base = construir_modelos()
    _, ampliadas = mallas_ampliadas(construir_modelos)
    for mallas in (base, ampliadas):
        for nombre, malla in mallas.items():
            busquedas = malla if isinstance(malla, list) else [malla]
            for grid in busquedas:
                assert all(valores for valores in grid.values())
                params = {clave: valores[0] for clave, valores in grid.items()}
                clone(modelos[nombre]).set_params(**params)
