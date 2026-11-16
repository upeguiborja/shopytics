"""Especificaciones de las particiones estratificadas 60/20/20."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proyecto_ml import (  # noqa: E402
    RAIZ,
    SEMILLA,
    cargar_datos,
    particionar,
    resumen_particiones,
)

N_UNICO = 12205
N_TR = 7323
N_VA = 2441
N_TE = 2441
POS_TR = 1145
POS_VA = 381
POS_TE = 382


def _particiones():
    return particionar(cargar_datos())


def test_tamanos_de_las_particiones():
    df_u, _, _, idx_tr, idx_va, idx_te = _particiones()
    assert len(df_u) == N_UNICO
    assert (len(idx_tr), len(idx_va), len(idx_te)) == (N_TR, N_VA, N_TE)
    assert len(idx_tr) + len(idx_va) + len(idx_te) == N_UNICO


def test_particiones_disjuntas_y_completas():
    df_u, _, _, idx_tr, idx_va, idx_te = _particiones()
    union = np.concatenate([idx_tr, idx_va, idx_te])
    assert len(np.unique(union)) == len(df_u)
    assert set(union.tolist()) == set(range(len(df_u)))


def test_estratificacion_por_clase():
    _, _, y, idx_tr, idx_va, idx_te = _particiones()
    positivos = tuple(int(y.iloc[ix].sum()) for ix in (idx_tr, idx_va, idx_te))
    assert positivos == (POS_TR, POS_VA, POS_TE)
    global_ = float(y.mean())
    for ix in (idx_tr, idx_va, idx_te):
        assert float(y.iloc[ix].mean()) == pytest.approx(global_, abs=1e-3)


def test_datos_sin_duplicados_exactos():
    df_u, _, _, _, _, _ = _particiones()
    assert len(df_u) == N_UNICO
    assert int(df_u.duplicated().sum()) == 0


def test_resumen_de_particiones():
    _, _, y, idx_tr, idx_va, idx_te = _particiones()
    resumen = resumen_particiones(y, idx_tr, idx_va, idx_te)
    assert list(resumen.columns) == [
        "particion", "n", "positivos", "negativos", "proporcion_compra",
    ]
    assert resumen["particion"].tolist() == ["Entrenamiento", "Validacion", "Test"]
    assert resumen["n"].tolist() == [N_TR, N_VA, N_TE]
    assert resumen["positivos"].tolist() == [POS_TR, POS_VA, POS_TE]
    assert resumen["negativos"].tolist() == [
        N_TR - POS_TR, N_VA - POS_VA, N_TE - POS_TE,
    ]

    ruta = RAIZ / "resultados_corregidos" / "resumen_particiones.csv"
    if ruta.exists():
        pd.testing.assert_frame_equal(resumen, pd.read_csv(ruta))


def test_particiones_reproducibles_con_la_semilla():
    assert SEMILLA == 42
    datos = cargar_datos()
    _, _, _, tr1, va1, te1 = particionar(datos)
    _, _, _, tr2, va2, te2 = particionar(datos)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(va1, va2)
    assert np.array_equal(te1, te2)


def test_manifiesto_coincide_con_las_particiones_cuando_existe():
    rutas = sorted(RAIZ.glob("resultados*/manifiesto.json"))
    if not rutas:
        pytest.skip("No hay manifiesto.json en resultados/")
    _, _, _, idx_tr, idx_va, idx_te = _particiones()
    esperado = {
        "n_train": len(idx_tr),
        "n_validacion": len(idx_va),
        "n_test": len(idx_te),
    }
    revisados = 0
    for ruta in rutas:
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        for clave, valor in esperado.items():
            if clave in manifiesto:
                assert manifiesto[clave] == valor, f"{clave} desactualizado en {ruta}"
                revisados += 1
    if revisados == 0:
        pytest.skip("Los manifiestos no registran tamanos de particion")
