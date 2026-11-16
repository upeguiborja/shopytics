"""Especificaciones del CSV de entrada y del hash del manifiesto."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proyecto_ml import (  # noqa: E402
    BINARIAS,
    CSV,
    ESPECIALES,
    NOMINALES,
    NUMERICAS,
    OBJETIVO,
    RAIZ,
    auditar,
    cargar_datos,
)

FILAS_CRUDAS = 12330
FILAS_UNICAS = 12205
DUPLICADOS = 125
POSITIVOS = 1908
NEGATIVOS = 10422


def _manifiestos():
    return sorted(RAIZ.glob("resultados*/manifiesto.json"))


def test_columnas_y_dimensiones_del_csv():
    df = pd.read_csv(CSV)
    assert list(df.columns) == NUMERICAS + ESPECIALES + NOMINALES + BINARIAS + [OBJETIVO]
    assert df.shape == (FILAS_CRUDAS, 18)


def test_sin_nulos_y_clases_esperadas():
    df = cargar_datos()
    assert int(df.isna().sum().sum()) == 0
    assert int(df[OBJETIVO].sum()) == POSITIVOS
    assert int((~df[OBJETIVO]).sum()) == NEGATIVOS
    assert float(df[OBJETIVO].mean()) == pytest.approx(0.15474452554744525)


def test_duplicados_exactos_esperados():
    df = pd.read_csv(CSV)
    assert int(df.duplicated().sum()) == DUPLICADOS
    duplicados = df[df.duplicated()]
    assert len(duplicados) == DUPLICADOS
    assert len(df.drop_duplicates()) == FILAS_UNICAS


def test_categorias_esperadas():
    df = cargar_datos()
    assert sorted(df["Month"].unique().tolist()) == [
        "Aug", "Dec", "Feb", "Jul", "June", "Mar", "May", "Nov", "Oct", "Sep",
    ]
    assert sorted(df["VisitorType"].unique().tolist()) == [
        "New_Visitor", "Other", "Returning_Visitor",
    ]
    assert sorted(df["Weekend"].unique().tolist()) == [False, True]
    assert sorted(df[OBJETIVO].unique().tolist()) == [False, True]
    assert sorted(df["SpecialDay"].unique().tolist()) == [
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
    ]


def test_auditoria_coincide_con_el_archivo():
    df = cargar_datos()
    aud = auditar(df)
    esperado = hashlib.sha256(CSV.read_bytes()).hexdigest()
    assert aud["sha256"] == esperado
    assert aud["filas"] == FILAS_CRUDAS
    assert aud["columnas"] == 18
    assert aud["nulos_totales"] == 0
    assert aud["duplicados_exactos"] == DUPLICADOS
    assert aud["clases"] == {"False": NEGATIVOS, "True": POSITIVOS}


def test_hash_del_manifiesto_cuando_existe():
    rutas = _manifiestos()
    if not rutas:
        pytest.skip("No hay manifiesto.json en resultados/")
    esperado = hashlib.sha256(CSV.read_bytes()).hexdigest()
    for ruta in rutas:
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        guardado = manifiesto.get("sha256_datos") or manifiesto.get("sha256_csv")
        assert guardado == esperado, f"hash desactualizado en {ruta}"
