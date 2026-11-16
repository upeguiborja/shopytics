"""Pruebas de ablacion: las variables excluidas no llegan al modelo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proyecto_ml import construir_preprocesador, cargar_datos, particionar  # noqa: E402


def _entrenamiento():
    datos = cargar_datos()
    _, X, y, idx_tr, _, _ = particionar(datos)
    return X.loc[idx_tr], y.loc[idx_tr].astype(int)


def _nombres_salida(pre, X):
    pre.fit(X)
    return list(pre.get_feature_names_out())


def test_sin_pagevalues_no_llega_al_modelo():
    X_tr, _ = _entrenamiento()
    nombres = _nombres_salida(construir_preprocesador(incluir_pagevalues=False), X_tr)
    assert not any("PageValues" in n for n in nombres)


def test_con_pagevalues_si_llega_al_modelo():
    X_tr, _ = _entrenamiento()
    nombres = _nombres_salida(construir_preprocesador(incluir_pagevalues=True), X_tr)
    assert any("PageValues" in n for n in nombres)


def test_sin_specialday_no_llega_al_modelo():
    X_tr, _ = _entrenamiento()
    nombres = _nombres_salida(construir_preprocesador(incluir_specialday=False), X_tr)
    assert not any("SpecialDay" in n for n in nombres)


def test_numero_columnas_ablacion():
    X_tr, _ = _entrenamiento()
    con = _nombres_salida(construir_preprocesador(incluir_pagevalues=True), X_tr)
    sin = _nombres_salida(construir_preprocesador(incluir_pagevalues=False), X_tr)
    assert len(con) - len(sin) == 1
