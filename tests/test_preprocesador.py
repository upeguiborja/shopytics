"""Pruebas focalizadas del preprocesador configurable."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proyecto_ml import (  # noqa: E402
    construir_modelos,
    construir_preprocesador,
    cargar_datos,
    particionar,
)


def _particiones():
    _, X, y, idx_tr, idx_va, _ = particionar(cargar_datos())
    return X.loc[idx_tr], y.loc[idx_tr], X.loc[idx_va]


def _nombres(pre, X):
    pre.fit(X)
    return list(pre.get_feature_names_out())


def test_excluir_quita_columnas_y_convive_con_banderas():
    X_tr, _, _ = _particiones()
    nombres = _nombres(
        construir_preprocesador(excluir=["PageValues", "SpecialDay"]), X_tr
    )
    assert not any("PageValues" in n for n in nombres)
    assert not any("SpecialDay" in n for n in nombres)

    por_banderas = _nombres(
        construir_preprocesador(incluir_pagevalues=False, incluir_specialday=False),
        X_tr,
    )
    assert nombres == por_banderas


def test_excluir_acepta_string_y_rechaza_desconocidas():
    X_tr, _, _ = _particiones()
    nombres = _nombres(construir_preprocesador(excluir="PageValues"), X_tr)
    assert not any("PageValues" in n for n in nombres)
    with pytest.raises(ValueError):
        construir_preprocesador(excluir=["NoExiste"])


def test_onehot_specialday_genera_seis_columnas():
    X_tr, _, _ = _particiones()
    onehot = _nombres(construir_preprocesador(codificar_specialday="onehot"), X_tr)
    esp = [n for n in onehot if "SpecialDay" in n]
    assert len(esp) == 6
    assert all(n.startswith("cat__SpecialDay_") for n in esp)

    numerico = _nombres(construir_preprocesador(codificar_specialday="numerico"), X_tr)
    assert [n for n in numerico if "SpecialDay" in n] == ["esp__SpecialDay"]


def test_onehot_aprende_niveles_solo_con_train():
    X_tr, _, X_va = _particiones()
    pre = construir_preprocesador(codificar_specialday="onehot")
    muestra = X_tr.iloc[:300]
    pre.fit(muestra)
    nombres = list(pre.transformers_[1][2])
    encoder = pre.named_transformers_["cat"].named_steps["onehot"]
    i = nombres.index("SpecialDay")
    assert list(encoder.categories_[i]) == sorted(muestra["SpecialDay"].unique())
    transformado = pre.transform(X_va.iloc[:20])
    assert transformado.shape[1] == len(pre.get_feature_names_out())


def test_construir_modelos_propaga_exclusiones():
    modelos, _ = construir_modelos(
        codificar_specialday="onehot", excluir=["PageValues", "SpecialDay"]
    )
    for modelo in modelos.values():
        columnas = [
            col
            for _, _, cols in modelo.named_steps["pre"].transformers
            for col in (cols if isinstance(cols, list) else [])
        ]
        assert "PageValues" not in columnas
        assert "SpecialDay" not in columnas


def test_sin_predictoras_falla():
    with pytest.raises(ValueError):
        construir_preprocesador(
            excluir=[
                "Administrative",
                "Administrative_Duration",
                "Informational",
                "Informational_Duration",
                "ProductRelated",
                "ProductRelated_Duration",
                "BounceRates",
                "ExitRates",
                "PageValues",
                "Month",
                "OperatingSystems",
                "Browser",
                "Region",
                "TrafficType",
                "VisitorType",
                "Weekend",
                "SpecialDay",
            ]
        )


def test_codificacion_invalida_falla():
    with pytest.raises(ValueError):
        construir_preprocesador(codificar_specialday="ordinal")
