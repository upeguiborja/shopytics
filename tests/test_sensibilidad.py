"""Comprueba la seleccion sin PageValues con una malla pequena."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sensibilidad  # noqa: E402
from proyecto_ml import cargar_datos, particionar  # noqa: E402


def test_busqueda_sin_pagevalues_usa_top_tres_y_no_prueba(tmp_path, monkeypatch):
    _, X, y, itr, iva, _ = particionar(cargar_datos())
    llamadas = []

    def mallas(constructor, rapido):
        modelos, _ = constructor()
        assert "PageValues" not in modelos["Regresion_logistica"].named_steps["pre"].transformers[0][2]
        return modelos, {"Regresion_logistica": {"clf__C": [0.1, 1, 10, 100]}}

    def buscar(nombre, modelo, malla, X_tr, y_tr, n_jobs):
        llamadas.append((len(X_tr), len(y_tr)))
        return pd.DataFrame({
            "modelo": [nombre] * 4,
            "params": [repr({"clf__C": c}) for c in [0.1, 1, 10, 100]],
            "mean_test_score": [0.1, 0.4, 0.3, 0.2],
            "std_test_score": [0.01] * 4,
        })

    monkeypatch.setattr(sensibilidad, "mallas_ampliadas", mallas)
    monkeypatch.setattr(sensibilidad, "buscar_familia", buscar)
    cv, validacion, mejores = sensibilidad.ejecutar_sensibilidad(
        X.iloc[itr], y.iloc[itr], X.iloc[iva], y.iloc[iva], tmp_path, rapido=True, n_jobs=1
    )
    assert llamadas == [(len(itr), len(itr))]
    assert len(cv) == 4
    assert len(validacion) == 3
    assert set(validacion.rango_cv) == {1, 2, 3}
    assert mejores.iloc[0].AP == validacion.AP.max()
    assert (tmp_path / "mejores_sin_pagevalues.csv").is_file()
