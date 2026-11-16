"""Repite la seleccion de las cinco familias sin PageValues.

La malla y la regla top-3 son las mismas que en la seleccion principal;
la prueba no se usa para buscar ni elegir configuraciones.
"""

from __future__ import annotations

import argparse
import ast
from functools import partial
from pathlib import Path

import pandas as pd

from busqueda import buscar_familia, evaluar_candidatos_validacion, mallas_ampliadas, seleccionar_top_k
from proyecto_ml import RAIZ, cargar_datos, construir_modelos, particionar


def comparar_pagevalues(ruta_base, ruta_sin, salida):
    base = pd.read_csv(ruta_base)[["modelo", "AP", "ROC_AUC"]]
    sin = pd.read_csv(ruta_sin)[["modelo", "AP", "ROC_AUC", "params"]]
    comparacion = base.merge(sin, on="modelo", suffixes=("_con", "_sin"),
                             validate="one_to_one")
    if len(comparacion) != 5 or len(base) != 5 or len(sin) != 5:
        raise ValueError("Las configuraciones no cubren las cinco familias")
    comparacion["delta_AP"] = comparacion.AP_con - comparacion.AP_sin
    comparacion.to_csv(salida, index=False)
    return comparacion


def ejecutar_sensibilidad(X_tr, y_tr, X_va, y_va, salida: Path, rapido=False, n_jobs=-1):
    """Busca con CV en train y elige entre el top-3 con AP de validacion."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    modelos, mallas = mallas_ampliadas(
        partial(construir_modelos, incluir_pagevalues=False), rapido=rapido
    )
    cv_todos, validacion_todos = [], []
    for nombre, malla in mallas.items():
        print(f"Busqueda sin PageValues: {nombre}", flush=True)
        cv = buscar_familia(nombre, modelos[nombre], malla, X_tr, y_tr, n_jobs=n_jobs)
        top = seleccionar_top_k(cv)
        parametros = {i: ast.literal_eval(p) for i, p in zip(top.index, top["params"])}
        validacion = evaluar_candidatos_validacion(
            nombre, modelos[nombre], top, X_tr, y_tr, X_va, y_va, parametros
        )
        cv_todos.append(cv)
        validacion_todos.append(validacion)

    cv_total = pd.concat(cv_todos, ignore_index=True)
    validacion_total = pd.concat(validacion_todos, ignore_index=True)
    mejores = (validacion_total.sort_values("AP", ascending=False)
               .groupby("modelo", as_index=False).head(1)
               .sort_values("AP", ascending=False).reset_index(drop=True))
    cv_total.to_csv(salida / "cv_sin_pagevalues.csv", index=False)
    validacion_total.to_csv(salida / "validacion_top3_sin_pagevalues.csv", index=False)
    mejores.to_csv(salida / "mejores_sin_pagevalues.csv", index=False)
    ruta_base = salida / "mejores_configuraciones.csv"
    if ruta_base.exists():
        comparar_pagevalues(ruta_base, salida / "mejores_sin_pagevalues.csv",
                            salida / "comparacion_pagevalues.csv")
    return cv_total, validacion_total, mejores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=RAIZ / "resultados")
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    _, X, y, itr, iva, _ = particionar(cargar_datos())
    ejecutar_sensibilidad(X.iloc[itr], y.iloc[itr], X.iloc[iva], y.iloc[iva],
                          args.salida, rapido=args.rapido, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
