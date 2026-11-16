"""Experimentos independientes de PCA y UMAP sin seleccion sobre prueba.

Uso: python src/experimentos_reduccion.py --salida resultados/reduccion [--rapido]
Las dos familias y sus parametros se leen de mejores_configuraciones.csv y se
ordenan por AP de validacion. PCA conserva al menos el 95% de la varianza en
cada ajuste. Para UMAP se comparan 2, 5 y 10 dimensiones por AP en CV de
entrenamiento; las dos mejores pasan a validacion, que elige la dimension.
En empates se prefiere la menor dimension. En modo rapido se usa solo UMAP 2,
dos folds y una muestra estratificada de hasta 600 filas de entrenamiento;
es una prueba de humo y sus metricas no son comparables a la ejecucion normal.
El preprocesador, reductor y clasificador siempre se ajustan juntos en un
Pipeline. Test se evalua solo para las variantes fijadas por este protocolo.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from proyecto_ml import (
    RAIZ, SEMILLA, TransformadorUMAP, cargar_datos, construir_modelos,
    metricas, particionar, predecir, puntuacion,
)


def cargar_configuraciones(ruta):
    tabla = pd.read_csv(ruta)
    requeridas = {"modelo", "AP", "params"}
    if not requeridas.issubset(tabla.columns):
        raise ValueError(f"Faltan columnas: {sorted(requeridas - set(tabla.columns))}")
    if tabla["modelo"].isna().any() or tabla["modelo"].duplicated().any():
        raise ValueError("Debe haber exactamente una configuracion por familia")
    tabla["AP"] = pd.to_numeric(tabla["AP"], errors="raise")
    if not np.isfinite(tabla["AP"]).all() or len(tabla) < 2:
        raise ValueError("Se requieren al menos dos familias con AP de validacion finita")
    tabla = tabla.sort_values(["AP", "modelo"], ascending=[False, True]).head(2)
    modelos, _ = construir_modelos()
    seleccion = []
    for fila in tabla.itertuples(index=False):
        if fila.modelo not in modelos:
            raise ValueError(f"Familia desconocida: {fila.modelo}")
        try:
            params = ast.literal_eval(fila.params)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Parametros invalidos para {fila.modelo}") from exc
        if not isinstance(params, dict) or not all(
            isinstance(k, str) and k.startswith(("pre__", "clf__")) for k in params
        ):
            raise ValueError(f"Parametros invalidos para {fila.modelo}")
        modelo = modelos[fila.modelo].set_params(**params)
        seleccion.append((fila.modelo, params, modelo))
    return seleccion


def crear_pipeline(modelo, metodo, dimensiones=None):
    base = clone(modelo)
    if metodo == "PCA":
        reductor = PCA(n_components=0.95, svd_solver="full")
    elif metodo == "UMAP":
        reductor = TransformadorUMAP(n_components=dimensiones, n_neighbors=15,
                                    min_dist=0.1, random_state=SEMILLA)
    else:
        raise ValueError(f"Metodo desconocido: {metodo}")
    return Pipeline([("pre", base.named_steps["pre"]), ("reduccion", reductor),
                     ("clf", base.named_steps["clf"])])


def ejecutar_experimentos(X_tr, y_tr, X_va, y_va, X_te, y_te, configuraciones,
                         salida, rapido=False):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    if rapido and len(X_tr) > 600:
        muestra, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=600,
                                                 random_state=SEMILLA).split(X_tr, y_tr))
        X_tr, y_tr = X_tr.iloc[muestra], y_tr.iloc[muestra]
    cv = StratifiedKFold(n_splits=2 if rapido else 5, shuffle=True, random_state=SEMILLA)
    dimensiones = [2] if rapido else [2, 5, 10]
    filas_cv, filas_val, finales = [], [], []

    for nombre, params, modelo in configuraciones:
        for metodo, candidatos in [("PCA", [None]), ("UMAP", dimensiones)]:
            scores = []
            for dim in candidatos:
                pipe = crear_pipeline(modelo, metodo, dim)
                valores = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                          scoring="average_precision", n_jobs=1,
                                          error_score="raise")
                scores.append((float(np.mean(valores)), dim))
                filas_cv.append({"modelo": nombre, "metodo": metodo,
                                 "dimensiones_solicitadas": dim, "AP_CV": np.mean(valores),
                                 "std_CV": np.std(valores), "AP_folds": json.dumps(valores.tolist())})
            scores.sort(key=lambda par: (-par[0], par[1] or 0))
            evaluados = []
            for ap_cv, dim in scores[:2]:
                pipe = crear_pipeline(modelo, metodo, dim).fit(X_tr, y_tr)
                met = metricas(y_va, predecir(pipe, X_va), puntuacion(pipe, X_va))
                n_componentes = (int(pipe.named_steps["reduccion"].n_components_)
                                 if metodo == "PCA" else dim)
                fila = {"modelo": nombre, "metodo": metodo,
                        "dimensiones_solicitadas": dim, "componentes": n_componentes,
                        "AP_CV": ap_cv, **met}
                filas_val.append(fila)
                evaluados.append((fila, pipe))
            evaluados.sort(key=lambda item: (-item[0]["AP"], item[0]["componentes"]))
            fila, pipe = evaluados[0]
            fila["seleccionado"] = True
            finales.append((fila, pipe, params))

    # No se consulta prueba hasta haber fijado todos los modelos y dimensiones.
    filas_test = []
    for fila, pipe, params in finales:
        met = metricas(y_te, predecir(pipe, X_te), puntuacion(pipe, X_te))
        filas_test.append({"modelo": fila["modelo"], "metodo": fila["metodo"],
                           "componentes": fila["componentes"], **met})
    pd.DataFrame(filas_cv).to_csv(salida / "reduccion_cv.csv", index=False)
    pd.DataFrame(filas_val).fillna({"seleccionado": False}).to_csv(
        salida / "reduccion_validacion.csv", index=False)
    pd.DataFrame(filas_test).to_csv(salida / "reduccion_test.csv", index=False)
    resumen = {
        "rapido": rapido, "n_entrenamiento": len(X_tr), "n_folds": cv.n_splits,
        "seleccion_familias": "dos mayores AP de validacion del CSV; empate por nombre",
        "pca": "minimo numero de componentes que conserva al menos 95% de varianza en cada ajuste",
        "umap": "2, 5 y 10 dimensiones; dos mejores por AP-CV, seleccion por AP de validacion; empate menor dimension" if not rapido else "solo 2 dimensiones en modo rapido",
        "prueba": "solo reporte de PCA y UMAP seleccionados por familia, sin decisiones",
        "configuraciones": [{"modelo": nombre, "params": repr(params)}
                            for nombre, params, _ in configuraciones],
        "seleccion": [{"modelo": fila["modelo"], "metodo": fila["metodo"],
                       "componentes": fila["componentes"], "AP_validacion": fila["AP"]}
                      for fila, _, _ in finales],
    }
    (salida / "reduccion_protocolo.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuraciones", type=Path,
                        default=RAIZ / "resultados" / "mejores_configuraciones.csv")
    parser.add_argument("--salida", type=Path, default=RAIZ / "resultados" / "reduccion")
    parser.add_argument("--rapido", action="store_true")
    args = parser.parse_args()
    configuraciones = cargar_configuraciones(args.configuraciones)
    _, X, y, tr, va, te = particionar(cargar_datos())
    ejecutar_experimentos(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va],
                          X.iloc[te], y.iloc[te], configuraciones, args.salida, args.rapido)
    print(f"Resultados en {args.salida}")


if __name__ == "__main__":
    main()
