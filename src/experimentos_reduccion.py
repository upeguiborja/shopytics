"""Experimentos independientes de PCA y UMAP sin seleccion sobre prueba.

Uso:
    python src/experimentos_reduccion.py --salida resultados/reduccion [--rapido]
    python src/experimentos_reduccion.py --n-boot 500 --semilla 42

Fase A.1:
- PCA: candidatos explicitos 5, 10, 15, 20, 28, 35 y 45 componentes, mas la
  variante que conserva al menos el 95% de la varianza. Se registra la
  varianza acumulada de cada variante y la curva completa por familia.
- UMAP: las quince combinaciones de 2, 5, 10, 15 y 20 dimensiones por 15, 30 y
  50 vecinos pasan por CV y por validacion; ninguna se descarta antes de la
  validacion.
- La seleccion final es por validacion, con regla de simplicidad: se parte del
  mayor AP de validacion y se sustituye por la variante mas simple cuando su AP
  no baja mas que la desviacion entre folds del ganador (diferencia <= std_CV).
  La frontera es inclusiva.
- Test se consulta solo despues de fijar todas las variantes. Por familia se
  reportan el modelo base sin reduccion, el PCA seleccionado, el UMAP
  seleccionado y el UMAP 2D, con bootstrap IC individual por metrica en
  reduccion_test.csv y diferencias pareadas de AP y ROC_AUC contra la base en
  reduccion_comparacion.csv. Las coordenadas UMAP2D se ajustan solo con train y
  se transforman a train/test.

El modo rapido es una prueba de humo: UMAP (2, 15), PCA (5, 15, 0.95), dos
folds, hasta 600 filas de entrenamiento y bootstrap reducido; sus metricas no
son comparables a la ejecucion normal. La rejilla UMAP completa, con quince
configuraciones por familia en CV y en validacion, es la parte con fuerte coste
de computo.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from proyecto_ml import (
    RAIZ, SEMILLA, TransformadorUMAP, bootstrap_diferencia, bootstrap_ic,
    cargar_datos, construir_modelos, metricas, particionar, predecir, puntuacion,
)

DIMENSIONES_UMAP = (2, 5, 10, 15, 20)
VECINOS_UMAP = (15, 30, 50)
COMPONENTES_PCA = (5, 10, 15, 20, 28, 35, 45, 0.95)
UMAP2D_VECINOS = 15
N_BOOT = 2000


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


def crear_pipeline(modelo, metodo, dimensiones=None, vecinos=None, semilla=SEMILLA):
    base = clone(modelo)
    if metodo == "PCA":
        if dimensiones is None:
            dimensiones = 0.95
        reductor = PCA(n_components=dimensiones, svd_solver="full")
    elif metodo == "UMAP":
        if dimensiones is None:
            raise ValueError("UMAP requiere un numero de dimensiones")
        reductor = TransformadorUMAP(
            n_components=dimensiones,
            n_neighbors=15 if vecinos is None else vecinos,
            min_dist=0.1,
            random_state=semilla,
        )
    else:
        raise ValueError(f"Metodo desconocido: {metodo}")
    return Pipeline([("pre", base.named_steps["pre"]), ("reduccion", reductor),
                     ("clf", base.named_steps["clf"])])


def clave_simplicidad(fila, metodo):
    """Orden ascendente de complejidad; la primera clave es la mas simple."""
    componentes = int(fila["componentes"])
    if metodo == "UMAP":
        return (componentes, -int(fila["vecinos"]))
    solicitada = fila["dimensiones_solicitadas"]
    return (componentes, 0 if isinstance(solicitada, (int, np.integer)) else 1)


def seleccionar_por_validacion(filas, metodo):
    """Elige por AP de validacion con una regla de simplicidad acotada.

    Se parte del mayor AP de validacion. Una variante mas simple lo sustituye
    si su AP no baja mas que la desviacion entre folds del ganador. La frontera
    es inclusiva: diferencia <= std_CV activa la simplicidad. Un std_CV no
    finito se trata como cero, de modo que solo los empates exactos activan la
    regla. En UMAP la complejidad es (n_components, -n_neighbors); en PCA es el
    numero de componentes ajustado, con el valor fijo antes que 0.95.
    """
    if not filas:
        raise ValueError("No hay candidatos para seleccionar")
    ganador = sorted(
        filas, key=lambda f: (-float(f["AP"]), clave_simplicidad(f, metodo))
    )[0]
    desviacion = float(ganador.get("std_CV", 0.0))
    if not np.isfinite(desviacion) or desviacion < 0:
        desviacion = 0.0
    umbral = float(ganador["AP"]) - desviacion
    elegibles = [
        f for f in filas
        if float(f["AP"]) >= umbral
        or np.isclose(float(f["AP"]), umbral, rtol=0.0, atol=1e-12)
    ]
    elegido = min(elegibles, key=lambda f: clave_simplicidad(f, metodo))
    if clave_simplicidad(elegido, metodo) < clave_simplicidad(ganador, metodo):
        motivo = "simplicidad"
    else:
        motivo = "mayor_AP_validacion"
    detalle = {
        "AP_max_validacion": float(ganador["AP"]),
        "desviacion_std_CV": desviacion,
        "umbral_simplicidad": umbral,
        "diferencia_con_max": float(ganador["AP"]) - float(elegido["AP"]),
        "motivo_seleccion": motivo,
    }
    return elegido, detalle


def _n_features_preprocesados(modelo, X_tr, y_tr):
    """Numero de columnas tras el preprocesador; acota los candidatos de PCA."""
    pre = crear_pipeline(modelo, "PCA", 1).named_steps["pre"]
    pre.fit(X_tr, y_tr)
    try:
        return int(len(pre.get_feature_names_out()))
    except AttributeError:
        return int(pre.transform(X_tr).shape[1])


def _cv(pipe, X_tr, y_tr, cv):
    valores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="average_precision",
                              n_jobs=1, error_score="raise")
    return float(np.mean(valores)), float(np.std(valores)), json.dumps(valores.tolist())


def _curva_varianza_pca(pipe, X_tr):
    """Varianza explicada de todos los componentes tras el preprocesador."""
    datos = pipe.named_steps["pre"].transform(X_tr)
    pca = PCA(svd_solver="full").fit(datos)
    return pd.DataFrame({
        "componente": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "varianza_explicada": pca.explained_variance_ratio_,
        "varianza_acumulada": np.cumsum(pca.explained_variance_ratio_),
    })


def _coordenadas_umap2d(pipe, X_tr, y_tr, X_te, y_te, nombre, vecinos):
    """Coordenadas 2D del UMAP ajustado solo con train, transformadas a test."""
    pre = pipe.named_steps["pre"]
    reductor = pipe.named_steps["reduccion"]
    coords_tr = np.asarray(reductor.transform(pre.transform(X_tr)))
    coords_te = np.asarray(reductor.transform(pre.transform(X_te)))
    train = pd.DataFrame({
        "modelo": nombre, "origen": "umap2d", "vecinos": int(vecinos),
        "x": coords_tr[:, 0], "y": coords_tr[:, 1],
        "indice": np.asarray(X_tr.index), "y_true": np.asarray(y_tr),
    })
    test = pd.DataFrame({
        "modelo": nombre, "origen": "umap2d", "vecinos": int(vecinos),
        "x": coords_te[:, 0], "y": coords_te[:, 1],
        "indice": np.asarray(X_te.index), "y_true": np.asarray(y_te),
    })
    return train, test


def _json_nativo(valor):
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        return None if not np.isfinite(valor) else float(valor)
    return valor


def _clave(fila):
    """Identifica una variante de forma estable entre las tablas del barrido."""
    vecinos = fila.get("vecinos")
    if vecinos is None or (isinstance(vecinos, float) and np.isnan(vecinos)):
        vecinos = None
    return (fila["modelo"], fila["metodo"], fila["dimensiones_solicitadas"], vecinos)


def _formato(valor):
    """Escribe un valor de la rejilla sin decimales innecesarios."""
    return str(int(valor)) if float(valor).is_integer() else str(valor)


def ejecutar_experimentos(X_tr, y_tr, X_va, y_va, X_te, y_te, configuraciones,
                          salida, rapido=False, n_boot=N_BOOT, semilla=SEMILLA):
    if n_boot < 1:
        raise ValueError("n_boot debe ser positivo")
    if not configuraciones:
        raise ValueError("Se requiere al menos una configuracion")
    n_boot, semilla = int(n_boot), int(semilla)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    if rapido and len(X_tr) > 600:
        muestra, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=600,
                                                 random_state=semilla).split(X_tr, y_tr))
        X_tr, y_tr = X_tr.iloc[muestra], y_tr.iloc[muestra]
    n_boot_efectivo = min(n_boot, 20) if rapido else n_boot
    cv = StratifiedKFold(n_splits=2 if rapido else 5, shuffle=True, random_state=semilla)
    if rapido:
        dimensiones_umap, vecinos_umap, componentes_pca = (2,), (15,), (5, 15, 0.95)
    else:
        dimensiones_umap = DIMENSIONES_UMAP
        vecinos_umap = VECINOS_UMAP
        componentes_pca = COMPONENTES_PCA
    rejilla_umap = [(d, v) for d in dimensiones_umap for v in vecinos_umap]
    filas_cv, filas_val = [], []
    variantes_test, curvas_varianza, descartados = [], [], []

    for nombre, params, modelo in configuraciones:
        n_features = _n_features_preprocesados(modelo, X_tr, y_tr)
        candidatos_pca = [c for c in componentes_pca if c == 0.95 or c <= n_features]
        descartados.append({
            "modelo": nombre,
            "candidatos_pca": [_formato(c) for c in componentes_pca],
            "candidatos_pca_usados": [_formato(c) for c in candidatos_pca],
            "descartados": [_formato(c) for c in componentes_pca if c not in candidatos_pca],
            "motivo_descarte": "mas componentes que variables tras el preprocesador",
            "n_features_preprocesadas": n_features,
        })

        # Referencia sin reduccion: no compite en la seleccion.
        base = clone(modelo).fit(X_tr, y_tr)
        met_base_va = metricas(y_va, predecir(base, X_va), puntuacion(base, X_va))
        filas_val.append({
            "modelo": nombre, "metodo": "Base",
            "dimensiones_solicitadas": np.nan, "vecinos": np.nan,
            "componentes": np.nan, "AP_CV": np.nan, "std_CV": np.nan,
            "AP_folds": None, "varianza_acumulada": np.nan,
            "seleccionado": False, "motivo_seleccion": "referencia_sin_reduccion",
            **met_base_va,
        })

        for metodo, candidatos in (
            ("PCA", [(c, None) for c in candidatos_pca]),
            ("UMAP", rejilla_umap),
        ):
            filas_metodo = []
            for dimensiones, vecinos in candidatos:
                pipe = crear_pipeline(modelo, metodo, dimensiones, vecinos, semilla)
                media, desv, folds = _cv(pipe, X_tr, y_tr, cv)
                fila_cv = {
                    "modelo": nombre, "metodo": metodo,
                    "dimensiones_solicitadas": dimensiones,
                    "vecinos": vecinos if metodo == "UMAP" else np.nan,
                    "componentes": (int(dimensiones)
                                    if isinstance(dimensiones, (int, np.integer))
                                    else np.nan),
                    "AP_CV": media, "std_CV": desv, "AP_folds": folds,
                    "finalista": False,
                }
                filas_cv.append(fila_cv)
                filas_metodo.append((fila_cv, dimensiones, vecinos))
            # Todas las variantes pasan por validacion antes de mirar la prueba.
            filas_metodo_val = []
            for fila_cv, dimensiones, vecinos in filas_metodo:
                fila_cv["finalista"] = True
                pipe = crear_pipeline(modelo, metodo, dimensiones, vecinos,
                                      semilla).fit(X_tr, y_tr)
                met = metricas(y_va, predecir(pipe, X_va), puntuacion(pipe, X_va))
                reductor = pipe.named_steps["reduccion"]
                if metodo == "PCA":
                    componentes = int(reductor.n_components_)
                    fila_cv["componentes"] = componentes
                    varianza = float(np.sum(reductor.explained_variance_ratio_))
                else:
                    componentes = int(dimensiones)
                    varianza = np.nan
                fila = {
                    "modelo": nombre, "metodo": metodo,
                    "dimensiones_solicitadas": dimensiones,
                    "vecinos": vecinos if metodo == "UMAP" else np.nan,
                    "componentes": componentes,
                    "AP_CV": fila_cv["AP_CV"], "std_CV": fila_cv["std_CV"],
                    "AP_folds": fila_cv["AP_folds"],
                    "varianza_acumulada": varianza,
                    "seleccionado": False,
                    **met,
                }
                filas_val.append(fila)
                filas_metodo_val.append((fila, pipe))
            elegido, detalle = seleccionar_por_validacion(
                [fila for fila, _ in filas_metodo_val], metodo
            )
            elegido["seleccionado"] = True
            elegido.update(detalle)
            pipe_elegido = next(p for f, p in filas_metodo_val if f is elegido)
            variantes_test.append({
                "modelo": nombre, "metodo": metodo,
                "dimensiones_solicitadas": elegido["dimensiones_solicitadas"],
                "vecinos": elegido["vecinos"], "componentes": elegido["componentes"],
                "AP_CV": float(elegido["AP_CV"]), "std_CV": float(elegido["std_CV"]),
                "AP_validacion": float(elegido["AP"]),
                "varianza_acumulada": elegido["varianza_acumulada"],
                "motivo_seleccion": detalle["motivo_seleccion"],
                "diferencia_con_max": detalle["diferencia_con_max"],
                "umbral_simplicidad": detalle["umbral_simplicidad"],
                "desviacion_std_CV": detalle["desviacion_std_CV"],
                "AP_max_validacion": detalle["AP_max_validacion"],
                "pipe": pipe_elegido,
            })
            if metodo == "PCA":
                pipe_95 = next(
                    (p for f, p in filas_metodo_val
                     if f["dimensiones_solicitadas"] == 0.95),
                    None,
                )
                if pipe_95 is not None:
                    curva = _curva_varianza_pca(pipe_95, X_tr)
                    curva.insert(0, "modelo", nombre)
                    curvas_varianza.append(curva)

        # Variante UMAP 2D por familia: referencia de perdida de senal.
        pipe_2d = crear_pipeline(modelo, "UMAP", 2, UMAP2D_VECINOS, semilla)
        pipe_2d.fit(X_tr, y_tr)
        met_2d_va = metricas(y_va, predecir(pipe_2d, X_va), puntuacion(pipe_2d, X_va))
        filas_val.append({
            "modelo": nombre, "metodo": "UMAP2D",
            "dimensiones_solicitadas": 2, "vecinos": UMAP2D_VECINOS,
            "componentes": 2, "AP_CV": np.nan, "std_CV": np.nan,
            "AP_folds": None, "varianza_acumulada": np.nan,
            "seleccionado": False, "motivo_seleccion": "referencia_umap_2d",
            **met_2d_va,
        })
        variantes_test.append({
            "modelo": nombre, "metodo": "UMAP2D",
            "dimensiones_solicitadas": 2, "vecinos": UMAP2D_VECINOS,
            "componentes": 2, "AP_CV": np.nan, "std_CV": np.nan,
            "AP_validacion": float(met_2d_va["AP"]),
            "varianza_acumulada": np.nan, "pipe": pipe_2d,
        })
        variantes_test.append({
            "modelo": nombre, "metodo": "Base",
            "dimensiones_solicitadas": np.nan, "vecinos": np.nan,
            "componentes": np.nan, "AP_CV": np.nan, "std_CV": np.nan,
            "AP_validacion": float(met_base_va["AP"]),
            "varianza_acumulada": np.nan, "pipe": base,
        })

    # No se consulta prueba hasta haber fijado todas las variantes.
    filas_test, comparaciones = [], []
    scores_por_familia = {}
    for variante in variantes_test:
        pipe = variante["pipe"]
        y_pred = predecir(pipe, X_te)
        y_score = np.asarray(puntuacion(pipe, X_te))
        met = metricas(y_te, y_pred, y_score)
        ic = bootstrap_ic(y_te, y_pred, y_score, n_boot=n_boot_efectivo,
                          semilla=semilla).set_index("metrica")
        fila = {
            "modelo": variante["modelo"], "metodo": variante["metodo"],
            "dimensiones_solicitadas": variante["dimensiones_solicitadas"],
            "vecinos": variante["vecinos"], "componentes": variante["componentes"],
            "AP_CV": variante["AP_CV"], "std_CV": variante["std_CV"],
            "AP_validacion": variante["AP_validacion"],
            "varianza_acumulada": variante["varianza_acumulada"],
            "n_boot": n_boot_efectivo,
            **met,
        }
        for nombre_metrica in met:
            fila[f"media_bootstrap_{nombre_metrica}"] = float(
                ic.loc[nombre_metrica, "media_bootstrap"]
            )
            fila[f"ic_{nombre_metrica}_inferior"] = float(
                ic.loc[nombre_metrica, "ic_inferior"]
            )
            fila[f"ic_{nombre_metrica}_superior"] = float(
                ic.loc[nombre_metrica, "ic_superior"]
            )
        filas_test.append(fila)
        scores_por_familia[(variante["modelo"], variante["metodo"])] = y_score

    # Diferencias pareadas de AP y ROC_AUC contra la base de cada familia.
    for (modelo, metodo), score in scores_por_familia.items():
        if metodo == "Base":
            continue
        score_base = scores_por_familia[(modelo, "Base")]
        variante = next(v for v in variantes_test
                        if v["modelo"] == modelo and v["metodo"] == metodo)
        for nombre_metrica, funcion in (("AP", average_precision_score),
                                        ("ROC_AUC", roc_auc_score)):
            puntual, media, inferior, superior = bootstrap_diferencia(
                y_te, score, score_base, funcion, n_boot=n_boot_efectivo,
                semilla=semilla,
            )
            comparaciones.append({
                "modelo": modelo, "metodo": metodo,
                "dimensiones_solicitadas": variante["dimensiones_solicitadas"],
                "vecinos": variante["vecinos"],
                "componentes": variante["componentes"],
                "metrica": nombre_metrica,
                "valor": float(funcion(y_te, score)),
                "valor_base": float(funcion(y_te, score_base)),
                "diferencia_puntual": puntual, "media_bootstrap": media,
                "ic_inferior": inferior, "ic_superior": superior,
                "n_boot": n_boot_efectivo,
            })

    coordenadas_train, coordenadas_test, por_familia = [], [], {}
    for variante in variantes_test:
        if variante["metodo"] != "UMAP2D":
            continue
        train, test = _coordenadas_umap2d(
            variante["pipe"], X_tr, y_tr, X_te, y_te, variante["modelo"],
            UMAP2D_VECINOS,
        )
        coordenadas_train.append(train)
        coordenadas_test.append(test)
        por_familia[variante["modelo"]] = {
            "train": train[["x", "y"]].to_numpy().tolist(),
            "test": test[["x", "y"]].to_numpy().tolist(),
            "y_train": train["y_true"].astype(int).tolist(),
            "y_test": test["y_true"].astype(int).tolist(),
        }

    pd.DataFrame(filas_cv).to_csv(salida / "reduccion_cv.csv", index=False)
    pd.DataFrame(filas_val).to_csv(salida / "reduccion_validacion.csv", index=False)
    pd.DataFrame(filas_test).to_csv(salida / "reduccion_test.csv", index=False)
    columnas_comp = ["modelo", "metodo", "dimensiones_solicitadas", "vecinos",
                     "componentes", "metrica", "valor", "valor_base",
                     "diferencia_puntual", "media_bootstrap", "ic_inferior",
                     "ic_superior", "n_boot"]
    pd.DataFrame(comparaciones, columns=columnas_comp).to_csv(
        salida / "reduccion_comparacion.csv", index=False
    )

    por_clave = {_clave(f): f for f in filas_val}
    barrido_pca, barrido_umap = [], []
    for c in filas_cv:
        f = por_clave.get(_clave(c))
        if c["metodo"] == "PCA":
            barrido_pca.append({
                "modelo": c["modelo"], "componentes": int(c["componentes"]),
                "dimensiones_solicitadas": c["dimensiones_solicitadas"],
                "AP_validacion": float(f["AP"]) if f else np.nan,
                "AP_CV": float(c["AP_CV"]), "std_CV": float(c["std_CV"]),
                "varianza_acumulada": f["varianza_acumulada"] if f else np.nan,
                "seleccionado": bool(f["seleccionado"]) if f else False,
            })
        else:
            barrido_umap.append({
                "modelo": c["modelo"], "dimensiones": int(c["componentes"]),
                "vecinos": int(c["vecinos"]),
                "AP_validacion": float(f["AP"]) if f else np.nan,
                "AP_CV": float(c["AP_CV"]), "std_CV": float(c["std_CV"]),
                "seleccionado": bool(f["seleccionado"]) if f else False,
            })
    pd.DataFrame(barrido_pca).to_csv(salida / "pca_barrido.csv", index=False)
    pd.DataFrame(barrido_umap).to_csv(salida / "umap_barrido.csv", index=False)

    columnas_var = ["modelo", "componente", "varianza_explicada", "varianza_acumulada"]
    tabla_varianza = (pd.concat(curvas_varianza, ignore_index=True)
                      if curvas_varianza else pd.DataFrame(columns=columnas_var))
    # Un archivo por familia: la grafica traza una curva por modelo.
    tabla_varianza.to_csv(salida / "pca_varianza_familias.csv", index=False)
    tabla_varianza.to_csv(salida / "pca_varianza.csv", index=False)
    if coordenadas_train:
        pd.concat(coordenadas_train, ignore_index=True).to_csv(
            salida / "umap2d_train.csv", index=False
        )
        pd.concat(coordenadas_test, ignore_index=True).to_csv(
            salida / "umap2d_test.csv", index=False
        )
        ap_umap2d = {v["modelo"]: v["AP_validacion"] for v in variantes_test
                     if v["metodo"] == "UMAP2D"}
        mejor = max(ap_umap2d, key=ap_umap2d.get)
        (salida / "umap_2d.json").write_text(
            json.dumps({
                "modelo": mejor,
                "train": por_familia[mejor]["train"],
                "test": por_familia[mejor]["test"],
                "y_train": por_familia[mejor]["y_train"],
                "y_test": por_familia[mejor]["y_test"],
                "por_familia": por_familia,
                "nota": ("UMAP 2D con 15 vecinos, ajustado solo con train y "
                         "transformado a test; las claves de primer nivel "
                         "corresponden a la familia con mayor AP de validacion"),
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    seleccion = [
        {
            "modelo": v["modelo"], "metodo": v["metodo"],
            "dimensiones_solicitadas": _json_nativo(v["dimensiones_solicitadas"]),
            "componentes": _json_nativo(v["componentes"]),
            "vecinos": _json_nativo(v["vecinos"]),
            "AP_CV": _json_nativo(v["AP_CV"]),
            "std_CV": _json_nativo(v["std_CV"]),
            "AP_validacion": float(v["AP_validacion"]),
            "varianza_acumulada": _json_nativo(v["varianza_acumulada"]),
            "motivo_seleccion": v["motivo_seleccion"],
            "diferencia_con_max": float(v["diferencia_con_max"]),
            "umbral_simplicidad": float(v["umbral_simplicidad"]),
            "desviacion_std_CV": float(v["desviacion_std_CV"]),
            "AP_max_validacion": float(v["AP_max_validacion"]),
        }
        for v in variantes_test
        if v["metodo"] in ("PCA", "UMAP")
    ]

    protocolo = {
        "rapido": rapido,
        "semilla": semilla,
        "n_boot": n_boot_efectivo,
        "n_entrenamiento": int(len(X_tr)),
        "n_validacion": int(len(X_va)),
        "n_test": int(len(X_te)),
        "n_folds": int(cv.n_splits),
        "seleccion_familias": "dos mayores AP de validacion del CSV; empate por nombre",
        "candidatos_pca": [_formato(c) for c in componentes_pca],
        "candidatos_pca_por_familia": descartados,
        "rejilla_umap": [{"dimensiones": int(d), "vecinos": int(v)}
                         for d, v in rejilla_umap],
        "umap_validacion": ("las quince configuraciones pasan por CV y por "
                            "validacion; ninguna se descarta antes de la "
                            "validacion"),
        "pca": (f"barrido {', '.join(_formato(c) for c in componentes_pca)}; CV y "
                "validacion para todas las variantes; varianza acumulada por "
                "variante y curva de varianza por familia"),
        "umap": (f"rejilla {len(dimensiones_umap)}x{len(vecinos_umap)}; CV y "
                 "validacion para todas las configuraciones; seleccion final por "
                 "AP de validacion"),
        "regla_simplicidad": {
            "criterio": ("se parte del mayor AP de validacion; una variante mas "
                         "simple lo sustituye si su AP_validacion >= AP_max - "
                         "std_CV del ganador"),
            "desviacion": ("std_CV poblacional entre folds del candidato con mayor "
                           "AP de validacion; un valor no finito se trata como 0"),
            "frontera": ("la igualdad cuenta como dentro del umbral (diferencia <= "
                         "std_CV); con std_CV=0 solo los empates exactos activan "
                         "la simplicidad"),
            "complejidad_umap": "menor n_components; a igualdad, mayor n_neighbors",
            "complejidad_pca": ("menor numero de componentes ajustado; a igualdad, "
                                "valor fijo antes que 0.95"),
            "empates": ("orden determinista por la clave de simplicidad y luego por "
                        "el orden de la rejilla"),
        },
        "variantes_test": {
            "Base": "modelo de la familia sin reduccion",
            "PCA": "variante de PCA seleccionada en validacion",
            "UMAP": "variante de UMAP seleccionada en validacion",
            "UMAP2D": ("UMAP con 2 dimensiones y 15 vecinos por familia, sin "
                       "seleccion"),
        },
        "prueba": ("test se consulta solo despues de fijar todas las variantes; "
                   "no participa en la seleccion"),
        "bootstrap": ("estratificado por clase, condicionado a las puntuaciones "
                      "fijas; los IC individuales de cada metrica van como columnas "
                      "en reduccion_test.csv"),
        "comparaciones": ("diferencias pareadas de AP y ROC_AUC de cada variante "
                          "reducida contra la base de su familia, con IC bootstrap, "
                          "en reduccion_comparacion.csv"),
        "varianza_pca": ("pca_varianza.csv y pca_varianza_familias.csv contienen la "
                         "curva de varianza acumulada de PCA de cada familia; "
                         "pca_barrido.csv es el barrido de componentes y la varianza "
                         "acumulada de cada variante aparece en validacion y test"),
        "coordenadas_umap2d": ("umap2d_train.csv y umap2d_test.csv por familia; UMAP "
                               "2D con 15 vecinos ajustado solo con train y "
                               "transformado a test; umap_2d.json repite las "
                               "coordenadas en listas"),
        "salidas": {
            "reduccion_cv.csv": ("AP de CV por variante: modelo, metodo, "
                                 "dimensiones_solicitadas, vecinos, componentes, "
                                 "AP_CV, std_CV, AP_folds, finalista"),
            "reduccion_validacion.csv": ("AP y metricas de validacion de todas las "
                                         "variantes, con varianza acumulada, "
                                         "seleccionado y motivo de la regla de "
                                         "simplicidad"),
            "reduccion_test.csv": ("base, PCA seleccionado, UMAP seleccionado y "
                                   "UMAP 2D por familia, con bootstrap IC "
                                   "individual por metrica en columnas"),
            "reduccion_comparacion.csv": ("diferencias pareadas de AP y ROC_AUC "
                                          "contra la base: modelo, metodo, metrica, "
                                          "valor, valor_base, diferencia_puntual, "
                                          "media_bootstrap, ic_inferior, ic_superior"),
            "pca_barrido.csv": ("barrido de PCA: componentes y AP de validacion "
                                "por familia"),
            "umap_barrido.csv": ("barrido de UMAP: dimensiones, vecinos y AP de "
                                 "validacion por familia"),
            "pca_varianza.csv": ("curva de varianza acumulada de PCA por familia; "
                                 "mismo contenido que pca_varianza_familias.csv"),
            "pca_varianza_familias.csv": ("curva de varianza acumulada de PCA de "
                                          "todas las familias"),
            "umap2d_train.csv": "coordenadas UMAP2D de train por familia",
            "umap2d_test.csv": "coordenadas UMAP2D de test por familia",
            "umap_2d.json": ("coordenadas UMAP2D de train y test en listas, por "
                             "familia"),
        },
        "computo": {
            "rejilla_umap": (f"{len(dimensiones_umap)}x{len(vecinos_umap)} = "
                             f"{len(rejilla_umap)} configuraciones por familia"),
            "folds": int(cv.n_splits),
            "ajustes_por_familia": {
                "pca": (f"{len(candidatos_pca)} candidatos x {cv.n_splits} folds de "
                        f"CV + {len(candidatos_pca)} ajustes sobre train"),
                "umap": (f"{len(rejilla_umap)} configuraciones x {cv.n_splits} folds "
                         f"de CV + {len(rejilla_umap)} ajustes sobre train"),
                "umap2d": "1 ajuste sobre train por familia",
            },
            "estimacion": ("la rejilla UMAP completa es la parte con fuerte coste "
                           "de computo: quince configuraciones por familia con CV "
                           "y validacion; no se ejecuta en modo rapido"),
            "modo_rapido": ("UMAP (2, 15), PCA (5, 15, 0.95), 2 folds, hasta 600 "
                            "filas de train y bootstrap <= 20; solo prueba de humo"),
        },
        "limitacion": ("la particion de prueba fue consultada historicamente; los IC "
                       "son condicionales a las variantes fijadas"),
        "pca_95": [
            {"modelo": f["modelo"], "componentes": int(f["componentes"]),
             "varianza_acumulada": _json_nativo(f["varianza_acumulada"])}
            for f in filas_val
            if f["metodo"] == "PCA" and f["dimensiones_solicitadas"] == 0.95
        ],
        "configuraciones": [{"modelo": nombre, "params": repr(params)}
                            for nombre, params, _ in configuraciones],
        "seleccion": seleccion,
    }
    (salida / "reduccion_protocolo.json").write_text(
        json.dumps(protocolo, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return protocolo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuraciones", type=Path,
                        default=RAIZ / "resultados" / "mejores_configuraciones.csv")
    parser.add_argument("--salida", type=Path, default=RAIZ / "resultados" / "reduccion")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--rapido", action="store_true")
    args = parser.parse_args()
    configuraciones = cargar_configuraciones(args.configuraciones)
    _, X, y, tr, va, te = particionar(cargar_datos())
    ejecutar_experimentos(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va],
                          X.iloc[te], y.iloc[te], configuraciones, args.salida,
                          args.rapido, args.n_boot, args.semilla)
    print(f"Resultados en {args.salida}")


if __name__ == "__main__":
    main()
