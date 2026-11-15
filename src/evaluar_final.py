"""Reproduce configuraciones finales y reutiliza cuatro ablaciones archivadas.

La prueba historica ya fue consultada: ordenar esta ejecucion no recupera su
independencia. Los IC bootstrap son condicionales a modelos, predicciones y
umbrales fijos; no incluyen incertidumbre de entrenamiento ni de seleccion.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from configuracion_final import CV_HISTORICO, PARAMS
from proyecto_ml import (
    RAIZ, SEMILLA, auditar, bootstrap_diferencia, bootstrap_ic, cargar_datos,
    construir_modelos, contar_combinaciones, metricas, particionar, predecir,
    puntuacion, resumen_particiones, tabla_auditoria,
)

ORDEN = tuple(PARAMS)
PARTICIONES = ("entrenamiento", "validacion", "test")


def _json(ruta, datos, indent=2):
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=indent), encoding="utf-8")


def _comprobar(archivo, nombre, valor, esperado):
    diferencia = abs(round(float(valor), 4) - esperado)
    if not np.isfinite(diferencia) or diferencia > 5e-4:
        raise ValueError(f"{archivo}: {nombre} difiere {diferencia:.6g} (obtenido={valor}, esperado={esperado})")


def ejecutar(resultados=RAIZ / "resultados_corregidos", ablaciones=RAIZ / "resultados"):
    salida, archivo = Path(resultados), Path(ablaciones)
    if salida.resolve() == archivo.resolve():
        raise ValueError("La salida no puede sobrescribir las ablaciones historicas")
    salida.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    df_u, X, y, itr, iva, ite = particionar(df)
    indices = dict(zip(PARTICIONES, (itr, iva, ite)))
    _json(salida / "auditoria.json", auditar(df))
    tabla_auditoria(df).to_csv(salida / "tabla_variables.csv", index=False)
    resumen_particiones(y, itr, iva, ite).to_csv(salida / "resumen_particiones.csv", index=False)
    mes = df_u.iloc[itr].groupby("Month", observed=True)["Revenue"].agg(["size", "mean"])
    mes.columns = ["sesiones", "tasa_compra"]
    mes.to_csv(salida / "cobertura_mensual_train.csv")

    modelos, mallas = construir_modelos()
    for nombre in ORDEN:
        modelos[nombre].set_params(**PARAMS[nombre]).fit(X.iloc[itr], y.iloc[itr])
        print(f"Ajustado {nombre}", flush=True)

    filas, predicciones, scores = {}, {}, {}
    for nombre in ORDEN:
        for part in PARTICIONES[:2]:
            ix = indices[part]
            predicciones[nombre, part] = predecir(modelos[nombre], X.iloc[ix])
            scores[nombre, part] = puntuacion(modelos[nombre], X.iloc[ix])
            d = metricas(y.iloc[ix], predicciones[nombre, part], scores[nombre, part])
            filas[nombre, part] = {**d, "modelo": nombre, "particion": part}

    ganador = max(ORDEN, key=lambda m: filas[m, "validacion"]["AP"])
    _json(salida / "ganador.json", {
        "modelo": ganador, "criterio": "mayor AP en validacion; la prueba no participa",
        "AP_validacion": filas[ganador, "validacion"]["AP"],
    })
    umbrales, curvas = {}, {}
    for nombre in ORDEN:
        sv = scores[nombre, "validacion"]
        p, r, thr = precision_recall_curve(y.iloc[iva], sv)
        f1 = 2 * p * r / (p + r + 1e-12)
        umbrales[nombre] = float(thr[np.argmax(f1[:-1])])
        fpr, tpr, _ = roc_curve(y.iloc[iva], sv)
        curvas[nombre] = {"pr_precision": p.tolist(), "pr_recall": r.tolist(),
                          "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist()}
    _json(salida / "curvas_validacion.json", curvas, indent=None)

    # Validar ANTES de sustituir los valores archivados de kNN o consultar test.
    comparaciones = {}
    for fichero, columna in (("comparacion_pagevalues.csv", "AP_con_PageValues"),
                             ("comparacion_specialday.csv", "AP_con_SpecialDay")):
        tabla = pd.read_csv(archivo / fichero)
        if len(tabla) != len(ORDEN) or set(tabla.modelo) != set(ORDEN):
            raise ValueError(f"{archivo / fichero}: modelos ausentes o duplicados")
        for nombre in ORDEN:
            if nombre == "kNN":
                continue
            fila = tabla.loc[tabla.modelo == nombre].iloc[0]
            for col, metrica in [(columna, "AP"), ("ROC_AUC_con", "ROC_AUC")]:
                if col in tabla:
                    diferencia = abs(float(fila[col]) - filas[nombre, "validacion"][metrica])
                    if not np.isfinite(diferencia) or diferencia > 5e-4:
                        raise ValueError(f"{archivo / fichero}: {nombre} {col} difiere {diferencia:.6g}")
        comparaciones[fichero] = tabla

    modelos_sp, mallas_sp = construir_modelos(incluir_pagevalues=False)
    gs = GridSearchCV(modelos_sp["kNN"], mallas_sp["kNN"], scoring="average_precision",
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA), n_jobs=-1)
    gs.fit(X.iloc[itr], y.iloc[itr])
    knn_sp = metricas(y.iloc[iva], predecir(gs, X.iloc[iva]), puntuacion(gs, X.iloc[iva]))
    modelos_sd, _ = construir_modelos(incluir_specialday=False)
    knn_sd = modelos_sd["kNN"].set_params(**PARAMS["kNN"]).fit(X.iloc[itr], y.iloc[itr])
    knn_sd_met = metricas(y.iloc[iva], predecir(knn_sd, X.iloc[iva]), puntuacion(knn_sd, X.iloc[iva]))
    print(f"kNN sin PageValues: {gs.best_params_}; AP={knn_sp['AP']:.6f}", flush=True)

    for nombre in ORDEN:
        predicciones[nombre, "test"] = predecir(modelos[nombre], X.iloc[ite])
        scores[nombre, "test"] = puntuacion(modelos[nombre], X.iloc[ite])
        d = metricas(y.iloc[ite], predicciones[nombre, "test"], scores[nombre, "test"])
        filas[nombre, "test"] = {**d, "modelo": nombre, "particion": "test"}
    met = pd.DataFrame([filas[m, part] for m in ORDEN for part in PARTICIONES])
    _comprobar("metricas_particiones.csv", "RF test AP", filas["RandomForest", "test"]["AP"], 0.7486)
    _comprobar("metricas_particiones.csv", "MLP test AP", filas["MLP", "test"]["AP"], 0.7377)
    met.to_csv(salida / "metricas_particiones.csv", index=False)

    filas_ic = []
    for nombre in ORDEN:
        for part in PARTICIONES:
            ic = bootstrap_ic(y.iloc[indices[part]], predicciones[nombre, part], scores[nombre, part])
            ic["modelo"], ic["particion"] = nombre, part
            filas_ic.append(ic[["modelo", "particion", "metrica", "media_bootstrap", "ic_inferior", "ic_superior"]])
        print(f"IC calculados: {nombre}", flush=True)
    pd.concat(filas_ic, ignore_index=True).to_csv(salida / "metricas_intervalos_confianza.csv", index=False)
    _json(salida / "matrices_confusion_test.json", {
        m: confusion_matrix(y.iloc[ite], predicciones[m, "test"], labels=[0, 1]).tolist() for m in ORDEN
    }, indent=None)

    filas_umbral = []
    for nombre in ORDEN:
        st = scores[nombre, "test"]
        ajustada = (st >= umbrales[nombre]).astype(int)
        d = metricas(y.iloc[ite], ajustada, st)
        fila = {"modelo": nombre, "umbral_validacion": umbrales[nombre],
                "F1_nativo": filas[nombre, "test"]["F1"], "F1_ajustado": d["F1"],
                "recall_ajustado": d["recall"], "precision_ajustado": d["precision"],
                "bal_acc_ajustada": d["balanced_accuracy"]}
        ic = bootstrap_ic(y.iloc[ite], ajustada, st).set_index("metrica")
        for metrica, columna in (("F1", "F1_ajustado"), ("recall", "recall_ajustado"),
                                 ("precision", "precision_ajustado"), ("balanced_accuracy", "bal_acc_ajustada")):
            for extremo in ("ic_inferior", "ic_superior"):
                fila[f"ic_{columna}_{extremo.removeprefix('ic_')}"] = ic.loc[metrica, extremo]
        filas_umbral.append(fila)
    _comprobar("umbrales.csv", "umbral RF", umbrales["RandomForest"], 0.3078)
    _comprobar("umbrales.csv", "F1 ajustado RF", filas_umbral[0]["F1_ajustado"], 0.6691)
    pd.DataFrame(filas_umbral).to_csv(salida / "umbrales.csv", index=False)

    pares = [("RandomForest", "MLP"), ("RandomForest", "SVM"), ("RandomForest", "kNN"),
             ("RandomForest", "Regresion_logistica"), ("MLP", "SVM")]
    filas_pares = []
    for a, b in pares:
        for nombre, fn in (("AP", average_precision_score), ("ROC_AUC", roc_auc_score)):
            puntual, media, inferior, superior = bootstrap_diferencia(
                y.iloc[ite].to_numpy(), scores[a, "test"], scores[b, "test"], fn
            )
            filas_pares.append({"comparacion": f"{a} - {b}", "metrica": nombre,
                                "diferencia_puntual": puntual, "media_bootstrap": media,
                                "ic_inferior": inferior, "ic_superior": superior})
    par_rf = filas_pares[0]
    # En el archivo historico `diferencia` era la media bootstrap, no el valor puntual.
    for campo, esperado in (("media_bootstrap", 0.0104), ("ic_inferior", -0.0100), ("ic_superior", 0.0321)):
        _comprobar("comparaciones_pareadas.csv", f"RF - MLP AP {campo}", par_rf[campo], esperado)
    pd.DataFrame(filas_pares).to_csv(salida / "comparaciones_pareadas.csv", index=False)

    for fichero, con, sin, sin_val in (
        ("comparacion_pagevalues.csv", "AP_con_PageValues", "AP_sin_PageValues", knn_sp),
        ("comparacion_specialday.csv", "AP_con_SpecialDay", "AP_sin_SpecialDay", knn_sd_met),
    ):
        tabla = comparaciones[fichero]
        ix = tabla.modelo == "kNN"
        tabla.loc[ix, con] = filas["kNN", "validacion"]["AP"]
        tabla.loc[ix, sin] = sin_val["AP"]
        tabla.loc[ix, "delta_AP"] = filas["kNN", "validacion"]["AP"] - sin_val["AP"]
        if "ROC_AUC_con" in tabla:
            tabla.loc[ix, "ROC_AUC_con"] = filas["kNN", "validacion"]["ROC_AUC"]
            tabla.loc[ix, "ROC_AUC_sin"] = sin_val["ROC_AUC"]
        tabla.to_csv(salida / fichero, index=False)
    print("Las ablaciones de las otras cuatro familias son archivadas, no regeneradas.", flush=True)

    pd.DataFrame([{"modelo": m, "AP_CV": CV_HISTORICO[m][0], "std_CV": CV_HISTORICO[m][1]}
                  for m in ORDEN]).to_csv(salida / "mejores_configuraciones.csv", index=False)
    contar_combinaciones(mallas).to_csv(salida / "combinaciones_malla.csv", index=False)
    sd = X["SpecialDay"].to_numpy()
    ordinal = np.rint(sd / 0.2)
    correlacion = float(np.corrcoef((sd - sd.mean()) / sd.std(),
                                    (ordinal - ordinal.mean()) / ordinal.std())[0, 1])
    _json(salida / "specialday_equivalencia.json", {
        "valores_unicos": sorted(float(v) for v in np.unique(sd)),
        "correlacion_representaciones_escaladas": correlacion,
        "conclusion": "La codificacion ordinal es afin a la numerica; no se compararon otras codificaciones.",
    })
    try:
        version_umap = metadata.version("umap-learn")
    except metadata.PackageNotFoundError:
        version_umap = "no instalado"
    print(f"Python {platform.python_version()}; scikit-learn {sklearn.__version__}; "
          f"numpy {np.__version__}; pandas {pd.__version__}; umap-learn {version_umap}")
    print(f"Salida: {salida.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resultados", type=Path, default=RAIZ / "resultados_corregidos")
    parser.add_argument("--ablaciones", type=Path, default=RAIZ / "resultados")
    args = parser.parse_args(argv)
    ejecutar(resultados=args.resultados, ablaciones=args.ablaciones)


if __name__ == "__main__":
    main()
