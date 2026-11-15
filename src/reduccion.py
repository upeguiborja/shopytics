"""Analisis de reduccion de dimension (PCA y UMAP) para el Entregable II.

Ajusta PCA y UMAP dentro del Pipeline, solo con el conjunto de entrenamiento,
sobre la matriz preprocesada (las variables numericas ya quedan estandarizadas
por el preprocesador). Compara Random Forest y MLP con y sin reduccion y guarda
las tablas en resultados_corregidos/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proyecto_ml import (  # noqa: E402
    CSV,
    SEMILLA,
    TransformadorUMAP,
    bootstrap_diferencia,
    bootstrap_ic,
    cargar_datos,
    construir_modelos,
    construir_preprocesador,
    metricas,
    particionar,
    predecir,
    puntuacion,
)
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from configuracion_final import PARAMS  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
OUT = RAIZ / "resultados_corregidos"
def clasificador(nombre):
    params = {k.removeprefix("clf__"): v for k, v in PARAMS[nombre].items()
              if k.startswith("clf__")}
    return (RandomForestClassifier if nombre == "RandomForest" else MLPClassifier)(**params)


def evaluar_modelo(nombre, modelo, X, y, itr, iva, ite, puntuaciones):
    t0 = time.time()
    modelo.fit(X.iloc[itr], y.iloc[itr])
    tiempo = time.time() - t0
    filas = []
    for particion, ix in [("validacion", iva), ("test", ite)]:
        yt = y.iloc[ix].to_numpy()
        yp = predecir(modelo, X.iloc[ix])
        ys = puntuacion(modelo, X.iloc[ix])
        puntuaciones[(nombre, particion)] = ys
        d = metricas(yt, yp, ys)
        fila = {
            "modelo": nombre,
            "particion": particion,
            "AP": d["AP"],
            "ROC_AUC": d["ROC_AUC"],
            "F1": d["F1"],
            "precision": d["precision"],
            "recall": d["recall"],
            "balanced_accuracy": d["balanced_accuracy"],
            "tiempo_seg": round(tiempo, 2),
        }
        ic = bootstrap_ic(yt, yp, ys)
        for _, r in ic.iterrows():
            fila[f"ic_{r['metrica']}_inferior"] = r["ic_inferior"]
            fila[f"ic_{r['metrica']}_superior"] = r["ic_superior"]
        filas.append(fila)
    return filas


def main(resultados=None):
    global OUT
    if resultados is not None:
        OUT = Path(resultados)
    OUT.mkdir(parents=True, exist_ok=True)
    df = cargar_datos(CSV)
    _, X, y, itr, iva, ite = particionar(df)

    # Numero de caracteristicas tras el preprocesamiento (referencia para el % de reduccion).
    pre_ref = construir_preprocesador(usar_log=True)
    pre_ref.fit(X.iloc[itr], y.iloc[itr])
    n_original = pre_ref.transform(X.iloc[:5]).shape[1]
    print(f"Caracteristicas tras preprocesamiento: {n_original}")

    filas = []
    componentes = {}
    puntuaciones = {}

    # Reajustar las bases para comparar sobre exactamente las mismas observaciones.
    modelos_base, _ = construir_modelos()
    base_ref = pd.read_csv(OUT / "metricas_particiones.csv")
    for nombre in ["RandomForest", "MLP"]:
        modelo = modelos_base[nombre].set_params(**PARAMS[nombre])
        modelo.fit(X.iloc[itr], y.iloc[itr])
        for particion, ix in [("validacion", iva), ("test", ite)]:
            score = puntuacion(modelo, X.iloc[ix])
            puntuaciones[(nombre, particion)] = score
            ref = base_ref[(base_ref.modelo == nombre) & (base_ref.particion == particion)]
            if len(ref) != 1:
                raise ValueError(f"Falta base en metricas_particiones.csv: {nombre}, {particion}")
            for metrica, fn in [("AP", average_precision_score), ("ROC_AUC", roc_auc_score)]:
                if not np.isclose(fn(y.iloc[ix], score), ref.iloc[0][metrica], atol=5e-4, rtol=0):
                    raise ValueError(f"{metrica} base no coincide: {nombre}, {particion}")

    # PCA con varianza explicada del 95 %.
    for nombre in ["RandomForest", "MLP"]:
        pipe = Pipeline(
            [
                ("pre", construir_preprocesador(usar_log=True)),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
                ("clf", clasificador(nombre)),
            ]
        )
        pipe.set_params(**{k: v for k, v in PARAMS[nombre].items() if k.startswith("pre__")})
        filas += evaluar_modelo(f"PCA-{nombre}", pipe, X, y, itr, iva, ite, puntuaciones)
        pca = pipe.named_steps["pca"]
        componentes[f"PCA-{nombre}"] = int(pca.n_components_)
        if nombre == "RandomForest":
            varianza = pd.DataFrame(
                {
                    "componente": range(1, len(pca.explained_variance_ratio_) + 1),
                    "varianza_explicada": pca.explained_variance_ratio_,
                    "varianza_acumulada": pca.explained_variance_ratio_.cumsum(),
                }
            )
            varianza.to_csv(OUT / "pca_varianza.csv", index=False)
        print(f"PCA-{nombre}: {pca.n_components_} componentes")

    # UMAP de 10 componentes para modelado.
    for nombre in ["RandomForest", "MLP"]:
        pipe = Pipeline(
            [
                ("pre", construir_preprocesador(usar_log=True)),
                (
                    "umap",
                    TransformadorUMAP(
                        n_components=10, n_neighbors=15, min_dist=0.1, random_state=SEMILLA
                    ),
                ),
                ("clf", clasificador(nombre)),
            ]
        )
        pipe.set_params(**{k: v for k, v in PARAMS[nombre].items() if k.startswith("pre__")})
        filas += evaluar_modelo(f"UMAP10-{nombre}", pipe, X, y, itr, iva, ite, puntuaciones)
        componentes[f"UMAP10-{nombre}"] = 10
        print(f"UMAP10-{nombre}: 10 componentes")

    tabla = pd.DataFrame(filas)
    tabla["reduccion_pct"] = tabla["modelo"].map(
        lambda m: 100 * (1 - componentes[m] / n_original)
    )
    tabla.to_csv(OUT / "reduccion_dimension.csv", index=False)

    # UMAP 2D para visualizacion.
    pre = construir_preprocesador(usar_log=True)
    Xtr = pre.fit_transform(X.iloc[itr], y.iloc[itr])
    Xva = pre.transform(X.iloc[iva])
    Xte = pre.transform(X.iloc[ite])
    umap2 = TransformadorUMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEMILLA
    )
    umap2.fit(Xtr)
    coords = {
        "train": umap2.transform(Xtr).tolist(),
        "validacion": umap2.transform(Xva).tolist(),
        "test": umap2.transform(Xte).tolist(),
        "y_train": y.iloc[itr].tolist(),
        "y_validacion": y.iloc[iva].tolist(),
        "y_test": y.iloc[ite].tolist(),
    }
    (OUT / "umap_2d.json").write_text(json.dumps(coords), encoding="utf-8")

    # UMAP 2D con clasificador, como referencia de perdida de senal.
    filas_2d = []
    for nombre in ["RandomForest", "MLP"]:
        pipe = Pipeline(
            [
                ("pre", construir_preprocesador(usar_log=True)),
                (
                    "umap",
                    TransformadorUMAP(
                        n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEMILLA
                    ),
                ),
                ("clf", clasificador(nombre)),
            ]
        )
        pipe.set_params(**{k: v for k, v in PARAMS[nombre].items() if k.startswith("pre__")})
        filas_2d += evaluar_modelo(f"UMAP2-{nombre}", pipe, X, y, itr, iva, ite, puntuaciones)
    tabla_2d = pd.DataFrame(filas_2d)
    tabla_2d.to_csv(OUT / "reduccion_umap2.csv", index=False)

    tabla_2d["reduccion_pct"] = 100 * (1 - 2 / n_original)
    comp = pd.concat([tabla, tabla_2d], ignore_index=True)
    base = base_ref[base_ref.particion.isin(["validacion", "test"])][
        ["modelo", "particion", "AP", "ROC_AUC", "F1"]
    ].rename(columns={"modelo": "modelo_base", "AP": "AP_base",
                      "ROC_AUC": "ROC_AUC_base", "F1": "F1_base"})
    comp["modelo_base"] = comp["modelo"].str.split("-").str[-1]
    comp = comp.merge(base, on=["modelo_base", "particion"], how="left", validate="many_to_one")
    for metrica in ["AP", "ROC_AUC", "F1"]:
        comp[f"delta_{metrica}"] = comp[metrica] - comp[f"{metrica}_base"]
    indices = {"validacion": iva, "test": ite}
    for i, fila in comp.iterrows():
        yt = y.iloc[indices[fila.particion]].to_numpy()
        for metrica, fn in [("AP", average_precision_score), ("ROC_AUC", roc_auc_score)]:
            puntual, media, inferior, superior = bootstrap_diferencia(
                yt, puntuaciones[(fila.modelo, fila.particion)],
                puntuaciones[(fila.modelo_base, fila.particion)], fn)
            comp.loc[i, f"delta_{metrica}"] = puntual
            comp.loc[i, f"delta_{metrica}_media_bootstrap"] = media
            comp.loc[i, f"delta_{metrica}_ic_inferior"] = inferior
            comp.loc[i, f"delta_{metrica}_ic_superior"] = superior
    comp.to_csv(OUT / "reduccion_comparacion.csv", index=False)

    print(tabla.to_string(index=False))
    print("Resultados en", OUT)


if __name__ == "__main__":
    main()
