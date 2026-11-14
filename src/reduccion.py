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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proyecto_ml import (  # noqa: E402
    CSV,
    SEMILLA,
    TransformadorUMAP,
    bootstrap_ic,
    cargar_datos,
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

RAIZ = Path(__file__).resolve().parent.parent
OUT = RAIZ / "resultados_corregidos"
OUT.mkdir(parents=True, exist_ok=True)

# Hiperparametros ganadores de la seccion 1 (tab:mejores del informe).
PARAMS = {
    "RandomForest": dict(
        max_depth=12,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight=None,
        n_estimators=300,
        random_state=SEMILLA,
        n_jobs=1,
    ),
    "MLP": dict(
        hidden_layer_sizes=(64,),
        alpha=0.0001,
        learning_rate_init=0.01,
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEMILLA,
    ),
}


def clasificador(nombre):
    if nombre == "RandomForest":
        return RandomForestClassifier(**PARAMS[nombre])
    return MLPClassifier(**PARAMS[nombre])


def evaluar_modelo(nombre, modelo, X, y, itr, iva, ite):
    t0 = time.time()
    modelo.fit(X.iloc[itr], y.iloc[itr])
    tiempo = time.time() - t0
    filas = []
    for particion, ix in [("validacion", iva), ("test", ite)]:
        yt = y.iloc[ix].to_numpy()
        yp = predecir(modelo, X.iloc[ix])
        ys = puntuacion(modelo, X.iloc[ix])
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


def main():
    df = cargar_datos(CSV)
    _, X, y, itr, iva, ite = particionar(df)

    # Numero de caracteristicas tras el preprocesamiento (referencia para el % de reduccion).
    pre_ref = construir_preprocesador(usar_log=True)
    pre_ref.fit(X.iloc[itr], y.iloc[itr])
    n_original = pre_ref.transform(X.iloc[:5]).shape[1]
    print(f"Caracteristicas tras preprocesamiento: {n_original}")

    filas = []
    componentes = {}

    # PCA con varianza explicada del 95 %.
    for nombre in ["RandomForest", "MLP"]:
        pipe = Pipeline(
            [
                ("pre", construir_preprocesador(usar_log=True)),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
                ("clf", clasificador(nombre)),
            ]
        )
        filas += evaluar_modelo(f"PCA-{nombre}", pipe, X, y, itr, iva, ite)
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
        filas += evaluar_modelo(f"UMAP10-{nombre}", pipe, X, y, itr, iva, ite)
        componentes[f"UMAP10-{nombre}"] = 10
        print(f"UMAP10-{nombre}: 10 componentes")

    tabla = pd.DataFrame(filas)
    tabla["reduccion_pct"] = tabla["modelo"].map(
        lambda m: 100 * (1 - componentes[m] / n_original)
    )
    tabla.to_csv(OUT / "reduccion_dimension.csv", index=False)

    # Comparacion con la linea base (resultados_corregidos/metricas_particiones.csv).
    base = pd.read_csv(OUT / "metricas_particiones.csv")
    base = (
        base[base.particion.isin(["validacion", "test"])][
            ["modelo", "particion", "AP", "ROC_AUC", "F1"]
        ]
        .rename(
            columns={
                "modelo": "modelo_base",
                "AP": "AP_base",
                "ROC_AUC": "ROC_AUC_base",
                "F1": "F1_base",
            }
        )
    )
    comp = tabla.copy()
    comp["modelo_base"] = comp["modelo"].str.split("-").str[-1]
    comp = comp.merge(base, on=["modelo_base", "particion"], how="left")
    comp["delta_AP"] = comp["AP"] - comp["AP_base"]
    comp["delta_ROC_AUC"] = comp["ROC_AUC"] - comp["ROC_AUC_base"]
    comp["delta_F1"] = comp["F1"] - comp["F1_base"]
    comp.to_csv(OUT / "reduccion_comparacion.csv", index=False)

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
        filas_2d += evaluar_modelo(f"UMAP2-{nombre}", pipe, X, y, itr, iva, ite)
    pd.DataFrame(filas_2d).to_csv(OUT / "reduccion_umap2.csv", index=False)

    print(tabla.to_string(index=False))
    print("Resultados en", OUT)


if __name__ == "__main__":
    main()
