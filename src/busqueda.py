"""Busqueda en rejilla con validacion cruzada (solo entrenamiento).

Regla uniforme para las cinco familias:
  1. GridSearchCV 5-fold estratificado sobre el conjunto de entrenamiento,
     metrica AP (average precision).
  2. Las 3 mejores configuraciones por AP-CV pasan a validacion.
  3. Se elige la configuracion con mayor AP en validacion.
  4. El conjunto de prueba no interviene en ninguna decision.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from proyecto_ml import SEMILLA, metricas, predecir, puntuacion

TOP_K = 3
N_SPLITS = 5


def mallas_ampliadas(construir_modelos_fn, rapido: bool = False):
    """Devuelve (modelos, mallas) con rejillas ampliadas.

    En modo rapido se reduce el numero de valores para pruebas de humo.
    """
    modelos, mallas = construir_modelos_fn()
    if rapido:
        mallas["RandomForest"] = {
            "pre__num__log__usar_log": [True],
            "clf__max_depth": [12],
            "clf__min_samples_leaf": [1],
            "clf__max_features": ["sqrt"],
            "clf__class_weight": [None],
        }
        mallas["MLP"] = {
            "pre__num__log__usar_log": [True],
            "clf__hidden_layer_sizes": [(64,)],
            "clf__alpha": [1e-4],
            "clf__learning_rate_init": [0.01],
        }
        mallas["SVM"] = [
            {"pre__num__log__usar_log": [True], "clf__kernel": ["linear"],
             "clf__C": [1], "clf__class_weight": [None]},
            {"pre__num__log__usar_log": [True], "clf__kernel": ["rbf"],
             "clf__C": [10], "clf__gamma": [0.01], "clf__class_weight": [None]},
        ]
        mallas["kNN"] = {
            "pre__num__log__usar_log": [True],
            "clf__n_neighbors": [61],
            "clf__weights": ["distance"],
            "clf__p": [2],
        }
        mallas["Regresion_logistica"] = {
            "pre__num__log__usar_log": [True],
            "clf__C": [0.1],
            "clf__class_weight": [None],
        }
        return modelos, mallas

    # RF: log(1+x) es monotona y no altera particiones de arboles; no se
    # explora usar_log para evitar candidatos duplicados en el top-k.
    mallas["RandomForest"] = {
        "pre__num__log__usar_log": [True],
        "clf__max_depth": [4, 6, 8, 10, 12, 20, None],
        "clf__max_features": ["log2", "sqrt", 0.3, 0.5, 0.7],
        "clf__min_samples_leaf": [1, 5, 10, 20],
        "clf__class_weight": [None, "balanced"],
    }
    mallas["MLP"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
        "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "clf__learning_rate_init": [0.001, 0.01, 0.03, 0.1, 0.3],
    }
    mallas["SVM"] = [
        {
            "pre__num__log__usar_log": [True, False],
            "clf__kernel": ["linear"],
            "clf__C": [0.01, 0.03, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "pre__num__log__usar_log": [True, False],
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1, 10, 30, 100],
            "clf__gamma": ["scale", 0.001, 0.01, 0.03, 0.1],
            "clf__class_weight": [None, "balanced"],
        },
    ]
    return modelos, mallas


def buscar_familia(nombre, modelo, malla, X_tr, y_tr, n_jobs=-1):
    """GridSearch con CV estratificada; devuelve el DataFrame completo de resultados."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEMILLA)
    t0 = time.time()
    gs = GridSearchCV(
        modelo, malla, scoring="average_precision", cv=cv,
        n_jobs=n_jobs, return_train_score=True, refit=False,
    )
    gs.fit(X_tr, y_tr)
    df = pd.DataFrame(gs.cv_results_)
    df["modelo"] = nombre
    df["tiempo_total_seg"] = time.time() - t0
    df["params"] = df["params"].apply(lambda d: repr(d))
    return df


def seleccionar_top_k(cv_df: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """Las k mejores filas por AP-CV de una familia."""
    return cv_df.sort_values("mean_test_score", ascending=False).head(k).reset_index(drop=True)


def evaluar_candidatos_validacion(
    nombre, modelo_base, top_df, X_tr, y_tr, X_va, y_va, parametros_por_fila
):
    """Ajusta cada candidato del top-k sobre train y evalua en validacion."""
    filas = []
    for i, row in top_df.iterrows():
        params = parametros_por_fila[i]
        m = clone(modelo_base).set_params(**params)
        m.fit(X_tr, y_tr)
        yp = predecir(m, X_va)
        ys = puntuacion(m, X_va)
        d = metricas(y_va, yp, ys)
        d.update({
            "modelo": nombre,
            "rango_cv": int(i) + 1,
            "AP_CV": float(row["mean_test_score"]),
            "std_CV": float(row["std_test_score"]),
            "params": repr(params),
        })
        filas.append(d)
    return pd.DataFrame(filas)
