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

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from shopytics.models import mallas_ampliadas
from shopytics.utils import SEMILLA
from proyecto_ml import metricas, predecir, puntuacion

TOP_K = 3
N_SPLITS = 5


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
