"""Entrenamiento, selección por AP y ablación sobre particiones ya definidas."""

import json

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from .models import construir_modelos
from .utils import SEMILLA


def pliegues():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


def buscar_configuraciones(modelo, malla, X, y, n_jobs=1):
    """Devuelve todos los candidatos en el orden de GridSearchCV."""
    busqueda = GridSearchCV(
        modelo, malla, scoring="average_precision", cv=pliegues(), n_jobs=n_jobs,
        return_train_score=True, refit=False, error_score="raise",
    ).fit(X, y)
    tabla = pd.DataFrame(busqueda.cv_results_)
    columnas = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    columnas += [f"split{i}_{parte}_score" for parte in ("test", "train") for i in range(5)]
    return tabla[columnas]


def evaluar(modelo, X, y):
    if hasattr(modelo, "predict_proba"):
        score = modelo.predict_proba(X)[:, 1]
        pred = score >= 0.5
    else:
        score = modelo.decision_function(X)
        pred = score >= 0
    return {
        "AP": average_precision_score(y, score), "ROC_AUC": roc_auc_score(y, score),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "accuracy": accuracy_score(y, pred),
    }


def validar_finalistas(modelo, candidatos, X_tr, y_tr, X_val, y_val):
    filas = []
    top = candidatos.sort_values("mean_test_score", ascending=False).head(3)
    for rango, fila in enumerate(top.itertuples(), start=1):
        ajustado = clone(modelo).set_params(**fila.params).fit(X_tr, y_tr)
        filas.append({
            "rango_cv": rango, "params": fila.params, "AP_CV": fila.mean_test_score,
            "std_CV": fila.std_test_score, **evaluar(ajustado, X_val, y_val),
        })
    return pd.DataFrame(filas)


def evaluar_ablacion(configuraciones, variantes, X_tr, y_tr, X_val, y_val):
    filas = []
    for familia, params in configuraciones.items():
        for nombre, excluidas in variantes.items():
            modelo = construir_modelos(excluir=excluidas)[familia].set_params(**params)
            scores = cross_val_score(modelo, X_tr, y_tr, cv=pliegues(), scoring="average_precision",
                                     n_jobs=1, error_score="raise")
            modelo.fit(X_tr, y_tr)
            filas.append({
                "modelo": familia, "variante": nombre, "excluidas": excluidas,
                "AP_folds": scores.tolist(), "AP_CV": scores.mean(), "std_CV": scores.std(),
                **evaluar(modelo, X_val, y_val),
            })
    tabla = pd.DataFrame(filas)
    base = tabla[tabla.variante == "completo"].set_index("modelo").AP
    tabla["delta_AP"] = tabla.AP - tabla.modelo.map(base)
    return tabla


def guardar_registros(tabla, ruta):
    ruta.write_text(json.dumps(tabla.to_dict("records"), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")
