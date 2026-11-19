"""Incertidumbre de las métricas de clasificación en particiones fijas."""

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.model_selection import cross_validate

from .training import pliegues
from .utils import SEMILLA


def intervalos_cv(modelo, X, y):
    """Intervalos t del 95 % entre los pliegues de entrenamiento."""
    medidas = {"AP": "average_precision", "ROC_AUC": "roc_auc", "F1": "f1"}
    scores = cross_validate(modelo, X, y, cv=pliegues(), scoring=medidas, error_score="raise")
    filas = []
    for metrica in medidas:
        valores = scores[f"test_{metrica}"]
        n = len(valores)
        media = valores.mean()
        desviacion = valores.std(ddof=1)
        error = desviacion / np.sqrt(n)
        margen = t.ppf(.975, n - 1) * error
        filas.append(dict(metrica=metrica, n_folds=n, media=media, desviacion=desviacion,
                          error_estandar=error, ic_inferior=media - margen, ic_superior=media + margen))
    return pd.DataFrame(filas)


def replicas(y, scores, umbral, n_boot=2000, semilla=SEMILLA):
    """Remuestrea sesiones por clase. La réplica cero contiene la muestra original.

    El orden fijo de las puntuaciones permite calcular AP y AUC por grupos de
    empates sin volver a ordenar cada muestra. La semilla empareja las sesiones
    entre modelos evaluados sobre la misma partición.
    """
    y = np.asarray(y)
    scores = np.asarray(scores)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if scores.shape != y.shape or not len(pos) or not len(neg) or len(pos) + len(neg) != len(y):
        raise ValueError("Se requieren puntuaciones y etiquetas de ambas clases binarias")
    if n_boot < 1 or y.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Se requieren vectores finitos y al menos una réplica")
    orden = np.argsort(-scores)
    limites = np.r_[np.flatnonzero(np.diff(scores[orden])), len(y) - 1]
    clase = y[orden]
    positivos = scores[orden] >= umbral
    rng = np.random.default_rng(semilla)
    filas = []
    for i in range(n_boot + 1):
        if i:
            indices = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
            pesos = np.bincount(indices, minlength=len(y))[orden]
        else:
            pesos = np.ones(len(y), dtype=int)
        vp = np.cumsum(pesos * clase)[limites]
        fp = np.cumsum(pesos * (1 - clase))[limites]
        precision = np.divide(vp, vp + fp, out=np.zeros(len(vp)), where=vp + fp > 0)
        ap = np.dot(np.diff(vp, prepend=0), precision) / len(pos)
        auc = np.trapezoid(np.r_[0, vp] / len(pos), np.r_[0, fp] / len(neg))
        tp = pesos[positivos] @ clase[positivos]
        falsas = pesos[positivos].sum() - tp
        recall = tp / len(pos)
        especificidad = 1 - falsas / len(neg)
        filas.append((ap, auc, tp / max(tp + falsas, 1), recall, 2 * tp / (len(pos) + tp + falsas),
                      (recall + especificidad) / 2, (tp + len(neg) - falsas) / len(y)))
    return pd.DataFrame(filas, columns=["AP", "ROC_AUC", "precision", "recall", "F1",
                                        "balanced_accuracy", "accuracy"])


def intervalos(muestras):
    """Medias y límites percentiles del 95 % para cada métrica."""
    boot = muestras.iloc[1:]
    return pd.DataFrame({"media_bootstrap": boot.mean(), "ic_inferior": boot.quantile(.025),
                         "ic_superior": boot.quantile(.975)}).rename_axis("metrica").reset_index()


def comparar(a, b):
    """Contraste pareado sobre las mismas sesiones y réplicas."""
    diferencias = (a - b)[["AP", "ROC_AUC"]]
    return intervalos(diferencias).assign(diferencia_puntual=diferencias.iloc[0].to_numpy(),
                                          n_boot=len(diferencias) - 1)
