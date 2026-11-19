"""Utilidades reproducibles para la seccion 1 del Entregable II.

Prediccion de la intencion de compra (Online Shoppers Purchasing Intention,
UCI #468). Todo el preprocesamiento se ajusta dentro de los folds de
entrenamiento mediante Pipeline/ColumnTransformer.
"""

from __future__ import annotations

import ast
import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from shopytics.utils import (
    BINARIAS, ESPECIALES, LOG_VARS, NOMINALES, NUMERICAS, PREDICTORAS, SEMILLA, OBJETIVO,
)
from shopytics.search import modelos_y_mallas as construir_modelos
from shopytics.models import (
    TransformadorLog, construir_preprocesador, orden_familias,
)

RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "data" / "online_shoppers_intention.csv"

DIR_RES = RAIZ / "resultados"


# ---------------------------------------------------------------------------
# Auditoria
# ---------------------------------------------------------------------------
def cargar_datos(ruta: str | Path = CSV) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df = df.copy()
    for c in ["Weekend", "Revenue"]:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.upper().map(
                {"TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
            )
    return df


def auditar(df: pd.DataFrame, ruta_csv: str | Path = CSV) -> dict:
    ruta_csv = Path(ruta_csv)
    h = hashlib.sha256(ruta_csv.read_bytes()).hexdigest()
    aud = {
        "archivo": ruta_csv.name,
        "sha256": h,
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nulos_totales": int(df.isna().sum().sum()),
        "duplicados_exactos": int(df.duplicated().sum()),
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "faltantes_por_columna": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "clases": {str(k): int(v) for k, v in df[OBJETIVO].value_counts().sort_index().items()},
        "proporcion_compra": float(df[OBJETIVO].mean()),
        "categorias": {
            c: {str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()}
            for c in NOMINALES + BINARIAS + ESPECIALES
        },
        "rango_numericas": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "media": float(df[c].mean()),
                "desv": float(df[c].std()),
                "ceros": int((df[c] == 0).sum()),
                "asimetria": float(df[c].skew()),
            }
            for c in NUMERICAS + ESPECIALES
        },
        "valores_especialday": sorted(float(v) for v in df["SpecialDay"].unique()),
        "version_python": platform.python_version(),
        "version_sklearn": sklearn.__version__,
        "semilla": SEMILLA,
    }
    return aud


def tabla_auditoria(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c in PREDICTORAS + [OBJETIVO]:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            filas.append(
                {
                    "variable": c,
                    "tipo": "numerica",
                    "n_unicos": int(s.nunique()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "media": float(s.mean()),
                    "nulos": int(s.isna().sum()),
                }
            )
        else:
            vc = s.value_counts()
            filas.append(
                {
                    "variable": c,
                    "tipo": "categorica",
                    "n_unicos": int(s.nunique()),
                    "min": np.nan,
                    "max": np.nan,
                    "media": np.nan,
                    "nulos": int(s.isna().sum()),
                    "categoria_mas_frecuente": str(vc.index[0]),
                    "frecuencia_max": int(vc.iloc[0]),
                }
            )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Particiones
# ---------------------------------------------------------------------------
def particionar(df: pd.DataFrame):
    """Particion estratificada 60/20/20 sobre datos sin duplicados exactos."""
    df_u = df.drop_duplicates().reset_index(drop=True)
    X = df_u[PREDICTORAS].copy()
    y = df_u[OBJETIVO].astype(int).copy()
    idx = np.arange(len(df_u))
    idx_tr, idx_resto = train_test_split(
        idx, test_size=0.40, stratify=y, random_state=SEMILLA
    )
    idx_val, idx_te = train_test_split(
        idx_resto, test_size=0.50, stratify=y.iloc[idx_resto], random_state=SEMILLA
    )
    return df_u, X, y, idx_tr, idx_val, idx_te


def resumen_particiones(y: pd.Series, idx_tr, idx_val, idx_te) -> pd.DataFrame:
    filas = []
    for nombre, ix in [
        ("Entrenamiento", idx_tr),
        ("Validacion", idx_val),
        ("Test", idx_te),
    ]:
        yy = y.iloc[ix]
        filas.append(
            {
                "particion": nombre,
                "n": int(len(ix)),
                "positivos": int(yy.sum()),
                "negativos": int(len(yy) - yy.sum()),
                "proporcion_compra": float(yy.mean()),
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
class TransformadorUMAP(BaseEstimator, TransformerMixin):
    """Envoltorio de UMAP compatible con Pipeline; se ajusta solo con train."""

    def __init__(self, n_components=10, n_neighbors=15, min_dist=0.1, random_state=SEMILLA):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    def fit(self, X, y=None):
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "Instale umap-learn para usar TransformadorUMAP: uv sync --locked"
            ) from exc
        self.modelo_ = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
            compatibility_layout=True,
        )
        self.modelo_.fit(X)
        return self

    def transform(self, X):
        return self.modelo_.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"umap_{i}" for i in range(self.n_components)], dtype=object)


MAX_RAPIDO = 600


def _submuestra_rapida(X_tr, y_tr, semilla=SEMILLA):
    """Submuestra estratificada de entrenamiento para el modo rapido."""
    if len(X_tr) <= MAX_RAPIDO:
        return X_tr, y_tr
    indices, _ = train_test_split(
        np.arange(len(X_tr)),
        train_size=MAX_RAPIDO,
        stratify=np.asarray(y_tr),
        random_state=semilla,
    )
    return X_tr.iloc[indices], y_tr.iloc[indices]


def leer_configuraciones(ruta) -> dict[str, dict]:
    """Lee un CSV con una configuracion por familia (columnas modelo y params).

    No fija el conjunto de familias: cada etapa decide cuales exige.
    """
    tabla = pd.read_csv(ruta)
    if not {"modelo", "params"}.issubset(tabla.columns):
        raise ValueError(f"{ruta} debe contener las columnas modelo y params")
    if tabla["modelo"].isna().any() or tabla["modelo"].duplicated().any():
        raise ValueError(f"{ruta} debe tener exactamente una configuracion por familia")
    configuraciones = {}
    for fila in tabla.itertuples(index=False):
        try:
            params = ast.literal_eval(fila.params)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Parametros invalidos para {fila.modelo}") from exc
        if not isinstance(params, dict):
            raise ValueError(f"Los parametros de {fila.modelo} deben ser un diccionario")
        configuraciones[fila.modelo] = params
    return configuraciones


def contar_combinaciones(mallas: dict) -> pd.DataFrame:
    filas = []
    for m, g in mallas.items():
        if isinstance(g, list):
            n = sum(int(np.prod([len(v) for v in grid.values()])) for grid in g)
        else:
            n = int(np.prod([len(v) for v in g.values()]))
        filas.append({"modelo": m, "combinaciones": n})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Puntuaciones y metricas
# ---------------------------------------------------------------------------
def puntuacion(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def predecir(modelo, X):
    """Umbral nativo: 0.5 probabilistico, 0 decision_function."""
    if hasattr(modelo, "predict_proba"):
        return (modelo.predict_proba(X)[:, 1] >= 0.5).astype(int)
    return (modelo.decision_function(X) >= 0).astype(int)


def metricas(y_true, y_pred, y_score) -> dict:
    return {
        "AP": float(average_precision_score(y_true, y_score)),
        "ROC_AUC": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def bootstrap_ic(y_true, y_pred, y_score, n_boot=2000, semilla=SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    registros = []
    for _ in range(n_boot):
        ip = rng.choice(pos, size=len(pos), replace=True)
        ineg = rng.choice(neg, size=len(neg), replace=True)
        ix = np.concatenate([ip, ineg])
        yt, yp, ys = y_true[ix], y_pred[ix], y_score[ix]
        if yt.min() == yt.max():
            continue
        registros.append(metricas(yt, yp, ys))
    b = pd.DataFrame(registros)
    return pd.DataFrame(
        {
            "metrica": b.columns,
            "media_bootstrap": b.mean().values,
            "ic_inferior": b.quantile(0.025).values,
            "ic_superior": b.quantile(0.975).values,
        }
    )


def bootstrap_diferencia(y_true, score_a, score_b, metrica, n_boot=2000, semilla=SEMILLA):
    """Diferencia puntual e IC pareado, condicionado a las puntuaciones fijas."""
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    if y_true.ndim != 1 or score_a.shape != y_true.shape or score_b.shape != y_true.shape:
        raise ValueError("Las etiquetas y puntuaciones deben ser vectores de igual longitud")
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if not len(pos) or not len(neg) or len(pos) + len(neg) != len(y_true) or n_boot < 1:
        raise ValueError("Se requieren ambas clases binarias y al menos una replica")
    rng = np.random.default_rng(semilla)
    diferencias = np.empty(n_boot)
    for i in range(n_boot):
        ix = np.concatenate((rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)))
        diferencias[i] = metrica(y_true[ix], score_a[ix]) - metrica(y_true[ix], score_b[ix])
    puntual = metrica(y_true, score_a) - metrica(y_true, score_b)
    inferior, superior = np.quantile(diferencias, [0.025, 0.975])
    return float(puntual), float(diferencias.mean()), float(inferior), float(superior)
