"""Utilidades reproducibles para la seccion 1 del Entregable II.

Prediccion de la intencion de compra (Online Shoppers Purchasing Intention,
UCI #468). Todo el preprocesamiento se ajusta dentro de los folds de
entrenamiento mediante Pipeline/ColumnTransformer.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.svm import SVC

SEMILLA = 42
RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "data" / "online_shoppers_intention.csv"

NUMERICAS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
]
LOG_VARS = NUMERICAS
NOMINALES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
BINARIAS = ["Weekend"]
ESPECIALES = ["SpecialDay"]
PREDICTORAS = NUMERICAS + NOMINALES + BINARIAS + ESPECIALES
OBJETIVO = "Revenue"

DIR_RES = RAIZ / "resultados"
DIR_FIG = DIR_RES / "figuras"


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
class TransformadorLog(BaseEstimator, TransformerMixin):
    """Aplica log1p opcionalmente; la decision se ajusta solo con train/CV."""

    def __init__(self, usar_log: bool = True):
        self.usar_log = usar_log

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.log1p(X) if self.usar_log else X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


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
                "Instale umap-learn para usar TransformadorUMAP: pip install umap-learn"
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


def _normalizar_excluir(excluir) -> tuple[str, ...]:
    """Convierte excluir en una tupla validada de predictoras."""
    if excluir is None:
        return ()
    if isinstance(excluir, str):
        nombres = (excluir,)
    else:
        try:
            nombres = tuple(excluir)
        except TypeError as exc:
            raise TypeError("excluir debe ser un iterable de nombres de columna") from exc
    desconocidas = [c for c in nombres if c not in PREDICTORAS]
    if desconocidas:
        raise ValueError(f"Columnas no predictoras en excluir: {desconocidas}")
    return nombres


def construir_preprocesador(
    usar_log: bool = True,
    incluir_pagevalues: bool = True,
    incluir_specialday: bool = True,
    codificar_specialday: str = "numerico",
    excluir=None,
):
    """Construye el ColumnTransformer; toda la codificacion se ajusta en train.

    excluir generaliza las banderas: los nombres indicados nunca llegan al
    modelo, ademas de PageValues cuando incluir_pagevalues es False y de
    SpecialDay cuando incluir_specialday es False. codificar_specialday admite
    "numerico" (columna estandarizada) y "onehot" (seis niveles dentro del
    bloque categorico).
    """
    if codificar_specialday not in {"numerico", "onehot"}:
        raise ValueError("codificar_specialday debe ser 'numerico' o 'onehot'")
    excluidas = set(_normalizar_excluir(excluir))
    if not incluir_pagevalues:
        excluidas.add("PageValues")
    if not incluir_specialday:
        excluidas.add("SpecialDay")
    numericas = [c for c in LOG_VARS if c not in excluidas]
    nominales = [c for c in NOMINALES if c not in excluidas]
    binarias = [c for c in BINARIAS if c not in excluidas]
    especiales_numericas = [c for c in ESPECIALES if c not in excluidas]
    especiales_onehot = []
    if codificar_specialday == "onehot":
        especiales_onehot = especiales_numericas
        especiales_numericas = []
    categorias = nominales + especiales_onehot
    num_pipe = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("log", TransformadorLog(usar_log=usar_log)),
            ("escalar", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformadores = []
    if numericas:
        transformadores.append(("num", num_pipe, numericas))
    if categorias:
        transformadores.append(("cat", cat_pipe, categorias))
    if binarias:
        transformadores.append(("bin", "passthrough", binarias))
    if especiales_numericas:
        transformadores.append(
            (
                "esp",
                Pipeline(
                    [
                        ("imputar", SimpleImputer(strategy="median")),
                        ("escalar", StandardScaler()),
                    ]
                ),
                especiales_numericas,
            )
        )
    if not transformadores:
        raise ValueError("El preprocesador no tiene columnas predictoras")
    return ColumnTransformer(transformadores, remainder="drop")


def construir_modelos(
    incluir_pagevalues: bool = True,
    incluir_specialday: bool = True,
    codificar_specialday: str = "numerico",
    excluir=None,
):
    """Devuelve (modelos, mallas) con transformacion log1p como hiperparametro."""
    excluir = _normalizar_excluir(excluir)

    def pre():
        return construir_preprocesador(
            incluir_pagevalues=incluir_pagevalues,
            incluir_specialday=incluir_specialday,
            codificar_specialday=codificar_specialday,
            excluir=excluir,
        )

    modelos = {}
    mallas = {}

    modelos["Regresion_logistica"] = Pipeline(
        [
            ("pre", pre()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEMILLA)),
        ]
    )
    mallas["Regresion_logistica"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": [None, "balanced"],
    }

    modelos["kNN"] = Pipeline(
        [
            ("pre", pre()),
            ("clf", KNeighborsClassifier()),
        ]
    )
    mallas["kNN"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__n_neighbors": [5, 15, 31, 45, 61, 81, 101],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }

    modelos["RandomForest"] = Pipeline(
        [
            ("pre", pre()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, random_state=SEMILLA, n_jobs=1
                ),
            ),
        ]
    )
    mallas["RandomForest"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__max_depth": [None, 12, 20, 30],
        "clf__min_samples_leaf": [1, 5],
        "clf__max_features": ["sqrt", 0.5],
        "clf__class_weight": [None, "balanced"],
    }

    modelos["MLP"] = Pipeline(
        [
            ("pre", pre()),
            (
                "clf",
                MLPClassifier(
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )
    mallas["MLP"] = {
        "pre__num__log__usar_log": [True, False],
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "clf__alpha": [0.0001, 0.01],
        "clf__learning_rate_init": [0.001, 0.01, 0.03, 0.1],
    }

    modelos["SVM"] = Pipeline(
        [
            ("pre", pre()),
            ("clf", SVC(probability=False, random_state=SEMILLA)),
        ]
    )
    mallas["SVM"] = [
        {
            "pre__num__log__usar_log": [True, False],
            "clf__kernel": ["linear"],
            "clf__C": [0.1, 1, 10, 30, 100],
            "clf__class_weight": [None, "balanced"],
        },
        {
            "pre__num__log__usar_log": [True, False],
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1, 10, 30, 100],
            "clf__gamma": ["scale", 0.01, 0.001],
            "clf__class_weight": [None, "balanced"],
        },
    ]
    return modelos, mallas


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


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
@dataclass
class ResultadoFamilia:
    nombre: str
    mejor_params: dict
    cv_ap: float
    cv_std: float
    modelo: object
    tiempo: float


def entrenar_familia(nombre, modelo, malla, X_tr, y_tr, cv, n_jobs=-1):
    t0 = time.time()
    gs = GridSearchCV(
        modelo,
        malla,
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        refit=True,
        verbose=0,
    )
    gs.fit(X_tr, y_tr)
    dt = time.time() - t0
    best_i = gs.best_index_
    res = ResultadoFamilia(
        nombre=nombre,
        mejor_params=gs.best_params_,
        cv_ap=float(gs.best_score_),
        cv_std=float(gs.cv_results_["std_test_score"][best_i]),
        modelo=gs.best_estimator_,
        tiempo=dt,
    )
    cv_df = pd.DataFrame(gs.cv_results_).copy()
    cv_df["modelo"] = nombre
    cv_df["mejor"] = cv_df.index == best_i
    return res, cv_df


def evaluar(res: ResultadoFamilia, X, y, particion: str) -> dict:
    y_pred = predecir(res.modelo, X)
    y_score = puntuacion(res.modelo, X)
    d = metricas(y, y_pred, y_score)
    d.update({"modelo": res.nombre, "particion": particion})
    return d


def matriz_confusion(res: ResultadoFamilia, X, y):
    return confusion_matrix(y, predecir(res.modelo, X), labels=[0, 1])
