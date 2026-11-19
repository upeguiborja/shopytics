"""Pipelines de clasificación y mallas de hiperparámetros para intención de compra."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .utils import BINARIAS, ESPECIALES, LOG_VARS, NOMINALES, PREDICTORAS, SEMILLA


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


def orden_familias() -> tuple[str, ...]:
    """Orden estable de las cinco familias, de la basica a la flexible."""
    return ("Regresion_logistica", "kNN", "RandomForest", "MLP", "SVM")


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


def mallas_ampliadas(construir_modelos_fn=construir_modelos, rapido: bool = False):
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
