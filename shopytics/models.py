"""Pipelines de clasificación para intención de compra."""

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
    """Construye un pipeline independiente por familia de clasificación."""
    excluir = _normalizar_excluir(excluir)

    def pre():
        return construir_preprocesador(
            incluir_pagevalues=incluir_pagevalues,
            incluir_specialday=incluir_specialday,
            codificar_specialday=codificar_specialday,
            excluir=excluir,
        )

    modelos = {}

    modelos["Regresion_logistica"] = Pipeline(
        [
            ("pre", pre()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=3000, random_state=SEMILLA)),
        ]
    )

    modelos["kNN"] = Pipeline(
        [
            ("pre", pre()),
            ("clf", KNeighborsClassifier()),
        ]
    )

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

    modelos["SVM"] = Pipeline(
        [
            ("pre", pre()),
            ("clf", SVC(probability=False, random_state=SEMILLA)),
        ]
    )
    return modelos
