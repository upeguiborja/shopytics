"""Analisis individual de las 17 variables predictoras y eliminacion.

Todo el analisis se calcula con el conjunto de entrenamiento:
  - correlacion punto-biserial, indice de Fisher y AUC univariado para las
    numericas;
  - V de Cramer y AUC univariado fuera de muestra con one-hot + regresion
    logistica y cross_val_predict estratificado de 5 folds para las categoricas;
  - informacion mutua con la etiqueta;
  - redundancia entre numericas (|r| >= 0.8).

El experimento de eliminacion toma RandomForest y MLP de las configuraciones
seleccionadas y compara tres variantes: todas las variables, sin las candidatas
que el propio analisis marca con AUC univariado menor que 0.55 y sin
ProductRelated_Duration ni BounceRates. Cada variante se evalua con validacion
cruzada de 5 folds en entrenamiento y con AP en validacion. La particion de
prueba no participa.

Salida por defecto: resultados/variables (configurable con --salida). En modo
rapido la eliminacion usa 2 folds y hasta 600 filas estratificadas de
entrenamiento; es una prueba de humo y sus cifras no son comparables. El
analisis descriptivo mantiene los 5 folds sobre todo el entrenamiento.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopytics.models import construir_modelos
from proyecto_ml import (  # noqa: E402
    BINARIAS,
    CSV,
    ESPECIALES,
    NOMINALES,
    NUMERICAS,
    SEMILLA,
    _submuestra_rapida,
    cargar_datos,
    metricas,
    particionar,
    predecir,
    puntuacion,
)

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "resultados" / "variables"
CONFIGURACIONES = RAIZ / "resultados" / "mejores_configuraciones.csv"
FAMILIAS_ELIMINACION = ("RandomForest", "MLP")
REDUNDANTES = ("ProductRelated_Duration", "BounceRates")
UMBRAL_CANDIDATA = 0.55
N_FOLDS = 5


def auc_univariado_numerica(x, y):
    """AUC de una numerica y direccion; se invierte si queda por debajo de 0.5."""
    x = np.asarray(x, dtype=float)
    auc = float(roc_auc_score(y, x))
    if auc < 0.5:
        return 1 - auc, "negativa"
    return auc, "positiva"


def auc_univariado_categorica(x, y, n_splits=N_FOLDS, semilla=SEMILLA):
    """AUC fuera de muestra de una categorica dentro del entrenamiento."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    modelo = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    proba = cross_val_predict(modelo, x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    auc = float(roc_auc_score(y, proba))
    if auc < 0.5:
        return 1 - auc, "negativa"
    return auc, "positiva"


def indice_fisher(x, y):
    """Indice de discriminacion de Fisher: (mu1 - mu0)^2 / (var1 + var0)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    positivos, negativos = x[y == 1], x[y == 0]
    if len(positivos) < 2 or len(negativos) < 2:
        return np.nan
    varianza = positivos.var(ddof=1) + negativos.var(ddof=1)
    if not np.isfinite(varianza) or varianza <= 0:
        return np.nan
    return float((positivos.mean() - negativos.mean()) ** 2 / varianza)


def cramers_v(x, y):
    tabla = pd.crosstab(x, y)
    if tabla.empty:
        return 0.0
    chi2, _, _, _ = chi2_contingency(tabla)
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    if n == 0 or min(r - 1, k - 1) == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * (min(r - 1, k - 1)))))


def analizar_variables(X_tr, y_tr, n_splits=N_FOLDS, semilla=SEMILLA):
    """Tabla descriptiva por variable; solo usa el entrenamiento recibido."""
    filas = []

    # Numericas y SpecialDay.
    for var in NUMERICAS + ESPECIALES:
        x = X_tr[var].to_numpy()
        corr, p = pointbiserialr(y_tr, x)
        auc, direccion = auc_univariado_numerica(x, y_tr)
        mi = mutual_info_classif(
            x.reshape(-1, 1), y_tr, discrete_features=False, random_state=semilla
        )[0]
        filas.append(
            {
                "variable": var,
                "tipo": "numérica",
                "metrica": "correlación punto-biserial",
                "valor": float(corr),
                "p_valor": float(p),
                "auc_univariado": float(auc),
                "direccion": direccion,
                "informacion_mutua": float(mi),
                "indice_fisher": indice_fisher(x, y_tr),
            }
        )

    # Nominales y binarias.
    for var in NOMINALES + BINARIAS:
        x = X_tr[var]
        v = cramers_v(x, y_tr)
        auc, direccion = auc_univariado_categorica(
            x, y_tr, n_splits=n_splits, semilla=semilla
        )
        codigos, _ = pd.factorize(x, sort=True)
        mi = mutual_info_classif(
            codigos.reshape(-1, 1), y_tr, discrete_features=True, random_state=semilla
        )[0]
        filas.append(
            {
                "variable": var,
                "tipo": "binaria" if var in BINARIAS else "categórica",
                "metrica": "V de Cramér",
                "valor": float(v),
                "p_valor": np.nan,
                "auc_univariado": float(auc),
                "direccion": direccion,
                "informacion_mutua": float(mi),
                "indice_fisher": np.nan,
            }
        )

    tabla = pd.DataFrame(filas)
    tabla = tabla.sort_values(
        ["auc_univariado", "variable"], ascending=[False, True]
    ).reset_index(drop=True)
    tabla["ranking"] = np.arange(1, len(tabla) + 1)
    tabla["recomendacion"] = np.where(
        tabla["auc_univariado"] >= 0.6,
        "mantener",
        np.where(tabla["auc_univariado"] >= 0.55, "revisar", "eliminar o combinar"),
    )
    return tabla


def redundancia_numericas(X_tr):
    """Pares de numericas con |r| >= 0.8, calculado solo en entrenamiento."""
    corr = X_tr[NUMERICAS].corr()
    pares = []
    for i, a in enumerate(NUMERICAS):
        for b in NUMERICAS[i + 1 :]:
            r = corr.loc[a, b]
            if abs(r) >= 0.8:
                pares.append({"variable_1": a, "variable_2": b, "correlacion": float(r)})
    return pd.DataFrame(pares, columns=["variable_1", "variable_2", "correlacion"])


def candidatas_eliminacion(tabla_analisis):
    """Variables con AUC univariado por debajo del umbral de descarte.

    El umbral es 0.55, el mismo que usa la recomendacion de la tabla: por
    debajo de ese valor la variable se marca como "eliminar o combinar".
    """
    if "auc_univariado" not in tabla_analisis.columns:
        raise ValueError("La tabla de analisis no contiene auc_univariado")
    orden = tabla_analisis.sort_values(
        ["auc_univariado", "variable"], ascending=[True, True]
    )
    return orden.loc[orden["auc_univariado"] < UMBRAL_CANDIDATA, "variable"].tolist()


def cargar_configuraciones(ruta, familias=FAMILIAS_ELIMINACION):
    """Una configuracion por familia desde el CSV; si hay duplicados, mayor AP."""
    tabla = pd.read_csv(ruta)
    requeridas = {"modelo", "params"}
    if not requeridas.issubset(tabla.columns):
        faltan = sorted(requeridas - set(tabla.columns))
        raise ValueError(f"Faltan columnas en {ruta}: {faltan}")
    disponibles = set(tabla["modelo"])
    ausentes = [familia for familia in familias if familia not in disponibles]
    if ausentes:
        raise ValueError(f"Familias ausentes en el CSV: {ausentes}")

    configuraciones = []
    for familia in familias:
        filas = tabla[tabla["modelo"] == familia].copy()
        if "AP" in filas.columns:
            filas["AP"] = pd.to_numeric(filas["AP"], errors="coerce")
            filas = filas.sort_values("AP", ascending=False, na_position="last")
        fila = filas.iloc[0]
        try:
            params = ast.literal_eval(fila["params"])
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Parametros invalidos para {familia}") from exc
        if not isinstance(params, dict):
            raise ValueError(f"Los parametros de {familia} deben ser un diccionario")
        configuraciones.append((familia, params))
    return configuraciones


def _validar_configuraciones(configuraciones):
    """Acepta una ruta CSV o un diccionario con RandomForest y MLP."""
    if isinstance(configuraciones, (str, Path)):
        return cargar_configuraciones(configuraciones)
    if not isinstance(configuraciones, dict):
        raise TypeError("configuraciones debe ser un CSV o un diccionario por familia")
    ausentes = [
        familia for familia in FAMILIAS_ELIMINACION if familia not in configuraciones
    ]
    if ausentes:
        raise ValueError(f"Faltan familias en configuraciones: {ausentes}")
    resultado = []
    for familia in FAMILIAS_ELIMINACION:
        params = configuraciones[familia]
        if not isinstance(params, dict):
            raise ValueError(f"Los parametros de {familia} deben ser un diccionario")
        resultado.append((familia, dict(params)))
    return resultado


def ejecutar_eliminacion(
    X_tr,
    y_tr,
    X_va,
    y_va,
    tabla_analisis,
    configuraciones=CONFIGURACIONES,
    salida=SALIDA,
    n_splits=N_FOLDS,
    semilla=SEMILLA,
):
    """Compara RF y MLP del CSV con tres variantes de variables.

    La validacion cruzada se hace solo en entrenamiento y el AP final se
    reporta en validacion; la prueba no se recibe ni se consulta. Las
    candidatas salen de la tabla de analisis, no de una lista fija.
    """
    candidatas = candidatas_eliminacion(tabla_analisis)
    variantes = [
        ("completo", []),
        ("sin_candidatas", candidatas),
        ("sin_redundantes", list(REDUNDANTES)),
    ]
    for _, excluir in variantes:
        desconocidas = sorted(set(excluir) - set(X_tr.columns))
        if desconocidas:
            raise ValueError(f"Variables ausentes en X_tr: {desconocidas}")

    configuraciones = _validar_configuraciones(configuraciones)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    filas_cv, filas_val = [], []
    for nombre, params in configuraciones:
        for variante, excluir in variantes:
            modelos = construir_modelos(excluir=excluir)
            modelo = modelos[nombre].set_params(**params)
            valores = cross_val_score(
                modelo,
                X_tr,
                y_tr,
                cv=cv,
                scoring="average_precision",
                n_jobs=1,
                error_score="raise",
            )
            fila_cv = {
                "modelo": nombre,
                "variante": variante,
                "excluidas": ";".join(excluir),
                "AP_CV": float(np.mean(valores)),
                "std_CV": float(np.std(valores)),
                "AP_folds": json.dumps(valores.tolist()),
            }
            filas_cv.append(fila_cv)

            modelo.fit(X_tr, y_tr)
            met = metricas(y_va, predecir(modelo, X_va), puntuacion(modelo, X_va))
            filas_val.append(
                {
                    "modelo": nombre,
                    "variante": variante,
                    "excluidas": ";".join(excluir),
                    "AP_CV": fila_cv["AP_CV"],
                    "std_CV": fila_cv["std_CV"],
                    **met,
                }
            )

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    cv_df = pd.DataFrame(filas_cv)
    val_df = pd.DataFrame(filas_val)
    cv_df.to_csv(salida / "eliminacion_cv.csv", index=False)
    val_df.to_csv(salida / "eliminacion_validacion.csv", index=False)

    delta_validacion, delta_cv = {}, {}
    for nombre in val_df["modelo"].unique():
        base_val = val_df[
            (val_df["modelo"] == nombre) & (val_df["variante"] == "completo")
        ].iloc[0]
        base_cv = cv_df[
            (cv_df["modelo"] == nombre) & (cv_df["variante"] == "completo")
        ].iloc[0]
        delta_validacion[nombre] = {
            fila.variante: float(fila.AP - base_val["AP"])
            for fila in val_df[val_df["modelo"] == nombre].itertuples()
        }
        delta_cv[nombre] = {
            fila.variante: float(fila.AP_CV - base_cv["AP_CV"])
            for fila in cv_df[cv_df["modelo"] == nombre].itertuples()
        }

    resumen = {
        "variantes": [
            {"nombre": nombre, "excluidas": list(excluir)} for nombre, excluir in variantes
        ],
        "candidatas": candidatas,
        "criterio_candidatas": (
            "variables con AUC univariado menor que 0.55, derivadas del propio "
            "analisis (recomendacion 'eliminar o combinar')"
        ),
        "n_folds": n_splits,
        "n_entrenamiento": int(len(X_tr)),
        "n_validacion": int(len(y_va)),
        "semilla": semilla,
        "delta_AP_validacion_respecto_completo": delta_validacion,
        "delta_AP_CV_respecto_completo": delta_cv,
        "prueba": "no participa en el analisis ni en la eliminacion",
    }
    (salida / "eliminacion_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"cv": cv_df, "validacion": val_df, "resumen": resumen}


def ejecutar_analisis(
    X_tr,
    y_tr,
    X_va,
    y_va,
    configuraciones=CONFIGURACIONES,
    salida=SALIDA,
    rapido=False,
    n_splits=N_FOLDS,
    semilla=SEMILLA,
    ejecutar_eliminacion_experimento=True,
):
    """Analiza las variables y, si se pide, compara la eliminacion.

    Las particiones llegan ya separadas; el analisis descriptivo usa todo el
    entrenamiento y la eliminacion usa validacion cruzada en entrenamiento y AP
    en validacion. Devuelve las tablas y el resumen del protocolo.
    """
    tabla = analizar_variables(X_tr, y_tr, n_splits=n_splits, semilla=semilla)
    redundancia = redundancia_numericas(X_tr)

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "analisis_variables.csv", index=False)
    redundancia.to_csv(salida / "redundancia_numericas.csv", index=False)

    eliminacion = None
    if ejecutar_eliminacion_experimento:
        n_folds_eliminacion = 2 if rapido else n_splits
        if rapido:
            X_cv, y_cv = _submuestra_rapida(X_tr, y_tr, semilla)
        else:
            X_cv, y_cv = X_tr, y_tr
        eliminacion = ejecutar_eliminacion(
            X_cv,
            y_cv,
            X_va,
            y_va,
            tabla,
            configuraciones,
            salida,
            n_splits=n_folds_eliminacion,
            semilla=semilla,
        )
    resumen = {
        "rapido": rapido,
        "analisis": {
            "n_entrenamiento": int(len(X_tr)),
            "n_validacion": int(len(X_va)),
            "n_folds": int(n_splits),
            "auc_categorica": "cross_val_predict estratificado en entrenamiento",
        },
        "eliminacion": eliminacion["resumen"] if eliminacion is not None else None,
    }
    return {
        "analisis": tabla,
        "redundancia": redundancia,
        "eliminacion": eliminacion,
        "resumen": resumen,
        "salida": salida,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=CSV)
    parser.add_argument("--configuraciones", type=Path, default=CONFIGURACIONES)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--sin-eliminacion", action="store_true")
    args = parser.parse_args()

    _, X, y, idx_tr, idx_va, _ = particionar(cargar_datos(args.datos))
    resultado = ejecutar_analisis(
        X.iloc[idx_tr],
        y.iloc[idx_tr].astype(int),
        X.iloc[idx_va],
        y.iloc[idx_va].astype(int),
        args.configuraciones,
        args.salida,
        rapido=args.rapido,
        ejecutar_eliminacion_experimento=not args.sin_eliminacion,
    )
    print(resultado["analisis"].to_string(index=False))
    print("\nPares con |r| >= 0.8:")
    print(resultado["redundancia"].to_string(index=False))
    if resultado["eliminacion"] is not None:
        print("\nEliminacion (validacion):")
        columnas = ["modelo", "variante", "excluidas", "AP_CV", "AP", "ROC_AUC"]
        print(resultado["eliminacion"]["validacion"][columnas].to_string(index=False))
    print("\nResultados en", resultado["salida"])


if __name__ == "__main__":
    main()
