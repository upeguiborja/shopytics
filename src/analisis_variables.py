"""Analisis individual de las 17 variables predictoras (RD-01).

Calcula, solo con el conjunto de entrenamiento:
  - correlacion punto-biserial y AUC univariado para las numericas;
  - V de Cramer y AUC univariado con one-hot para las categoricas;
  - informacion mutua con la etiqueta;
  - redundancia entre numericas (|r| > 0.8).

Guarda resultados_corregidos/analisis_variables.csv y
resultados_corregidos/redundancia_numericas.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proyecto_ml import (  # noqa: E402
    BINARIAS,
    CSV,
    ESPECIALES,
    NOMINALES,
    NUMERICAS,
    cargar_datos,
    particionar,
)
from scipy.stats import chi2_contingency, pointbiserialr  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
OUT = RAIZ / "resultados_corregidos"
OUT.mkdir(parents=True, exist_ok=True)


def auc_univariado_numerica(x, y):
    x = np.asarray(x, dtype=float)
    auc = roc_auc_score(y, x)
    if auc < 0.5:
        return 1 - auc, "negativa"
    return auc, "positiva"


def auc_univariado_categorica(x, y):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X = enc.fit_transform(np.asarray(x).reshape(-1, 1))
    modelo = LogisticRegression(max_iter=2000, solver="lbfgs")
    modelo.fit(X, y)
    proba = modelo.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    if auc < 0.5:
        return 1 - auc, "negativa"
    return auc, "positiva"


def cramers_v(x, y):
    tabla = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(tabla)
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    if n == 0 or min(r - 1, k - 1) == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * (min(r - 1, k - 1)))))


def main():
    df = cargar_datos(CSV)
    df_u, X, y, itr, iva, ite = particionar(df)
    Xtr = X.iloc[itr].reset_index(drop=True)
    ytr = y.iloc[itr].reset_index(drop=True)

    filas = []

    # Numericas y SpecialDay.
    for var in NUMERICAS + ESPECIALES:
        x = Xtr[var].to_numpy()
        corr, p = pointbiserialr(ytr, x)
        auc, direccion = auc_univariado_numerica(x, ytr)
        mi = mutual_info_classif(
            x.reshape(-1, 1), ytr, discrete_features=False, random_state=42
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
            }
        )

    # Nominales y binarias.
    for var in NOMINALES + BINARIAS:
        x = Xtr[var]
        v = cramers_v(x, ytr)
        auc, direccion = auc_univariado_categorica(x, ytr)
        codigos, _ = pd.factorize(x, sort=True)
        mi = mutual_info_classif(
            codigos.reshape(-1, 1), ytr, discrete_features=True, random_state=42
        )[0]
        filas.append(
            {
                "variable": var,
                "tipo": "categórica",
                "metrica": "V de Cramér",
                "valor": float(v),
                "p_valor": np.nan,
                "auc_univariado": float(auc),
                "direccion": direccion,
                "informacion_mutua": float(mi),
            }
        )

    tabla = pd.DataFrame(filas)
    tabla = tabla.sort_values("auc_univariado", ascending=False).reset_index(drop=True)
    tabla["ranking"] = np.arange(1, len(tabla) + 1)
    tabla["recomendacion"] = np.where(
        tabla["auc_univariado"] >= 0.6,
        "mantener",
        np.where(tabla["auc_univariado"] >= 0.55, "revisar", "eliminar o combinar"),
    )
    tabla.to_csv(OUT / "analisis_variables.csv", index=False)

    # Redundancia entre numericas.
    corr = Xtr[NUMERICAS].corr()
    pares = []
    for i, a in enumerate(NUMERICAS):
        for b in NUMERICAS[i + 1 :]:
            r = corr.loc[a, b]
            if abs(r) >= 0.8:
                pares.append({"variable_1": a, "variable_2": b, "correlacion": float(r)})
    pd.DataFrame(pares).to_csv(OUT / "redundancia_numericas.csv", index=False)

    print(tabla.to_string(index=False))
    print("\nPares con |r| >= 0.8:")
    print(pd.DataFrame(pares).to_string(index=False))
    print("\nResultados en", OUT)


if __name__ == "__main__":
    main()
