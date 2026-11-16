"""Ablacion y codificacion de SpecialDay con hiperparametros fijos.

Uso: python src/specialday.py --salida resultados [--rapido]

La ablacion compara, para las cinco familias, el AP de validacion con y sin
SpecialDay; la convencion es delta_AP_con_menos_sin = con - sin. La
codificacion compara la columna numerica actual con one-hot de seis niveles
mediante AP de CV de 5 folds en entrenamiento y AP de validacion. El modo
rapido reduce a 2 folds y hasta 600 filas de entrenamiento: es solo una
prueba de humo y sus cifras no son comparables con la ejecucion normal.
La prueba no se consulta en ningun momento.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from proyecto_ml import (
    leer_configuraciones,
    CSV,
    RAIZ,
    SEMILLA,
    _submuestra_rapida,
    cargar_datos,
    construir_modelos,
    particionar,
    puntuacion,
)

SALIDA = RAIZ / "resultados"
FAMILIAS = (
    "Regresion_logistica",
    "kNN",
    "RandomForest",
    "MLP",
    "SVM",
)
CODIFICACIONES = ("numerico", "onehot")
N_FOLDS = 5
ORDEN_MESES = (
    "Jan", "Feb", "Mar", "Apr", "May", "June",
    "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def cargar_configuraciones(ruta) -> dict[str, dict]:
    """Carga exactamente una configuracion fija por cada una de las familias."""
    configuraciones = leer_configuraciones(ruta)
    if set(configuraciones) != set(FAMILIAS):
        raise ValueError("Se requiere exactamente una configuracion por familia")
    return {nombre: configuraciones[nombre] for nombre in FAMILIAS}


def _validar_configuraciones(configuraciones) -> dict[str, dict]:
    if isinstance(configuraciones, (str, Path)):
        return cargar_configuraciones(configuraciones)
    if not isinstance(configuraciones, dict):
        raise TypeError("configuraciones debe ser un CSV o un diccionario por familia")
    if set(configuraciones) != set(FAMILIAS):
        raise ValueError("Se requiere exactamente una configuracion por familia")
    for nombre, params in configuraciones.items():
        if not isinstance(params, dict):
            raise ValueError(f"Los parametros de {nombre} deben ser un diccionario")
    return {nombre: dict(configuraciones[nombre]) for nombre in FAMILIAS}


def _construir_modelo(nombre, params, codificar_specialday="numerico", incluir_specialday=True):
    """Pipeline con parametros fijos; el preprocesador se ajusta dentro de el."""
    modelos, _ = construir_modelos(
        incluir_specialday=incluir_specialday,
        codificar_specialday=codificar_specialday,
    )
    return clone(modelos[nombre]).set_params(**params)


def _ap_cv(modelo, X, y, n_folds, n_jobs, semilla):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    valores = cross_val_score(
        modelo,
        X,
        y,
        scoring="average_precision",
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    return float(np.mean(valores)), float(np.std(valores)), valores.tolist()


def _ap_validacion(modelo, X_tr, y_tr, X_va, y_va) -> float:
    modelo.fit(X_tr, y_tr)
    return float(average_precision_score(y_va, puntuacion(modelo, X_va)))


def calcular_ablacion(configuraciones, X_tr, y_tr, X_va, y_va) -> pd.DataFrame:
    """Ablacion con las cinco familias y parametros fijos del CSV."""
    filas = []
    for nombre in FAMILIAS:
        params = configuraciones[nombre]
        ap_con = _ap_validacion(
            _construir_modelo(nombre, params, incluir_specialday=True),
            X_tr, y_tr, X_va, y_va,
        )
        ap_sin = _ap_validacion(
            _construir_modelo(nombre, params, incluir_specialday=False),
            X_tr, y_tr, X_va, y_va,
        )
        filas.append(
            {
                "modelo": nombre,
                "AP_con_SpecialDay": ap_con,
                "AP_sin_SpecialDay": ap_sin,
                "delta_AP_con_menos_sin": ap_con - ap_sin,
                "params": repr(params),
            }
        )
    return pd.DataFrame(filas)


def calcular_codificacion(
    configuraciones, X_tr, y_tr, X_va, y_va, n_folds, n_jobs=1, semilla=SEMILLA
) -> pd.DataFrame:
    """AP de CV y validacion para SpecialDay numerico y one-hot."""
    filas = []
    for nombre in FAMILIAS:
        params = configuraciones[nombre]
        medidas = {}
        for codificacion in CODIFICACIONES:
            modelo = _construir_modelo(nombre, params, codificar_specialday=codificacion)
            ap_cv, std_cv, folds = _ap_cv(modelo, X_tr, y_tr, n_folds, n_jobs, semilla)
            ap_va = _ap_validacion(modelo, X_tr, y_tr, X_va, y_va)
            n_columnas = len(modelo.named_steps["pre"].get_feature_names_out())
            medidas[codificacion] = {
                "AP_CV": ap_cv,
                "std_CV": std_cv,
                "folds": json.dumps(folds),
                "AP_validacion": ap_va,
                "n_columnas": n_columnas,
            }
        filas.append(
            {
                "modelo": nombre,
                "AP_CV_numerico": medidas["numerico"]["AP_CV"],
                "std_CV_numerico": medidas["numerico"]["std_CV"],
                "AP_CV_onehot": medidas["onehot"]["AP_CV"],
                "std_CV_onehot": medidas["onehot"]["std_CV"],
                "delta_AP_CV_onehot_menos_numerico": (
                    medidas["onehot"]["AP_CV"] - medidas["numerico"]["AP_CV"]
                ),
                "AP_validacion_numerico": medidas["numerico"]["AP_validacion"],
                "AP_validacion_onehot": medidas["onehot"]["AP_validacion"],
                "delta_AP_validacion_onehot_menos_numerico": (
                    medidas["onehot"]["AP_validacion"] - medidas["numerico"]["AP_validacion"]
                ),
                "n_columnas_numerico": medidas["numerico"]["n_columnas"],
                "n_columnas_onehot": medidas["onehot"]["n_columnas"],
                "folds_numerico": medidas["numerico"]["folds"],
                "folds_onehot": medidas["onehot"]["folds"],
            }
        )
    return pd.DataFrame(filas)


def _orden_mes(mes):
    try:
        return ORDEN_MESES.index(mes), mes
    except ValueError:
        return len(ORDEN_MESES), mes


def calcular_por_mes(nombre, params, X_tr, y_tr, X_va, y_va) -> pd.DataFrame:
    """AP del ganador en cada mes de validacion, sin usar la prueba."""
    modelo = _construir_modelo(nombre, params, incluir_specialday=True)
    modelo.fit(X_tr, y_tr)
    scores = puntuacion(modelo, X_va)
    meses = X_va["Month"].astype(str)
    filas = []
    for mes in sorted(meses.unique(), key=_orden_mes):
        mascara = (meses == mes).to_numpy()
        yt = np.asarray(y_va)[mascara]
        ys = np.asarray(scores)[mascara]
        ap = (
            float(average_precision_score(yt, ys))
            if len(yt) and len(np.unique(yt)) == 2
            else np.nan
        )
        filas.append(
            {
                "modelo": nombre,
                "mes": mes,
                "n": int(mascara.sum()),
                "positivos": int(yt.sum()),
                "AP_validacion": ap,
            }
        )
    return pd.DataFrame(filas)


def ejecutar_specialday(
    X_tr,
    y_tr,
    X_va,
    y_va,
    configuraciones,
    salida=SALIDA,
    rapido=False,
    por_mes=True,
    n_jobs=1,
    semilla=SEMILLA,
):
    """Ejecuta ablacion, comparacion de codificacion y AP por mes.

    Devuelve las tablas y un resumen con el protocolo aplicado. Todas las
    decisiones usan solo entrenamiento y validacion; la prueba no interviene.
    """
    configuraciones = _validar_configuraciones(configuraciones)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    n_folds = 2 if rapido else N_FOLDS
    X_cv, y_cv = _submuestra_rapida(X_tr, y_tr, semilla) if rapido else (X_tr, y_tr)
    y_cv = pd.Series(np.asarray(y_cv), index=X_cv.index)
    y_va = pd.Series(np.asarray(y_va), index=X_va.index)

    ablacion = calcular_ablacion(configuraciones, X_cv, y_cv, X_va, y_va)
    codificacion = calcular_codificacion(
        configuraciones, X_cv, y_cv, X_va, y_va, n_folds, n_jobs, semilla
    )
    tabla_mes = None
    ganador = str(ablacion.sort_values("AP_con_SpecialDay", ascending=False).iloc[0]["modelo"])
    if por_mes:
        tabla_mes = calcular_por_mes(
            ganador, configuraciones[ganador], X_cv, y_cv, X_va, y_va
        )

    ablacion.to_csv(salida / "specialday_ablacion.csv", index=False)
    codificacion.to_csv(salida / "specialday_codificacion.csv", index=False)
    if tabla_mes is not None:
        tabla_mes.to_csv(salida / "specialday_por_mes.csv", index=False)

    resumen = {
        "rapido": rapido,
        "n_folds": n_folds,
        "n_entrenamiento": int(len(X_cv)),
        "n_validacion": int(len(X_va)),
        "ganador_validacion": ganador,
        "AP_validacion_ganador": float(
            ablacion.loc[ablacion["modelo"] == ganador, "AP_con_SpecialDay"].iloc[0]
        ),
        "delta_ablacion": "con - sin",
        "delta_codificacion": "onehot - numerico",
        "decision": "solo entrenamiento y validacion; prueba no consultada",
    }
    return {
        "ablacion": ablacion,
        "codificacion": codificacion,
        "por_mes": tabla_mes,
        "resumen": resumen,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=CSV)
    parser.add_argument(
        "--configuraciones",
        type=Path,
        default=RAIZ / "resultados" / "mejores_configuraciones.csv",
    )
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--rapido", action="store_true")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--por-mes", dest="por_mes", action="store_true")
    grupo.add_argument("--no-por-mes", dest="por_mes", action="store_false")
    parser.set_defaults(por_mes=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    configuraciones = cargar_configuraciones(args.configuraciones)
    _, X, y, idx_tr, idx_va, _ = particionar(cargar_datos(args.datos))
    resultado = ejecutar_specialday(
        X.iloc[idx_tr],
        y.iloc[idx_tr],
        X.iloc[idx_va],
        y.iloc[idx_va],
        configuraciones,
        args.salida,
        rapido=args.rapido,
        por_mes=args.por_mes,
        n_jobs=args.n_jobs,
    )
    print(resultado["ablacion"].to_string(index=False))
    print(resultado["codificacion"].to_string(index=False))
    if resultado["por_mes"] is not None:
        print(resultado["por_mes"].to_string(index=False))
    print(f"Resultados en {args.salida}")


if __name__ == "__main__":
    main()
