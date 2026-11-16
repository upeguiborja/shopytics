"""Evaluacion reproducible de configuraciones seleccionadas, sin busqueda.

Incluye validacion cruzada 5-fold sobre entrenamiento con las mismas
particiones de la busqueda, intervalos t al 95 % con desviacion muestral y
comparaciones pareadas de AP y ROC-AUC en prueba. Los intervalos bootstrap son
condicionales a modelos fijos y la particion de prueba ya se consulto
historicamente, por lo que no constituye una prueba independiente.
"""

from __future__ import annotations

import ast
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proyecto_ml import (  # noqa: E402
    CSV,
    RAIZ,
    SEMILLA,
    bootstrap_diferencia,
    bootstrap_ic,
    cargar_datos,
    construir_modelos,
    metricas,
    particionar,
    predecir,
    puntuacion,
)

SALIDA = RAIZ / "resultados"
SALIDA_FAMILIAS = SALIDA / "evaluacion_familias"
N_FOLDS_CV = 5
METRICAS_CV = {"AP": "average_precision", "ROC_AUC": "roc_auc", "F1": "f1"}


def ajustar(nombre, params, X_tr, y_tr, incluir_pagevalues=True, incluir_specialday=True):
    modelos, _ = construir_modelos(
        incluir_pagevalues=incluir_pagevalues, incluir_specialday=incluir_specialday
    )
    m = clone(modelos[nombre]).set_params(**params)
    m.fit(X_tr, y_tr)
    return m


def metricas_particion(modelo, X, y, particion, nombre):
    d = metricas(y, predecir(modelo, X), puntuacion(modelo, X))
    d.update({"modelo": nombre, "particion": particion})
    return d


def cargar_configuraciones(ruta):
    """Carga exactamente una configuracion seleccionada para cada familia."""
    tabla = pd.read_csv(ruta)
    modelos, _ = construir_modelos()
    if not {"modelo", "params"}.issubset(tabla.columns):
        raise ValueError("El CSV debe contener las columnas modelo y params")
    if tabla["modelo"].duplicated().any() or set(tabla["modelo"]) != set(modelos):
        raise ValueError("Se requiere exactamente una configuracion por cada una de las cinco familias")
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


def umbral_f1_validacion(y, scores):
    """Maximiza F1 en validacion; en empates elige el umbral mas bajo."""
    precision, recall, umbrales = precision_recall_curve(y, scores)
    if not len(umbrales):
        raise ValueError("No hay puntuaciones para seleccionar el umbral")
    suma = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], suma,
                   out=np.zeros_like(suma), where=suma != 0)
    i = int(np.argmax(f1))
    return float(umbrales[i]), float(f1[i])


def intervalo_t(valores, confianza=0.95):
    """Media, desviacion muestral e IC t de las puntuaciones de los folds."""
    valores = np.asarray(valores, dtype=float)
    if valores.ndim != 1 or valores.size < 2:
        raise ValueError("Se requieren al menos dos folds para el intervalo t")
    n = valores.size
    media = float(valores.mean())
    desviacion = float(valores.std(ddof=1))
    error_estandar = desviacion / np.sqrt(n)
    t_critico = float(t_dist.ppf(1 - (1 - confianza) / 2, df=n - 1))
    return {
        "n_folds": int(n),
        "media": media,
        "desviacion": desviacion,
        "error_estandar": float(error_estandar),
        "ic_inferior": float(media - t_critico * error_estandar),
        "ic_superior": float(media + t_critico * error_estandar),
    }


def validacion_cruzada(configuraciones, X, y, n_splits=N_FOLDS_CV, semilla=SEMILLA):
    """CV estratificada de las configuraciones finales sobre entrenamiento.

    Usa el mismo esquema que la busqueda: StratifiedKFold con mezcla y semilla.
    """
    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    modelos, _ = construir_modelos()
    filas = []
    for nombre, params in configuraciones.items():
        modelo = clone(modelos[nombre]).set_params(**params)
        resultados = cross_validate(
            modelo, X, y, cv=cv, scoring=METRICAS_CV, n_jobs=1, error_score="raise"
        )
        for metrica in METRICAS_CV:
            filas.append({"modelo": nombre, "metrica": metrica,
                          **intervalo_t(resultados[f"test_{metrica}"])})
    columnas = ["modelo", "metrica", "n_folds", "media", "desviacion",
                "error_estandar", "ic_inferior", "ic_superior"]
    return pd.DataFrame(filas, columns=columnas)


def evaluar_familias(
    ruta_datos=CSV,
    ruta_configuraciones=SALIDA / "mejores_configuraciones.csv",
    salida=SALIDA_FAMILIAS,
    n_boot=2000,
    semilla=SEMILLA,
    rapido=False,
    ejecutar_ablaciones=True,
):
    """Evalua configuraciones fijas; rapido limita bootstrap sin alterar modelos.

    Devuelve tablas y estructuras tambien guardadas en el directorio de salida.
    Los IC son bootstrap estratificado condicionado a los modelos ajustados.
    """
    if n_boot < 1:
        raise ValueError("n_boot debe ser positivo")
    if rapido:
        n_boot = min(n_boot, 20)
    configuraciones = cargar_configuraciones(ruta_configuraciones)
    _, X, y, idx_tr, idx_va, idx_te = particionar(cargar_datos(ruta_datos))
    particiones = {"entrenamiento": idx_tr, "validacion": idx_va, "test": idx_te}
    X_tr, y_tr = X.iloc[idx_tr], y.iloc[idx_tr].astype(int)
    # La CV reproduce los folds de la busqueda y no consulta validacion ni prueba.
    cv_intervalos = validacion_cruzada(configuraciones, X_tr, y_tr, semilla=semilla)
    modelos = {nombre: ajustar(nombre, params, X_tr, y_tr)
               for nombre, params in configuraciones.items()}

    # Las decisiones preceden al acceso a las etiquetas y puntuaciones de test.
    umbrales = []
    ap_validacion = {}
    for nombre, modelo in modelos.items():
        ys = puntuacion(modelo, X.iloc[idx_va])
        umbral, f1 = umbral_f1_validacion(y.iloc[idx_va], ys)
        umbrales.append({"modelo": nombre, "umbral": umbral, "F1_validacion": f1})
        ap_validacion[nombre] = average_precision_score(y.iloc[idx_va], ys)
    ganador = max(ap_validacion, key=ap_validacion.get)
    umbral_por_modelo = {fila["modelo"]: fila["umbral"] for fila in umbrales}

    filas, intervalos, matrices, curvas, scores_test, filas_umbral = [], [], {}, {}, {}, []
    for nombre, modelo in modelos.items():
        matrices[nombre], curvas[nombre] = {}, {}
        for particion, idx in particiones.items():
            yt = y.iloc[idx].to_numpy()
            ys = np.asarray(puntuacion(modelo, X.iloc[idx]))
            yp = predecir(modelo, X.iloc[idx])
            fila = metricas(yt, yp, ys)
            fila.update(modelo=nombre, particion=particion)
            filas.append(fila)
            ic = bootstrap_ic(yt, yp, ys, n_boot=n_boot, semilla=semilla)
            ic["modelo"], ic["particion"] = nombre, particion
            intervalos.append(ic)
            matrices[nombre][particion] = confusion_matrix(yt, yp, labels=[0, 1]).tolist()
            precision, recall, _ = precision_recall_curve(yt, ys)
            fpr, tpr, _ = roc_curve(yt, ys)
            curvas[nombre][particion] = {
                "pr_precision": precision.tolist(), "pr_recall": recall.tolist(),
                "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
            }
            if particion == "test":
                scores_test[nombre] = ys
                yp_umbral = (ys >= umbral_por_modelo[nombre]).astype(int)
                fila_umbral = metricas(yt, yp_umbral, ys)
                fila_umbral.update(modelo=nombre, particion="test",
                                   umbral=umbral_por_modelo[nombre])
                filas_umbral.append(fila_umbral)
                matrices[nombre]["test_umbral_f1_validacion"] = confusion_matrix(
                    yt, yp_umbral, labels=[0, 1]
                ).tolist()

    y_test = y.iloc[idx_te].to_numpy()
    comparaciones = []
    for a, b in combinations(configuraciones, 2):
        for metrica, funcion in (("AP", average_precision_score), ("ROC_AUC", roc_auc_score)):
            puntual, media, inferior, superior = bootstrap_diferencia(
                y_test, scores_test[a], scores_test[b], funcion,
                n_boot=n_boot, semilla=semilla,
            )
            comparaciones.append({"modelo_a": a, "modelo_b": b, "metrica": metrica,
                                  "diferencia_puntual": puntual, "media_bootstrap": media,
                                  "ic_inferior": inferior, "ic_superior": superior,
                                  "n_boot": n_boot})

    resultado = {
        "cv_intervalos": cv_intervalos,
        "metricas": pd.DataFrame(filas),
        "intervalos": pd.concat(intervalos, ignore_index=True),
        "umbrales": pd.DataFrame(umbrales),
        "metricas_test_umbral": pd.DataFrame(filas_umbral),
        "comparaciones_test": pd.DataFrame(comparaciones),
        "matrices": matrices, "curvas": curvas,
        "resumen": {"ganador_validacion": ganador, "AP_validacion": ap_validacion,
                     "n_boot": n_boot, "semilla": semilla,
                     "cv": {"n_folds": N_FOLDS_CV, "particion": "entrenamiento",
                            "intervalo": "t de Student al 95% con desviacion muestral"},
                     "bootstrap": "estratificado pareado, condicionado a puntuaciones fijas",
                     "limitacion_prueba": "La particion de prueba fue consultada en el analisis historico."},
    }
    if ejecutar_ablaciones:
        ablaciones = {}
        for nombre_ablacion, opciones in (
            ("sin_pagevalues", {"incluir_pagevalues": False}),
            ("sin_specialday", {"incluir_specialday": False}),
        ):
            modelo = ajustar(ganador, configuraciones[ganador], X_tr, y_tr, **opciones)
            ablaciones[nombre_ablacion] = pd.DataFrame([
                metricas_particion(modelo, X.iloc[idx], y.iloc[idx], particion, ganador)
                for particion, idx in (("entrenamiento", idx_tr), ("validacion", idx_va))
            ])
        resultado["ablaciones"] = ablaciones
        resultado["resumen"]["ablaciones"] = {
            nombre: {"AP_validacion": float(tabla.loc[tabla.particion == "validacion", "AP"].iloc[0]),
                     "delta_AP": float(ap_validacion[ganador] - tabla.loc[tabla.particion == "validacion", "AP"].iloc[0])}
            for nombre, tabla in ablaciones.items()
        }

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    for nombre in ("cv_intervalos", "metricas", "intervalos", "umbrales",
                   "metricas_test_umbral", "comparaciones_test"):
        resultado[nombre].to_csv(salida / f"{nombre}.csv", index=False)
    for nombre in ("matrices", "curvas", "resumen"):
        (salida / f"{nombre}.json").write_text(
            json.dumps(resultado[nombre], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    for nombre, tabla in resultado.get("ablaciones", {}).items():
        tabla.to_csv(salida / f"ablacion_{nombre}.csv", index=False)
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=CSV)
    parser.add_argument("--configuraciones", type=Path, default=SALIDA / "mejores_configuraciones.csv")
    parser.add_argument("--salida", type=Path, default=SALIDA_FAMILIAS)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--sin-ablaciones", action="store_true")
    args = parser.parse_args()
    evaluar_familias(args.datos, args.configuraciones, args.salida, args.n_boot,
                     args.semilla, args.rapido, not args.sin_ablaciones)
