"""Nombres legibles para mostrar tablas y figuras.

Los identificadores internos (claves de familia, nombres de parámetros de
scikit-learn, columnas de los JSON) no cambian; estas funciones solo traducen lo
que se muestra en los notebooks.
"""

import numpy as np
import pandas as pd

FAMILIAS = {
    "Regresion_logistica": "Regresión logística",
    "kNN": "k-NN",
    "RandomForest": "Random Forest",
    "MLP": "MLP",
    "SVM": "SVM",
}
PARTICIONES = {"entrenamiento": "Entrenamiento", "validacion": "Validación", "prueba": "Prueba"}
METRICAS = {
    "AP": "AP",
    "ROC_AUC": "ROC-AUC",
    "precision": "Precisión",
    "recall": "Recall",
    "F1": "F1",
    "balanced_accuracy": "Exactitud balanceada",
    "accuracy": "Exactitud",
}
VARIANTES = {"completo": "Completo", "sin_candidatas": "Sin candidatas",
             "sin_redundantes": "Sin redundantes"}
PARAMETROS = {
    "pre__num__log__usar_log": "log1p",
    "clf__C": "C",
    "clf__class_weight": "Pesos de clase",
    "clf__n_neighbors": "Vecinos (k)",
    "clf__weights": "Ponderación",
    "clf__p": "Distancia",
    "clf__max_depth": "Profundidad máxima",
    "clf__max_features": "Variables por división",
    "clf__min_samples_leaf": "Mínimo de sesiones por hoja",
    "clf__hidden_layer_sizes": "Capas ocultas",
    "clf__alpha": "alpha",
    "clf__learning_rate_init": "Tasa de aprendizaje inicial",
    "clf__kernel": "Kernel",
    "clf__gamma": "gamma",
}
COLUMNAS = {
    **METRICAS,
    "modelo": "Modelo", "modelo_a": "Modelo A", "modelo_b": "Modelo B",
    "particion": "Partición", "metrica": "Métrica", "params": "Configuración",
    "variante": "Variante", "excluidas": "Variables retiradas", "delta_AP": "ΔAP validación",
    "AP_CV": "AP-CV", "std_CV": "Desv. CV", "rango_cv": "Puesto en CV",
    "n_folds": "Pliegues", "media": "Media", "desviacion": "Desviación",
    "error_estandar": "Error estándar", "ic_inferior": "IC inferior",
    "ic_superior": "IC superior", "media_bootstrap": "Media bootstrap",
    "diferencia_puntual": "Diferencia", "n_boot": "Réplicas", "regla": "Regla",
    "umbral": "Umbral", "AP_con": "AP con la variable", "AP_sin": "AP sin la variable",
    "ROC_AUC_sin": "ROC-AUC sin la variable", "delta_con_menos_sin": "ΔAP (con − sin)",
    "metodo": "Método", "componentes": "Componentes", "vecinos": "Vecinos",
    "mes": "Mes", "sesiones": "Sesiones", "compras": "Compras", "variable": "Variable",
    "numérico": "AP numérico", "sin": "AP sin SpecialDay", "onehot": "AP one-hot",
    "AP_CV_numerico": "AP-CV numérico", "AP_CV_onehot": "AP-CV one-hot",
    "con_menos_sin": "ΔAP (con − sin)", "onehot_menos_numérico": "ΔAP (one-hot − numérico)",
    "ic_AP_inferior": "IC AP inferior", "ic_AP_superior": "IC AP superior",
    "reduccion": "Reducción (%)", "AP_validacion": "AP validación",
    "predictores": "Predictores", "delta_AP_CV": "ΔAP-CV",
}
METODOS = {"UMAP2D": "UMAP 2D"}
MESES = {
    "Jan": "Enero", "Feb": "Febrero", "Mar": "Marzo", "Apr": "Abril", "May": "Mayo",
    "June": "Junio", "Jul": "Julio", "Aug": "Agosto", "Sep": "Septiembre",
    "Oct": "Octubre", "Nov": "Noviembre", "Dec": "Diciembre",
}
_VALORES = {**FAMILIAS, **PARTICIONES, **VARIANTES, **METODOS}


def valor_legible(parametro, valor):
    """Escribe un valor de hiperparámetro como se leería en una tabla."""
    if valor is None:
        return "sin límite" if parametro.endswith("max_depth") else "ninguno"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if parametro.endswith("__p"):
        return {1: "Manhattan", 2: "euclídea"}[valor]
    if parametro.endswith("__kernel"):
        return {"linear": "lineal", "rbf": "RBF"}.get(valor, valor)
    if isinstance(valor, list):
        valor = tuple(valor)
    if isinstance(valor, float):
        return np.format_float_positional(round(valor, 10), trim="-").replace(".", ",")
    return str(valor)


def describir_configuracion(params):
    """Resume un diccionario de hiperparámetros en una línea legible."""
    return "; ".join(f"{PARAMETROS.get(k, k)} {valor_legible(k, v)}"
                     for k, v in params.items())


def _traducir(valor):
    return _VALORES.get(valor, valor) if isinstance(valor, str) else valor


def legible(tabla):
    """Copia de la tabla con nombres de familias, particiones, métricas y
    columnas legibles, y la configuración escrita como texto."""
    tabla = tabla.copy()
    if isinstance(tabla, pd.Series):
        tabla = tabla.to_frame()
    if "params" in tabla.columns:
        tabla["params"] = tabla["params"].map(describir_configuracion)
    for columna in ("modelo", "modelo_a", "modelo_b", "particion", "variante"):
        if columna in tabla.columns:
            tabla[columna] = tabla[columna].map(_traducir)
    if "metrica" in tabla.columns:
        tabla["metrica"] = tabla["metrica"].map(lambda m: METRICAS.get(m, m))
    tabla = tabla.rename(index=lambda v: METRICAS.get(v, _traducir(v)) if isinstance(v, str) else v)
    tabla.index.names = [COLUMNAS.get(n, n) for n in tabla.index.names]
    tabla = tabla.rename(columns=lambda c: COLUMNAS.get(c, _traducir(c)) if isinstance(c, str) else c)
    if isinstance(tabla.columns, pd.MultiIndex):
        tabla.columns.names = [COLUMNAS.get(n, n) for n in tabla.columns.names]
    return tabla
