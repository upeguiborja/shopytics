"""Seleccion del modelo final (solo validacion; el conjunto de prueba no participa).

Pasos:
  - Por familia: busqueda CV sobre entrenamiento (busqueda.py).
  - Top-3 por AP-CV pasa a validacion.
  - Ganador global: mayor AP en validacion.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from busqueda import (
    buscar_familia,
    evaluar_candidatos_validacion,
    seleccionar_top_k,
)
from shopytics.search import mallas_ampliadas


def ejecutar_seleccion(X_tr, y_tr, X_va, y_va, salida: Path, rapido: bool = False):
    salida.mkdir(parents=True, exist_ok=True)
    modelos, mallas = mallas_ampliadas(rapido=rapido)

    cv_todos, val_todos = [], []
    for nombre, malla in mallas.items():
        modelo = modelos[nombre]
        cv_df = buscar_familia(nombre, modelo, malla, X_tr, y_tr)
        cv_todos.append(cv_df)

        top = seleccionar_top_k(cv_df)
        params_por_fila = {i: ast.literal_eval(p) for i, p in zip(top.index, top["params"])}
        val_df = evaluar_candidatos_validacion(
            nombre, modelo, top, X_tr, y_tr, X_va, y_va, params_por_fila
        )
        val_todos.append(val_df)

    cv_total = pd.concat(cv_todos, ignore_index=True)
    val_total = pd.concat(val_todos, ignore_index=True)

    cv_total.to_csv(salida / "cv_todos_los_candidatos.csv", index=False)
    val_total.to_csv(salida / "validacion_top3.csv", index=False)

    mejores = (
        val_total.sort_values("AP", ascending=False)
        .groupby("modelo", as_index=False)
        .head(1)
        .sort_values("AP", ascending=False)
        .reset_index(drop=True)
    )
    mejores.to_csv(salida / "mejores_configuraciones.csv", index=False)

    ganador = mejores.iloc[0]
    criterio = {
        "modelo": ganador["modelo"],
        "criterio": "mayor AP en validacion entre top-3 por AP-CV de cada familia; la prueba no participa",
        "AP_validacion": float(ganador["AP"]),
        "params": ganador["params"],
    }
    (salida / "ganador.json").write_text(json.dumps(criterio, indent=2, ensure_ascii=False), encoding="utf-8")
    return cv_total, val_total, mejores, criterio
