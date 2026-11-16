"""Evaluacion reproducible de configuraciones seleccionadas, sin busqueda.

Los intervalos son condicionales a modelos fijos y la particion de prueba ya
se consulto historicamente, por lo que no constituye una prueba independiente.
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
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_curve

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


def cargar_ganador(salida: Path = SALIDA):
    ganador = json.loads((salida / "ganador.json").read_text(encoding="utf-8"))
    nombre = ganador["modelo"]
    params = ast.literal_eval(ganador["params"])
    return nombre, params


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

    comparaciones = []
    for a, b in combinations(configuraciones, 2):
        puntual, media, inferior, superior = bootstrap_diferencia(
            y.iloc[idx_te].to_numpy(), scores_test[a], scores_test[b],
            average_precision_score, n_boot=n_boot, semilla=semilla,
        )
        comparaciones.append({"modelo_a": a, "modelo_b": b, "diferencia_AP": puntual,
                              "media_bootstrap": media, "ic_inferior": inferior,
                              "ic_superior": superior, "n_boot": n_boot})

    resultado = {
        "metricas": pd.DataFrame(filas),
        "intervalos": pd.concat(intervalos, ignore_index=True),
        "umbrales": pd.DataFrame(umbrales),
        "metricas_test_umbral": pd.DataFrame(filas_umbral),
        "comparaciones_ap_test": pd.DataFrame(comparaciones),
        "matrices": matrices, "curvas": curvas,
        "resumen": {"ganador_validacion": ganador, "AP_validacion": ap_validacion,
                     "n_boot": n_boot, "semilla": semilla,
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
    for nombre in ("metricas", "intervalos", "umbrales", "metricas_test_umbral", "comparaciones_ap_test"):
        resultado[nombre].to_csv(salida / f"{nombre}.csv", index=False)
    for nombre in ("matrices", "curvas", "resumen"):
        (salida / f"{nombre}.json").write_text(
            json.dumps(resultado[nombre], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    for nombre, tabla in resultado.get("ablaciones", {}).items():
        tabla.to_csv(salida / f"ablacion_{nombre}.csv", index=False)
    return resultado


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos(CSV)
    _, X, y, idx_tr, idx_va, idx_te = particionar(datos)
    X_tr, y_tr = X.loc[idx_tr], y.loc[idx_tr].astype(int)
    X_va, y_va = X.loc[idx_va], y.loc[idx_va].astype(int)
    X_te, y_te = X.loc[idx_te], y.loc[idx_te].astype(int)

    nombre, params = cargar_ganador()
    print(f"Ganador: {nombre} {params}")

    modelo = ajustar(nombre, params, X_tr, y_tr)

    # Metricas por particion (test solo se reporta, no participa en decisiones)
    filas = [
        metricas_particion(modelo, X_tr, y_tr, "entrenamiento", nombre),
        metricas_particion(modelo, X_va, y_va, "validacion", nombre),
        metricas_particion(modelo, X_te, y_te, "test", nombre),
    ]
    pd.DataFrame(filas).to_csv(SALIDA / "metricas_ganador.csv", index=False)

    # IC bootstrap por particion
    ic = []
    for part, Xp, yp in [("entrenamiento", X_tr, y_tr), ("validacion", X_va, y_va), ("test", X_te, y_te)]:
        tabla = bootstrap_ic(yp.to_numpy(), predecir(modelo, Xp), puntuacion(modelo, Xp))
        tabla["modelo"] = nombre
        tabla["particion"] = part
        ic.append(tabla)
    pd.concat(ic, ignore_index=True).to_csv(SALIDA / "intervalos_confianza_ganador.csv", index=False)

    # Matriz de confusion en prueba
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_te, predecir(modelo, X_te), labels=[0, 1])
    (SALIDA / "matriz_confusion_test.json").write_text(
        json.dumps({"modelo": nombre, "matriz": cm.tolist()}, indent=2), encoding="utf-8"
    )

    # Curvas PR/ROC en validacion
    ys = puntuacion(modelo, X_va)
    pr_p, pr_r, _ = precision_recall_curve(y_va, ys)
    fpr, tpr, _ = roc_curve(y_va, ys)
    (SALIDA / "curvas_validacion_ganador.json").write_text(
        json.dumps(
            {
                "modelo": nombre,
                "pr_precision": pr_p.tolist(),
                "pr_recall": pr_r.tolist(),
                "roc_fpr": fpr.tolist(),
                "roc_tpr": tpr.tolist(),
            }
        ),
        encoding="utf-8",
    )

    # Sensibilidad sin PageValues (hiperparametros fijos)
    m_sin_pv = ajustar(nombre, params, X_tr, y_tr, incluir_pagevalues=False)
    filas_sin_pv = [
        metricas_particion(m_sin_pv, X_tr, y_tr, "entrenamiento", nombre),
        metricas_particion(m_sin_pv, X_va, y_va, "validacion", nombre),
    ]
    pd.DataFrame(filas_sin_pv).to_csv(SALIDA / "sensibilidad_sin_pagevalues_ganador.csv", index=False)

    # Ablacion sin SpecialDay (hiperparametros fijos)
    m_sin_sd = ajustar(nombre, params, X_tr, y_tr, incluir_specialday=False)
    filas_sin_sd = [
        metricas_particion(m_sin_sd, X_tr, y_tr, "entrenamiento", nombre),
        metricas_particion(m_sin_sd, X_va, y_va, "validacion", nombre),
    ]
    pd.DataFrame(filas_sin_sd).to_csv(SALIDA / "ablacion_sin_specialday_ganador.csv", index=False)

    # Comparacion AP validacion con/sin PageValues y SpecialDay
    ap_con = filas[1]["AP"]
    ap_sin_pv = filas_sin_pv[1]["AP"]
    ap_sin_sd = filas_sin_sd[1]["AP"]
    resumen = {
        "modelo": nombre,
        "params": params,
        "AP_validacion_con_todas": ap_con,
        "AP_validacion_sin_PageValues": ap_sin_pv,
        "AP_validacion_sin_SpecialDay": ap_sin_sd,
        "delta_AP_sin_PageValues": ap_con - ap_sin_pv,
        "delta_AP_sin_SpecialDay": ap_con - ap_sin_sd,
    }
    (SALIDA / "resumen_evaluacion.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(pd.DataFrame(filas).to_string(index=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


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
