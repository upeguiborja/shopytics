"""Ejecuta las etapas reproducibles; --desde reutiliza resultados previos."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from proyecto_ml import RAIZ, SEMILLA, auditar, cargar_datos, particionar, resumen_particiones, tabla_auditoria

ETAPAS = ("datos", "busqueda", "evaluacion", "sensibilidad", "specialday", "variables", "reduccion", "figuras")


def main(rapido=False, desde="datos"):
    if desde not in ETAPAS:
        raise ValueError(f"Etapa desconocida: {desde}")
    salida = RAIZ / ("resultados_rapido" if rapido else "resultados")
    salida.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos()
    df_u, X, y, itr, iva, ite = particionar(datos)
    X_tr, y_tr, X_va, y_va = X.iloc[itr], y.iloc[itr], X.iloc[iva], y.iloc[iva]
    configuraciones = salida / "mejores_configuraciones.csv"
    duraciones = {}
    manifiesto_ruta = salida / "manifiesto.json"
    anterior = json.loads(manifiesto_ruta.read_text(encoding="utf-8")) if manifiesto_ruta.exists() else {}
    etapas_anteriores = anterior.get("duracion_etapas_seg", {})

    def etapa(nombre, funcion):
        if ETAPAS.index(nombre) < ETAPAS.index(desde):
            return
        print(f"Etapa: {nombre}", flush=True)
        inicio = time.perf_counter()
        funcion()
        duraciones[nombre] = round(time.perf_counter() - inicio, 3)

    def auditar_datos():
        (salida / "auditoria.json").write_text(json.dumps(auditar(datos), indent=2, ensure_ascii=False), encoding="utf-8")
        tabla_auditoria(datos).to_csv(salida / "tabla_variables.csv", index=False)
        resumen_particiones(y, itr, iva, ite).to_csv(salida / "resumen_particiones.csv", index=False)
        mensual = df_u.iloc[itr].groupby("Month", observed=True)["Revenue"].agg(["size", "mean"])
        mensual.columns = ["sesiones", "tasa_compra"]
        mensual.to_csv(salida / "cobertura_mensual_train.csv")

    def seleccionar():
        from seleccion import ejecutar_seleccion
        ejecutar_seleccion(X_tr, y_tr, X_va, y_va, salida, rapido=rapido)
        from busqueda import mallas_ampliadas
        from proyecto_ml import contar_combinaciones, construir_modelos
        _, mallas = mallas_ampliadas(construir_modelos, rapido=rapido)
        contar_combinaciones(mallas).to_csv(salida / "combinaciones_malla.csv", index=False)

    def evaluar():
        from evaluacion import evaluar_familias
        evaluar_familias(ruta_configuraciones=configuraciones, salida=salida / "evaluacion_familias", rapido=rapido)

    def sensibilidad():
        from sensibilidad import ejecutar_sensibilidad
        ejecutar_sensibilidad(X_tr, y_tr, X_va, y_va, salida, rapido=rapido)

    def specialday():
        from specialday import ejecutar_specialday
        ejecutar_specialday(X_tr, y_tr, X_va, y_va, configuraciones, salida, rapido=rapido)

    def variables():
        from analisis_variables import ejecutar_analisis
        ejecutar_analisis(X_tr, y_tr, X_va, y_va, configuraciones, salida / "variables", rapido=rapido)

    def reduccion():
        from experimentos_reduccion import cargar_configuraciones, ejecutar_experimentos
        ejecutar_experimentos(X_tr, y_tr, X_va, y_va, X.iloc[ite], y.iloc[ite],
                              cargar_configuraciones(configuraciones), salida / "reduccion", rapido=rapido)

    def figuras():
        from figuras import generar_figuras
        generar_figuras(salida, rapido=rapido)

    for nombre, funcion in zip(ETAPAS, (auditar_datos, seleccionar, evaluar, sensibilidad,
                                      specialday, variables, reduccion, figuras)):
        etapa(nombre, funcion)

    versiones = {}
    for paquete in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "umap-learn", "numba"):
        try:
            versiones[paquete] = metadata.version(paquete)
        except metadata.PackageNotFoundError:
            versiones[paquete] = None
    manifiesto = {"sha256_datos": auditar(datos)["sha256"], "semilla": SEMILLA,
                  "fecha_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                  "versiones": versiones, "rapido": rapido, "desde": desde,
                  "n_train": len(itr), "n_validacion": len(iva), "n_test": len(ite),
                  "duracion_etapas_seg": {**etapas_anteriores, **duraciones}}
    manifiesto_ruta.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rapido", action="store_true", help="prueba de humo con calculos reducidos")
    parser.add_argument("--desde", choices=ETAPAS, default="datos")
    args = parser.parse_args()
    main(args.rapido, args.desde)
