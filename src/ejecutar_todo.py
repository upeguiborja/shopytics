"""Punto de entrada unico del proyecto.

Uso:
  python src/ejecutar_todo.py            # busqueda completa (mallas ampliadas)
  python src/ejecutar_todo.py --rapido   # prueba de humo con mallas reducidas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from proyecto_ml import RAIZ, cargar_datos, particionar  # noqa: E402
from seleccion import ejecutar_seleccion  # noqa: E402


def main(rapido: bool) -> None:
    datos = cargar_datos()
    _, X, y, idx_tr, idx_va, _ = particionar(datos)
    X_tr, y_tr = X.loc[idx_tr], y.loc[idx_tr].astype(int)
    X_va, y_va = X.loc[idx_va], y.loc[idx_va].astype(int)

    salida = RAIZ / ("resultados_rapido" if rapido else "resultados")
    cv, val, mejores, criterio = ejecutar_seleccion(
        X_tr, y_tr, X_va, y_va, salida, rapido=rapido
    )
    print(f"Salida: {salida}")
    print(mejores[["modelo", "rango_cv", "AP_CV", "AP"]].to_string(index=False))
    print(f"Ganador: {criterio['modelo']} (AP validacion = {criterio['AP_validacion']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true", help="mallas reducidas para prueba de humo")
    args = parser.parse_args()
    main(args.rapido)
