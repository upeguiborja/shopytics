"""Figuras del analisis de reduccion de dimension (PCA y UMAP)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _argumentos():
    p = argparse.ArgumentParser(description="Genera las figuras de PCA y UMAP.")
    p.add_argument("--resultados", type=Path, default=ROOT / "resultados_corregidos")
    p.add_argument(
        "--figuras",
        type=Path,
        default=ROOT / "informe_entregable_II_latex" / "figuras",
    )
    return p.parse_args()


ARGS = _argumentos()
RES = ARGS.resultados
FIG = ARGS.figuras
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
    }
)


def fig_varianza():
    var = pd.read_csv(RES / "pca_varianza.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(var["componente"], var["varianza_acumulada"], color="#2c7fb8", marker="o", markersize=2)
    ax.axhline(0.95, color="#cb181d", ls="--", linewidth=1, label="95 % de varianza")
    ax.axvline(28, color="gray", ls=":", linewidth=1, label="28 componentes")
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza acumulada")
    ax.set_title("PCA: varianza acumulada")
    ax.legend(fontsize=7)
    fig.savefig(FIG / "fig11_pca_varianza.png")
    plt.close(fig)


def fig_umap():
    coords = json.loads((RES / "umap_2d.json").read_text(encoding="utf-8"))
    xy = np.asarray(coords["test"])
    y = np.asarray(coords["y_test"])
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    for valor, color, etiqueta in [(0, "#9ecae1", "Sin compra"), (1, "#cb181d", "Compra")]:
        m = y == valor
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.55, color=color, label=etiqueta, linewidths=0)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP 2D en prueba")
    ax.legend(fontsize=7, markerscale=2)
    fig.savefig(FIG / "fig12_umap_2d.png")
    plt.close(fig)


def fig_comparacion():
    base = pd.read_csv(RES / "metricas_particiones.csv")
    base = base[(base.particion == "test") & (base.modelo.isin(["RandomForest", "MLP"]))]
    base = base.set_index("modelo")["AP"]
    red = pd.read_csv(RES / "reduccion_dimension.csv")
    red = red[red.particion == "test"]
    u2 = pd.read_csv(RES / "reduccion_umap2.csv")
    u2 = u2[u2.particion == "test"]

    etiquetas = ["Base", "PCA", "UMAP 10", "UMAP 2"]
    modelos = ["RandomForest", "MLP"]
    colores = {"RandomForest": "#2c7fb8", "MLP": "#d95f02"}
    x = np.arange(len(etiquetas))
    ancho = 0.35
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for k, m in enumerate(modelos):
        valores = [
            base.loc[m],
            red[(red.modelo == f"PCA-{m}")]["AP"].iloc[0],
            red[(red.modelo == f"UMAP10-{m}")]["AP"].iloc[0],
            u2[(u2.modelo == f"UMAP2-{m}")]["AP"].iloc[0],
        ]
        ax.bar(x + (k - 0.5) * ancho, valores, width=ancho,
               color=colores[m], label="Random Forest" if m == "RandomForest" else "MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("AP en prueba")
    ax.set_xlabel("Representación")
    ax.set_title("Efecto de la reducción sobre el AP")
    ax.legend(fontsize=7)
    fig.savefig(FIG / "fig13_comparacion_reduccion.png")
    plt.close(fig)


fig_varianza()
fig_umap()
fig_comparacion()
print("Figuras en", FIG)
