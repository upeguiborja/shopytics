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


def _argumentos(argv=None):
    p = argparse.ArgumentParser(description="Genera las figuras de PCA y UMAP.")
    p.add_argument("--resultados", type=Path, default=ROOT / "resultados_corregidos")
    p.add_argument(
        "--figuras",
        type=Path,
        default=ROOT / "informe_entregable_II_latex" / "figuras",
    )
    return p.parse_args(argv)


RES = ROOT / "resultados_corregidos"
FIG = ROOT / "informe_entregable_II_latex" / "figuras"

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
    alcanzada = var.loc[var["varianza_acumulada"] >= 0.95, "componente"]
    if alcanzada.empty:
        raise ValueError("La varianza acumulada no alcanza el 95 %.")
    componentes = alcanzada.iloc[0]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(var["componente"], var["varianza_acumulada"], color="#2c7fb8", marker="o", markersize=2)
    ax.axhline(0.95, color="#cb181d", ls="--", linewidth=1, label="95 % de varianza")
    ax.axvline(componentes, color="gray", ls=":", linewidth=1, label=f"{componentes} componentes")
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


def fig_variables_top10():
    variables = pd.read_csv(RES / "analisis_variables.csv")
    variables = variables.sort_values("auc_univariado", ascending=False).head(10)
    colores = {"mantener": "#2c7fb8", "revisar": "#d95f02", "eliminar o combinar": "#cb181d"}
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    barras = ax.barh(variables["variable"], variables["auc_univariado"],
                     color=[colores.get(r, "#969696") for r in variables["recomendacion"]])
    for barra, im in zip(barras, variables["informacion_mutua"]):
        ax.text(barra.get_width() + 0.01, barra.get_y() + barra.get_height() / 2,
                f"IM {im:.3f}", va="center", fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, variables["auc_univariado"].max() + 0.18))
    ax.set_xlabel("AUC univariado")
    ax.set_title("Variables con mayor AUC univariado")
    for recomendacion in variables["recomendacion"].unique():
        etiqueta = "candidata a eliminar o combinar" if recomendacion == "eliminar o combinar" else recomendacion
        ax.barh([], [], color=colores.get(recomendacion, "#969696"), label=etiqueta)
    ax.legend(title="Recomendación descriptiva", fontsize=7, title_fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig14_variables_top10.png")
    plt.close(fig)


def generar(resultados, mallas=None, figuras=None):
    global RES, FIG
    RES = Path(resultados)
    FIG = Path(figuras) if figuras is not None else ROOT / "informe_entregable_II_latex" / "figuras"
    FIG.mkdir(parents=True, exist_ok=True)
    fig_varianza()
    fig_umap()
    fig_comparacion()
    fig_variables_top10()
    print("Figuras en", FIG)


def main(argv=None):
    args = _argumentos(argv)
    generar(args.resultados, figuras=args.figuras)


if __name__ == "__main__":
    main()
