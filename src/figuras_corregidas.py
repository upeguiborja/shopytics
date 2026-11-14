import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _argumentos():
    p = argparse.ArgumentParser(
        description="Regenera las figuras del informe del Entregable II."
    )
    p.add_argument(
        "--resultados",
        type=Path,
        default=ROOT / "resultados_corregidos",
        help="Carpeta con las metricas corregidas (por defecto resultados_corregidos).",
    )
    p.add_argument(
        "--mallas",
        type=Path,
        default=None,
        help="Carpeta con cv_ext_*.csv (por defecto <resultados>/mallas_ext).",
    )
    p.add_argument(
        "--figuras",
        type=Path,
        default=ROOT / "informe_entregable_II_latex" / "figuras",
        help="Carpeta de salida de las figuras.",
    )
    return p.parse_args()


ARGS = _argumentos()
RES = ARGS.resultados
CVEXT = ARGS.mallas or (RES / "mallas_ext")
ORIG = ROOT / "resultados"
FIG = ARGS.figuras
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5})

MODELOS = ["RandomForest", "MLP", "SVM", "kNN", "Regresion_logistica"]
ETIQUETAS = {
    "RandomForest": "Random Forest",
    "MLP": "Red neuronal (MLP)",
    "SVM": "SVM (RBF)",
    "kNN": "k-NN",
    "Regresion_logistica": "Regresión logística",
}
COLORES = {
    "RandomForest": "#2c7fb8",
    "MLP": "#d95f02",
    "SVM": "#7570b3",
    "kNN": "#1b9e77",
    "Regresion_logistica": "#e7298a",
}


def combinar_cv():
    if not CVEXT.is_dir():
        raise SystemExit(
            f"No existe la carpeta de mallas extendidas: {CVEXT}\n"
            "Use --mallas o copie los cv_ext_*.csv en resultados_corregidos/mallas_ext."
        )
    frames = []
    for m, f in [
        ("Regresion_logistica", "cv_ext_Regresion_logistica.csv"),
        ("kNN", "cv_ext_kNN.csv"),
        ("RandomForest", "cv_ext_RandomForest.csv"),
        ("MLP", "cv_ext_MLP.csv"),
    ]:
        d = pd.read_csv(CVEXT / f)
        d["modelo"] = m
        frames.append(d)
    for f in ["cv_ext_SVM_lineal.csv", "cv_ext_SVM_rbf.csv"]:
        d = pd.read_csv(CVEXT / f)
        d["modelo"] = "SVM"
        frames.append(d)
    cv = pd.concat(frames, ignore_index=True)
    cv.to_csv(RES / "cv_todos_los_candidatos.csv", index=False)
    return cv


def fig_clases():
    aud = json.loads((ORIG / "auditoria.json").read_text(encoding="utf-8"))
    clases = aud["clases"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    etq = ["Sin compra", "Compra"]
    vals = [clases["False"], clases["True"]]
    barras = ax.bar(etq, vals, color=["#9ecae1", "#3182bd"])
    for b, v in zip(barras, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 120, f"{v:,}".replace(",", ".") + f" ({v/sum(vals):.1%})",
                ha="center", fontsize=8)
    ax.set_ylabel("Número de sesiones")
    ax.set_xlabel("Clase de Revenue")
    ax.set_ylim(0, max(vals) * 1.16)
    ax.set_title("Distribución de clases")
    fig.savefig(FIG / "fig01_distribucion_clases.png")
    plt.close(fig)


def fig_mensual():
    mes = pd.read_csv(ORIG / "cobertura_mensual_train.csv").sort_values("tasa_compra", ascending=False)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(mes["Month"], mes["sesiones"], color="#9ecae1", label="Sesiones")
    ax.set_xlabel("Mes de la sesión")
    ax.set_ylabel("Sesiones en entrenamiento")
    ax2 = ax.twinx()
    ax2.plot(mes["Month"], mes["tasa_compra"], "o-", color="#d95f02", label="Tasa de compra")
    ax2.set_ylabel("Tasa de compra")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
    ax.set_title("Cobertura mensual y tasa de compra")
    fig.savefig(FIG / "fig02_cobertura_mensual.png")
    plt.close(fig)


def panel(ax, cv, modelo, param, xlabel, orden, etiquetas, titulo):
    sub = cv[cv.modelo == modelo].copy()
    sub["clave"] = sub[param].fillna("sin límite").astype(str)
    agg = sub.groupby("clave").agg(
        ap=("mean_test_score", "mean"),
        std=("mean_test_score", "std"),
        train=("mean_train_score", "mean"),
    )
    agg = agg.loc[orden]
    x = np.arange(len(orden))
    ax.errorbar(x, agg["ap"], yerr=agg["std"], marker="o", capsize=3,
                color=COLORES[modelo], label="Validación (AP media)")
    ax.plot(x, agg["train"], "s--", alpha=0.45, color="gray", label="Entrenamiento")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=7, rotation=25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AP")
    ax.set_title(titulo, fontsize=9)
    ax.legend(fontsize=6.5)


def fig_hiperparametros():
    cv = combinar_cv()
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6))
    axes = axes.ravel()
    panel(axes[0], cv, "Regresion_logistica", "param_clf__C", "C (regularización L2)",
          ["0.01", "0.1", "1.0", "10.0", "100.0"], ["0,01", "0,1", "1", "10", "100"],
          ETIQUETAS["Regresion_logistica"])
    panel(axes[1], cv, "kNN", "param_clf__n_neighbors", "Número de vecinos (k)",
          ["5.0", "15.0", "31.0", "45.0", "61.0", "81.0", "101.0"], ["5", "15", "31", "45", "61", "81", "101"],
          ETIQUETAS["kNN"])
    panel(axes[2], cv, "RandomForest", "param_clf__max_depth", "Profundidad máxima",
          ["12.0", "20.0", "30.0", "sin límite"], ["12", "20", "30", "sin límite"],
          ETIQUETAS["RandomForest"])
    panel(axes[3], cv, "MLP", "param_clf__hidden_layer_sizes", "Neuronas por capa",
          ["(32,)", "(64,)", "(64, 32)"], ["32", "64", "64-32"], ETIQUETAS["MLP"])
    panel(axes[4], cv, "SVM", "param_clf__C", "C (SVM, kernel RBF)",
          ["0.1", "1.0", "10.0", "30.0", "100.0"], ["0,1", "1", "10", "30", "100"],
          "SVM (RBF): regularización")
    ax = axes[5]
    sub = cv[(cv.modelo == "SVM")].groupby("param_clf__kernel")["mean_test_score"].agg(["mean", "std"])
    sub = sub.loc[["linear", "rbf"]]
    ax.bar(["Lineal", "RBF"], sub["mean"], yerr=sub["std"], capsize=4, color="#7570b3")
    ax.set_xlabel("Kernel")
    ax.set_ylabel("AP (validación cruzada)")
    ax.set_title("SVM: efecto del kernel", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_efecto_hiperparametros.png")
    plt.close(fig)


def fig_train_val():
    met = pd.read_csv(RES / "metricas_particiones.csv")
    met = met[met.particion.isin(["entrenamiento", "validacion"])]
    piv = met.pivot_table(index="modelo", columns="particion", values="AP").loc[MODELOS]
    x = np.arange(len(MODELOS))
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.bar(x - 0.2, piv["entrenamiento"], width=0.4, label="Entrenamiento", color="#bdbdbd")
    ax.bar(x + 0.2, piv["validacion"], width=0.4, label="Validación", color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in MODELOS], fontsize=7, rotation=18)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_xlabel("Modelo")
    ax.legend(fontsize=7)
    ax.set_title("AP en entrenamiento frente a validación")
    fig.savefig(FIG / "fig04_train_vs_validacion.png")
    plt.close(fig)


def fig_pr_roc():
    cur = json.loads((RES / "curvas_validacion.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    for m in MODELOS:
        ax.plot(cur[m]["pr_recall"], cur[m]["pr_precision"], color=COLORES[m], label=ETIQUETAS[m])
    ax.axhline(0.156, color="k", ls=":", linewidth=1, label="Tasa base (0,156)")
    ax.set_xlabel("Recall (clase compra)")
    ax.set_ylabel("Precisión (clase compra)")
    ax.set_title("Curvas precisión-recall en validación")
    ax.legend(fontsize=6.3, loc="upper right")
    fig.savefig(FIG / "fig05_curvas_pr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    for m in MODELOS:
        ax.plot(cur[m]["roc_fpr"], cur[m]["roc_tpr"], color=COLORES[m], label=ETIQUETAS[m])
    ax.plot([0, 1], [0, 1], "k:", linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC en validación")
    ax.legend(fontsize=6.3, loc="lower right")
    fig.savefig(FIG / "fig06_curvas_roc.png")
    plt.close(fig)


def fig_confusion():
    mc = json.loads((RES / "matrices_confusion_test.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 5, figsize=(12.0, 2.7))
    for ax, m in zip(axes, MODELOS):
        cm = np.array(mc[m])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Sí"], fontsize=7)
        ax.set_yticklabels(["No", "Sí"], fontsize=7)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(ETIQUETAS[m], fontsize=7.5)
    fig.suptitle("Matrices de confusión en prueba (umbral nativo)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_matrices_confusion_test.png")
    plt.close(fig)


def fig_ap_ic():
    ic = pd.read_csv(RES / "metricas_intervalos_confianza.csv")
    ic = ic[(ic.metrica == "AP") & (ic.particion.isin(["validacion", "test"]))]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(MODELOS))
    ancho = 0.35
    for k, part in enumerate(["validacion", "test"]):
        sub = ic[ic.particion == part].set_index("modelo").loc[MODELOS]
        err = np.vstack([sub["media_bootstrap"] - sub["ic_inferior"],
                         sub["ic_superior"] - sub["media_bootstrap"]])
        ax.bar(x + (k - 0.5) * ancho, sub["media_bootstrap"], width=ancho,
               yerr=err, capsize=3, label="Validación" if part == "validacion" else "Prueba",
               color=["#2c7fb8", "#fdae6b"][k])
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in MODELOS], fontsize=7, rotation=18)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_xlabel("Modelo")
    ax.legend(fontsize=7)
    ax.set_title("AP con intervalos de confianza del 95 %")
    fig.savefig(FIG / "fig08_ap_intervalos.png")
    plt.close(fig)


def fig_pagevalues():
    cmp = pd.read_csv(RES / "comparacion_pagevalues.csv").set_index("modelo").loc[MODELOS]
    x = np.arange(len(MODELOS))
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.bar(x - 0.2, cmp["AP_con_PageValues"], width=0.4, label="Con PageValues", color="#2c7fb8")
    ax.bar(x + 0.2, cmp["AP_sin_PageValues"], width=0.4, label="Sin PageValues", color="#cb181d")
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in MODELOS], fontsize=7, rotation=18)
    ax.set_ylabel("AP en validación")
    ax.set_xlabel("Modelo")
    ax.legend(fontsize=7)
    ax.set_title("Sensibilidad al retirar PageValues")
    fig.savefig(FIG / "fig09_sensibilidad_pagevalues.png")
    plt.close(fig)


def fig_specialday():
    abl = pd.read_csv(RES / "comparacion_specialday.csv").set_index("modelo").loc[MODELOS]
    x = np.arange(len(MODELOS))
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.bar(x, abl["delta_AP"], color="#756bb1")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in MODELOS], fontsize=7, rotation=18)
    ax.set_ylabel("Cambio en AP (validación)")
    ax.set_xlabel("Modelo")
    ax.set_title("Ablación: retirar SpecialDay")
    fig.savefig(FIG / "fig10_ablacion_specialday.png")
    plt.close(fig)


fig_clases()
fig_mensual()
fig_hiperparametros()
fig_train_val()
fig_pr_roc()
fig_confusion()
fig_ap_ic()
fig_pagevalues()
fig_specialday()
print("Figuras en", FIG)
