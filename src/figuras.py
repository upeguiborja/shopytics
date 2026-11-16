"""Genera las figuras del informe y del notebook a partir de resultados/.

Uso:
    python src/figuras.py [--resultados DIR] [--figuras-informe DIR] [--rapido]

Lee exclusivamente la carpeta de resultados del flujo actual. No usa
resultados_corregidos ni respaldos historicos. En modo rapido escribe las
figuras del informe bajo <resultados>/figuras_informe para no sobrescribir
las figuras definitivas del informe.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parents[1]
DIR_RES = RAIZ / "resultados"
DIR_INFORME = RAIZ / "informe_entregable_II_latex" / "figuras"

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

MESES_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MESES_COMPLETOS = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
MESES = {en: es for en, es in zip(MESES_EN, MESES_ES)}
MESES.update({en: es for en, es in zip(MESES_COMPLETOS, MESES_ES)})
ORDEN_MESES = {mes: i for i, mes in enumerate(MESES_ES)}
ORDEN_MESES.update({mes: i for i, mes in enumerate(MESES_EN)})
ORDEN_MESES.update({mes: i for i, mes in enumerate(MESES_COMPLETOS)})

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Formato en espanol
# ---------------------------------------------------------------------------
def _texto_entero(valor) -> str:
    return f"{float(valor):,.0f}".replace(",", ".")


def _texto_decimal(valor, decimales: int = 2) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    return f"{float(valor):.{decimales}f}".replace(".", ",")


def _texto_numero(valor, decimales: int = 5) -> str:
    """Numero corto con coma decimal y sin ceros sobrantes."""
    if isinstance(valor, str):
        try:
            valor = float(valor)
        except ValueError:
            return valor
    numero = float(valor)
    if numero.is_integer():
        return f"{int(numero):,}".replace(",", ".")
    texto = f"{numero:.{decimales}f}".rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def _texto_porcentaje(valor, decimales: int = 1) -> str:
    return f"{float(valor) * 100:.{decimales}f} %".replace(".", ",")


def _formateador_entero(x, _pos=None):
    return _texto_entero(x)


def _formateador_decimal(decimales: int = 2):
    def formatear(x, _pos=None):
        return _texto_decimal(x, decimales)

    return formatear


def _formateador_porcentaje(decimales: int = 0):
    def formatear(x, _pos=None):
        return _texto_porcentaje(x, decimales)

    return formatear


def _orden_valor(valor):
    if isinstance(valor, str) and valor.strip().lower() in {"sin límite", "sin limite", "none"}:
        return (1, float("inf"))
    try:
        return (0, float(valor))
    except (TypeError, ValueError):
        return (2, str(valor))


def _etiqueta_valor(valor) -> str:
    if isinstance(valor, str) and valor.strip().lower() in {"sin límite", "sin limite", "none"}:
        return "sin límite"
    return _texto_numero(valor)


def _mes_espanol(mes) -> str:
    texto = str(mes).strip()
    return MESES.get(texto, texto)


def _orden_mes(mes) -> int:
    texto = str(mes).strip()
    return ORDEN_MESES.get(texto, 99)


# ---------------------------------------------------------------------------
# Lectura estricta de resultados
# ---------------------------------------------------------------------------
def _ruta(resultados: Path, *partes: str) -> Path:
    return Path(resultados).joinpath(*partes)


def _leer_csv(resultados: Path, *partes: str, requeridas=()) -> pd.DataFrame:
    ruta = _ruta(resultados, *partes)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Falta {ruta}. Ejecute la etapa correspondiente de src/ejecutar_todo.py."
        )
    tabla = pd.read_csv(ruta)
    faltantes = [c for c in requeridas if c not in tabla.columns]
    if faltantes:
        raise ValueError(f"{ruta} no contiene las columnas {faltantes}")
    return tabla


def _leer_json(resultados: Path, *partes: str):
    ruta = _ruta(resultados, *partes)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Falta {ruta}. Ejecute la etapa correspondiente de src/ejecutar_todo.py."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def _leer_opcional(resultados: Path, *partes: str):
    ruta = _ruta(resultados, *partes)
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.suffix == ".json" else pd.read_csv(ruta)


def _localizar(resultados: Path, candidatos) -> Path:
    for partes in candidatos:
        ruta = _ruta(resultados, *partes)
        if ruta.is_file():
            return ruta
    opciones = ", ".join(str(_ruta(resultados, *partes)) for partes in candidatos)
    raise FileNotFoundError(
        f"No se encontro ninguna de estas salidas: {opciones}. "
        "Ejecute la etapa correspondiente de src/ejecutar_todo.py."
    )


def _columna(tabla: pd.DataFrame, candidatas, descripcion: str, ruta: Path | None = None) -> str:
    for columna in candidatas:
        if columna in tabla.columns:
            return columna
    destino = f" en {ruta}" if ruta is not None else ""
    raise ValueError(f"Falta la columna de {descripcion}{destino}; se esperaba una de {list(candidatas)}")


def _columnas_split(tabla: pd.DataFrame, tipo: str) -> list[str]:
    patron = re.compile(rf"^split(\d+)_{tipo}_score$")
    pares = []
    for columna in tabla.columns:
        coincidencia = patron.match(columna)
        if coincidencia:
            pares.append((int(coincidencia.group(1)), columna))
    if not pares:
        raise ValueError(f"La tabla de CV no contiene columnas split*_{tipo}_score")
    return [columna for _, columna in sorted(pares)]


def _guardar(fig, destino: Path, dpi: int) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=dpi)
    plt.close(fig)
    return destino


# ---------------------------------------------------------------------------
# Datos auxiliares
# ---------------------------------------------------------------------------
def clases_auditoria(auditoria: dict):
    clases = auditoria.get("clases", {})

    def buscar(candidatos, nombre):
        for clave in candidatos:
            if clave in clases:
                return int(clases[clave])
        raise ValueError(f"No se encontro la clase {nombre} en auditoria.json")

    negativos = buscar([False, "False", "0", 0], "0")
    positivos = buscar([True, "True", "1", 1], "1")
    return negativos, positivos


def resumen_parametro(cv: pd.DataFrame, modelo: str, parametro: str, filtro=None) -> pd.DataFrame:
    """AP media y desviacion entre folds para cada valor de un hiperparametro.

    La desviacion se calcula sobre las cinco medias de fold, promediando
    primero las configuraciones que comparten el valor del parametro. No se
    usa std_test_score, que mide dispersion entre configuraciones.
    """
    sub = cv[cv["modelo"] == modelo].copy()
    if filtro is not None:
        sub = sub[filtro(sub)]
    if sub.empty:
        raise ValueError(f"No hay filas de CV para {modelo}")
    if parametro not in sub.columns:
        raise ValueError(f"La tabla de CV no contiene {parametro}")
    sub[parametro] = sub[parametro].where(sub[parametro].notna(), "sin límite")
    columnas_test = _columnas_split(sub, "test")
    columnas_train = _columnas_split(sub, "train")

    filas = []
    for valor, grupo in sub.groupby(parametro, sort=False, dropna=False):
        test = grupo[columnas_test].to_numpy(dtype=float)
        train = grupo[columnas_train].to_numpy(dtype=float)
        media_fold = np.nanmean(test, axis=0)
        media_train_fold = np.nanmean(train, axis=0)
        desviacion = float(np.nanstd(media_fold, ddof=1)) if len(media_fold) > 1 else 0.0
        filas.append({
            "valor": valor,
            "ap": float(np.nanmean(media_fold)),
            "desviacion": desviacion,
            "train": float(np.nanmean(media_train_fold)),
        })
    tabla = pd.DataFrame(filas)
    tabla["_orden"] = tabla["valor"].map(_orden_valor)
    return (tabla.sort_values("_orden", kind="stable")
            .drop(columns="_orden").reset_index(drop=True))


def _modelos_presentes(tabla: pd.DataFrame) -> list[str]:
    disponibles = set(tabla["modelo"].astype(str))
    ordenados = [m for m in MODELOS if m in disponibles]
    return ordenados + sorted(disponibles - set(ordenados))


def _etiqueta_modelo(modelo: str) -> str:
    return ETIQUETAS.get(modelo, modelo.replace("_", " "))


def _cargar_barrido_umap(resultados: Path):
    """Devuelve el barrido UMAP y la etiqueta de la metrica de AP."""
    ruta = _ruta(resultados, "reduccion", "umap_barrido.csv")
    if ruta.is_file():
        return pd.read_csv(ruta), "AP de validación"

    # Compatibilidad con la primera version del experimento, que dejaba el
    # barrido UMAP en reduccion_cv.csv con AP de validacion cruzada.
    ruta = _ruta(resultados, "reduccion", "reduccion_cv.csv")
    if ruta.is_file():
        tabla = pd.read_csv(ruta)
        if "metodo" in tabla.columns:
            tabla = tabla[tabla["metodo"].astype(str).str.upper().str.contains("UMAP")]
        if "AP_validacion" in tabla.columns:
            return tabla, "AP de validación"
        if "AP" in tabla.columns:
            return tabla, "AP de validación"
        if "AP_CV" in tabla.columns:
            return tabla, "AP de CV"
    raise FileNotFoundError(
        f"Falta el barrido UMAP en {resultados / 'reduccion'}: "
        "se esperaba umap_barrido.csv o reduccion_cv.csv."
    )


def _selecciones(protocolo) -> list[dict]:
    if isinstance(protocolo, dict):
        seleccion = protocolo.get("seleccion", [])
        if isinstance(seleccion, list):
            return [fila for fila in seleccion if isinstance(fila, dict)]
    return []


# ---------------------------------------------------------------------------
# Figuras del informe
# ---------------------------------------------------------------------------
def figura_clases_cobertura(auditoria: dict, mensual: pd.DataFrame, destino: Path, dpi: int = 200) -> Path:
    negativos, positivos = clases_auditoria(auditoria)
    total = negativos + positivos
    fig, (ax_clases, ax_mes) = plt.subplots(
        1, 2, figsize=(9.2, 3.3), gridspec_kw={"width_ratios": [0.72, 1.65]}
    )

    valores = [negativos, positivos]
    barras = ax_clases.bar(["Sin compra", "Compra"], valores, color=["#9ecae1", "#3182bd"])
    for barra, valor in zip(barras, valores):
        ax_clases.text(
            barra.get_x() + barra.get_width() / 2,
            valor + total * 0.015,
            f"{_texto_entero(valor)} ({_texto_porcentaje(valor / total)})",
            ha="center", va="bottom", fontsize=8,
        )
    ax_clases.set_xlabel("Clase de Revenue")
    ax_clases.set_ylabel("Número de sesiones")
    ax_clases.set_ylim(0, max(valores) * 1.16)
    ax_clases.yaxis.set_major_formatter(FuncFormatter(_formateador_entero))
    ax_clases.set_title("(a) Distribución de clases")

    mes = mensual.copy()
    mes["Month"] = mes["Month"].astype(str).str.strip()
    mes["_orden"] = mes["Month"].map(_orden_mes)
    mes = mes.sort_values(["_orden", "Month"]).reset_index(drop=True)
    etiquetas_mes = mes["Month"].map(_mes_espanol)
    ax_mes.bar(etiquetas_mes, mes["sesiones"], color="#9ecae1", label="Sesiones")
    ax_mes.set_xlabel("Mes de la sesión")
    ax_mes.set_ylabel("Sesiones en entrenamiento")
    ax_mes.yaxis.set_major_formatter(FuncFormatter(_formateador_entero))

    ax_tasa = ax_mes.twinx()
    ax_tasa.plot(etiquetas_mes, mes["tasa_compra"], "o-", color="#d95f02",
                 label="Tasa de compra")
    ax_tasa.set_ylabel("Tasa de compra")
    ax_tasa.yaxis.set_major_formatter(FuncFormatter(_formateador_porcentaje(0)))
    ax_tasa.grid(False)
    ax_tasa.spines["right"].set_visible(True)

    manejadores_1, etiquetas_1 = ax_mes.get_legend_handles_labels()
    manejadores_2, etiquetas_2 = ax_tasa.get_legend_handles_labels()
    ax_mes.legend(manejadores_1 + manejadores_2, etiquetas_1 + etiquetas_2,
                  loc="upper right", fontsize=7)
    ax_mes.set_title("(b) Cobertura mensual y tasa de compra")
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def figura_hiperparametros(cv: pd.DataFrame, destino: Path, dpi: int = 200) -> Path:
    paneles = [
        ("RandomForest", "param_clf__max_depth", "Random Forest: profundidad máxima",
         "Profundidad máxima", None),
        ("RandomForest", "param_clf__min_samples_leaf", "Random Forest: hojas mínimas",
         "Hojas mínimas", None),
        ("MLP", "param_clf__alpha", "MLP: regularización α", "α", None),
        ("MLP", "param_clf__learning_rate_init", "MLP: tasa de aprendizaje",
         "Tasa de aprendizaje", None),
        ("SVM", "param_clf__C", "SVM (RBF): C", "C",
         lambda tabla: tabla["param_clf__kernel"].astype(str).str.lower() == "rbf"),
        ("kNN", "param_clf__n_neighbors", "k-NN: vecinos (k)", "Número de vecinos (k)", None),
        ("Regresion_logistica", "param_clf__C", "Regresión logística: C", "C", None),
    ]
    fig, ejes = plt.subplots(2, 4, figsize=(12.4, 6.1))
    ejes = ejes.ravel()
    for eje, (modelo, parametro, titulo, etiqueta_x, filtro) in zip(ejes, paneles):
        tabla = resumen_parametro(cv, modelo, parametro, filtro=filtro)
        x = np.arange(len(tabla))
        eje.errorbar(
            x, tabla["ap"].to_numpy(), yerr=tabla["desviacion"].to_numpy(),
            marker="o", capsize=3,
            color=COLORES.get(modelo, "C0"), label="Validación (AP media)", linewidth=1.2,
        )
        eje.plot(x, tabla["train"].to_numpy(), "s--", alpha=0.5, color="gray",
                 label="Entrenamiento", linewidth=1)
        eje.set_xticks(x)
        eje.set_xticklabels([_etiqueta_valor(v) for v in tabla["valor"]],
                            fontsize=7, rotation=30, ha="right")
        eje.set_xlabel(etiqueta_x)
        eje.set_ylabel("AP")
        eje.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(2)))
        eje.set_title(titulo, fontsize=9)
        eje.set_xlim(-0.6, len(tabla) - 0.4)
        eje.grid(axis="y", alpha=0.25)
    ejes[0].legend(fontsize=6.5, loc="best")
    ejes[-1].set_visible(False)
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def figura_pr_roc(curvas: dict, prevalencia: float, destino: Path, dpi: int = 200) -> Path:
    modelos = [m for m in MODELOS if m in curvas and "validacion" in curvas[m]]
    if not modelos:
        raise ValueError("curvas.json no contiene curvas de validacion para las familias")
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(10.6, 4.1))
    for modelo in modelos:
        curva = curvas[modelo]["validacion"]
        color = COLORES.get(modelo, "C0")
        ax_pr.plot(curva["pr_recall"], curva["pr_precision"], color=color,
                   label=_etiqueta_modelo(modelo), linewidth=1.4)
        ax_roc.plot(curva["roc_fpr"], curva["roc_tpr"], color=color,
                    label=_etiqueta_modelo(modelo), linewidth=1.4)

    if np.isfinite(prevalencia):
        ax_pr.axhline(prevalencia, color="black", linestyle=":", linewidth=1,
                      label=f"Tasa base ({_texto_decimal(prevalencia, 3)})")
    ax_pr.set_xlabel("Recall (clase compra)")
    ax_pr.set_ylabel("Precisión (clase compra)")
    ax_pr.set_title("(a) Curvas precisión-recall en validación")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.grid(alpha=0.25)

    ax_roc.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos")
    ax_roc.set_title("(b) Curvas ROC en validación")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.grid(alpha=0.25)

    manejadores, etiquetas = ax_pr.get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="lower center",
               ncol=min(6, len(etiquetas)), frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _guardar(fig, destino, dpi)


def figura_variables(variables: pd.DataFrame, destino: Path, dpi: int = 200) -> Path:
    columna_auc = _columna(variables, ["auc_univariado", "auc_cv", "AUC"],
                           "AUC univariado")
    top = variables.sort_values(columna_auc, ascending=False).head(10).iloc[::-1].copy()
    colores_recomendacion = {
        "mantener": "#2c7fb8",
        "revisar": "#d95f02",
        "eliminar o combinar": "#cb181d",
        "otra": "#969696",
    }

    def familia(valor):
        texto = str(valor).strip().lower()
        if texto.startswith("mantener"):
            return "mantener"
        if texto.startswith("revisar"):
            return "revisar"
        if texto.startswith("eliminar") or texto.startswith("combinar"):
            return "eliminar o combinar"
        return "otra"

    colores = [colores_recomendacion[familia(v)] for v in top.get("recomendacion", [""] * len(top))]
    fig, eje = plt.subplots(figsize=(7.4, 4.3))
    barras = eje.barh(top["variable"], top[columna_auc], color=colores, height=0.72)
    if "informacion_mutua" in top.columns:
        for barra, valor in zip(barras, top["informacion_mutua"]):
            eje.text(barra.get_width() + 0.008,
                     barra.get_y() + barra.get_height() / 2,
                     f"IM {_texto_decimal(valor, 3)}", va="center", fontsize=7)
    eje.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Azar (0,5)")
    eje.set_xlim(min(0.45, float(top[columna_auc].min()) - 0.02),
                 max(1.0, float(top[columna_auc].max()) + 0.10))
    eje.set_xlabel("AUC univariado")
    eje.set_ylabel("Variable")
    eje.set_title("Variables con mayor AUC univariado")
    eje.xaxis.set_major_formatter(FuncFormatter(_formateador_decimal(2)))
    eje.grid(axis="x", alpha=0.25)

    etiquetas_leyenda = {
        "mantener": "Mantener",
        "revisar": "Revisar",
        "eliminar o combinar": "Eliminar o combinar",
        "otra": "Sin recomendación",
    }
    presentes = []
    for valor in top.get("recomendacion", []):
        clave = familia(valor)
        if clave not in presentes:
            presentes.append(clave)
    proxy = [Patch(facecolor=colores_recomendacion[clave], label=etiquetas_leyenda[clave])
             for clave in presentes]
    manejadores, etiquetas = eje.get_legend_handles_labels()
    eje.legend(manejadores + proxy, etiquetas + [p.get_label() for p in proxy],
               fontsize=7, loc="lower right")
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def figura_reduccion(
    pca_varianza: pd.DataFrame,
    pca_barrido: pd.DataFrame,
    umap_barrido: pd.DataFrame,
    etiqueta_ap_umap: str,
    protocolo,
    destino: Path,
    dpi: int = 200,
) -> Path:
    fig, (ax_var, ax_pca, ax_umap) = plt.subplots(1, 3, figsize=(12.8, 3.5))

    var = pca_varianza.copy()
    if "modelo" in var.columns:
        curvas = []
        for modelo in _modelos_presentes(var):
            sub = var[var["modelo"] == modelo].sort_values("componente")
            alcanzado = sub.loc[sub["varianza_acumulada"] >= 0.95, "componente"]
            n_95_modelo = (int(alcanzado.iloc[0]) if not alcanzado.empty
                           else int(sub["componente"].max()))
            curvas.append((modelo, sub, n_95_modelo))
    else:
        sub = var.sort_values("componente")
        alcanzado = sub.loc[sub["varianza_acumulada"] >= 0.95, "componente"]
        n_95_sin_modelo = (int(alcanzado.iloc[0]) if not alcanzado.empty
                           else int(sub["componente"].max()))
        curvas = [(None, sub, n_95_sin_modelo)]
    n_95 = int(np.median([curva[2] for curva in curvas]))
    selecciones = _selecciones(protocolo)
    componentes_pca = [int(fila["componentes"]) for fila in selecciones
                       if str(fila.get("metodo", "")).lower() == "pca"
                       and fila.get("componentes") is not None]
    n_pca = int(np.median(componentes_pca)) if componentes_pca else n_95

    for modelo, sub, _ in curvas:
        ax_var.plot(sub["componente"], sub["varianza_acumulada"],
                    color=COLORES.get(modelo, "#2c7fb8"), marker="o", markersize=2.5,
                    linewidth=1.4,
                    label=_etiqueta_modelo(modelo) if modelo else "PCA")
    ax_var.axhline(0.95, color="#cb181d", linestyle="--", linewidth=1,
                   label="95 % de varianza")
    ax_var.axvline(n_pca, color="gray", linestyle=":", linewidth=1,
                   label=f"{n_pca} componentes")
    ax_var.set_xlabel("Número de componentes")
    ax_var.set_ylabel("Varianza acumulada")
    ax_var.set_title("(a) PCA: varianza acumulada")
    ax_var.yaxis.set_major_formatter(FuncFormatter(_formateador_porcentaje(0)))
    ax_var.legend(fontsize=7, loc="lower right")
    ax_var.grid(alpha=0.25)

    pca = pca_barrido.copy()
    col_x_pca = _columna(pca, ["componentes", "n_componentes", "dimensiones"],
                         "componentes de PCA")
    col_ap_pca = _columna(pca, ["AP", "AP_validacion", "AP_validación"], "AP de PCA")
    pca[col_x_pca] = pd.to_numeric(pca[col_x_pca], errors="coerce")
    pca[col_ap_pca] = pd.to_numeric(pca[col_ap_pca], errors="coerce")
    for modelo in _modelos_presentes(pca):
        sub = (pca[pca["modelo"] == modelo]
               .groupby(col_x_pca, as_index=False)[col_ap_pca].mean()
               .sort_values(col_x_pca))
        ax_pca.plot(sub[col_x_pca], sub[col_ap_pca], marker="o", markersize=3,
                    color=COLORES.get(modelo, "C0"), label=_etiqueta_modelo(modelo))
        for fila in selecciones:
            if str(fila.get("modelo")) == modelo and str(fila.get("metodo", "")).lower() == "pca":
                componente = fila.get("componentes")
                if componente is not None and not sub.empty:
                    cercano = sub.iloc[(sub[col_x_pca] - float(componente)).abs().argmin()]
                    ax_pca.plot(cercano[col_x_pca], cercano[col_ap_pca], marker="*",
                                markersize=10, color=COLORES.get(modelo, "C0"),
                                markeredgecolor="black", markeredgewidth=0.4, linestyle="none")
    ax_pca.set_xlabel("Número de componentes")
    ax_pca.set_ylabel("AP de validación")
    ax_pca.set_title("(b) PCA: AP de validación")
    ax_pca.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(2)))
    ax_pca.legend(fontsize=7, loc="best")
    ax_pca.grid(alpha=0.25)

    umap = umap_barrido.copy()
    col_x_umap = _columna(umap, ["dimensiones", "dimensiones_solicitadas", "componentes"],
                          "dimensiones de UMAP")
    col_ap_umap = _columna(umap, ["AP", "AP_validacion", "AP_validación", "AP_CV"],
                           "AP de UMAP")
    umap[col_x_umap] = pd.to_numeric(umap[col_x_umap], errors="coerce")
    umap[col_ap_umap] = pd.to_numeric(umap[col_ap_umap], errors="coerce")
    for modelo in _modelos_presentes(umap):
        sub = umap[umap["modelo"] == modelo]
        sub = (sub.groupby(col_x_umap, as_index=False)[col_ap_umap].max()
               .sort_values(col_x_umap))
        if sub.empty:
            continue
        ax_umap.plot(sub[col_x_umap], sub[col_ap_umap], marker="o", markersize=3,
                     color=COLORES.get(modelo, "C0"), label=_etiqueta_modelo(modelo))
        for fila in selecciones:
            if str(fila.get("modelo")) == modelo and "umap" in str(fila.get("metodo", "")).lower():
                componente = fila.get("componentes")
                if componente is not None:
                    cercano = sub.iloc[(sub[col_x_umap] - float(componente)).abs().argmin()]
                    ax_umap.plot(cercano[col_x_umap], cercano[col_ap_umap], marker="*",
                                 markersize=10, color=COLORES.get(modelo, "C0"),
                                 markeredgecolor="black", markeredgewidth=0.4, linestyle="none")
    ax_umap.set_xlabel("Dimensiones UMAP")
    ax_umap.set_ylabel(etiqueta_ap_umap)
    ax_umap.set_title("(c) UMAP: " + etiqueta_ap_umap.lower())
    ax_umap.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(2)))
    ax_umap.legend(fontsize=7, loc="best")
    ax_umap.grid(alpha=0.25)

    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def figura_umap(coordenadas: dict, destino: Path, dpi: int = 200) -> Path:
    if "test" not in coordenadas or "y_test" not in coordenadas:
        raise ValueError("umap_2d.json debe contener las claves test e y_test")
    xy = np.asarray(coordenadas["test"], dtype=float)
    y = np.asarray(coordenadas["y_test"], dtype=int)
    if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) != len(y):
        raise ValueError("Las coordenadas UMAP de prueba no coinciden con sus etiquetas")
    fig, eje = plt.subplots(figsize=(5.4, 4.2))
    for valor, color, etiqueta in ((0, "#9ecae1", "Sin compra"), (1, "#cb181d", "Compra")):
        mascara = y == valor
        eje.scatter(xy[mascara, 0], xy[mascara, 1], s=8, alpha=0.6,
                    color=color, label=etiqueta, linewidths=0)
    eje.set_xlabel("UMAP 1")
    eje.set_ylabel("UMAP 2")
    eje.set_title("UMAP 2D en prueba")
    eje.xaxis.set_major_formatter(FuncFormatter(_formateador_decimal(1)))
    eje.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(1)))
    eje.legend(fontsize=7.5, markerscale=2)
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


# ---------------------------------------------------------------------------
# Figuras del notebook
# ---------------------------------------------------------------------------
def figura_train_validacion(metricas: pd.DataFrame, destino: Path, dpi: int = 200) -> Path:
    met = metricas[metricas["particion"].isin(["entrenamiento", "validacion"])]
    pivote = met.pivot_table(index="modelo", columns="particion", values="AP")
    modelos = [m for m in MODELOS if m in pivote.index]
    x = np.arange(len(modelos))
    fig, eje = plt.subplots(figsize=(7.0, 3.3))
    eje.bar(x - 0.19, pivote.loc[modelos, "entrenamiento"], width=0.38,
            label="Entrenamiento", color="#bdbdbd")
    eje.bar(x + 0.19, pivote.loc[modelos, "validacion"], width=0.38,
            label="Validación", color="#2c7fb8")
    eje.set_xticks(x)
    eje.set_xticklabels([_etiqueta_modelo(m) for m in modelos], fontsize=7.5, rotation=15)
    eje.set_ylabel("Average Precision (AP)")
    eje.set_xlabel("Modelo")
    eje.set_title("AP en entrenamiento y validación")
    eje.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(2)))
    eje.legend(fontsize=7)
    eje.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def figura_matrices_confusion(matrices: dict, destino: Path, dpi: int = 200) -> Path:
    modelos = [m for m in MODELOS if m in matrices and "test" in matrices[m]]
    if not modelos:
        raise ValueError("matrices.json no contiene matrices de prueba")
    tiene_umbral = all("test_umbral_f1_validacion" in matrices[m] for m in modelos)
    filas = 2 if tiene_umbral else 1
    fig, ejes = plt.subplots(filas, len(modelos), figsize=(13.2, 2.7 * filas),
                             squeeze=False)
    for indice, modelo in enumerate(modelos):
        capas = [("test", "Umbral nativo")]
        if tiene_umbral:
            capas.append(("test_umbral_f1_validacion", "Umbral F1 (validación)"))
        for fila, (clave, titulo_umbral) in enumerate(capas):
            eje = ejes[fila, indice]
            matriz = np.asarray(matrices[modelo][clave])
            eje.imshow(matriz, cmap="Blues", vmin=0)
            limite = matriz.max() if matriz.max() else 1
            for i in range(matriz.shape[0]):
                for j in range(matriz.shape[1]):
                    eje.text(j, i, _texto_entero(matriz[i, j]), ha="center", va="center",
                             color="white" if matriz[i, j] > limite / 2 else "black",
                             fontsize=8)
            eje.set_xticks([0, 1])
            eje.set_yticks([0, 1])
            eje.set_xticklabels(["No", "Sí"], fontsize=7)
            eje.set_yticklabels(["No", "Sí"], fontsize=7)
            eje.set_xlabel("Predicción")
            eje.set_ylabel("Real" if indice == 0 else "")
            eje.set_title(f"{_etiqueta_modelo(modelo)}\n({titulo_umbral})" if fila == 0
                          else titulo_umbral, fontsize=8)
    fig.suptitle("Matrices de confusión en prueba", fontsize=9.5)
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def figura_ap_intervalos(intervalos: pd.DataFrame, destino: Path, dpi: int = 200) -> Path:
    ic = intervalos[(intervalos["metrica"] == "AP")
                    & (intervalos["particion"].isin(["validacion", "test"]))]
    if ic.empty:
        raise ValueError("intervalos.csv no contiene AP de validacion y prueba")
    modelos = [m for m in MODELOS if m in set(ic["modelo"])]
    x = np.arange(len(modelos))
    ancho = 0.36
    fig, eje = plt.subplots(figsize=(7.2, 3.4))
    for indice, (particion, color, etiqueta) in enumerate((
        ("validacion", "#2c7fb8", "Validación"),
        ("test", "#fdae6b", "Prueba"),
    )):
        sub = ic[ic["particion"] == particion].set_index("modelo").reindex(modelos)
        error = np.vstack([
            sub["media_bootstrap"] - sub["ic_inferior"],
            sub["ic_superior"] - sub["media_bootstrap"],
        ])
        eje.bar(x + (indice - 0.5) * ancho, sub["media_bootstrap"], width=ancho,
                yerr=error, capsize=3, label=etiqueta, color=color)
    eje.set_xticks(x)
    eje.set_xticklabels([_etiqueta_modelo(m) for m in modelos], fontsize=7.5, rotation=15)
    eje.set_ylabel("Average Precision (AP)")
    eje.set_xlabel("Modelo")
    eje.set_title("AP con intervalos bootstrap del 95 %")
    eje.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(2)))
    eje.legend(fontsize=7)
    eje.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def figura_pagevalues(comparacion: pd.DataFrame, destino: Path, dpi: int = 200) -> Path:
    for columna in ("modelo", "AP_con", "AP_sin"):
        if columna not in comparacion.columns:
            raise ValueError(f"comparacion_pagevalues.csv no contiene {columna}")
    tabla = comparacion.set_index("modelo")
    modelos = [m for m in MODELOS if m in tabla.index]
    x = np.arange(len(modelos))
    fig, eje = plt.subplots(figsize=(7.0, 3.3))
    eje.bar(x - 0.19, tabla.loc[modelos, "AP_con"], width=0.38,
            label="Con PageValues", color="#2c7fb8")
    eje.bar(x + 0.19, tabla.loc[modelos, "AP_sin"], width=0.38,
            label="Sin PageValues", color="#cb181d")
    eje.set_xticks(x)
    eje.set_xticklabels([_etiqueta_modelo(m) for m in modelos], fontsize=7.5, rotation=15)
    eje.set_ylabel("AP en validación")
    eje.set_xlabel("Modelo")
    eje.set_title("Sensibilidad al retirar PageValues")
    eje.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(2)))
    eje.legend(fontsize=7)
    eje.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def figura_specialday(ablacion: pd.DataFrame, destino: Path, dpi: int = 200) -> Path:
    if "delta_AP_con_menos_sin" in ablacion.columns:
        delta = ablacion[["modelo", "delta_AP_con_menos_sin"]].copy()
        delta = delta.rename(columns={"delta_AP_con_menos_sin": "delta_AP"})
    elif "delta_AP" in ablacion.columns:
        delta = ablacion[["modelo", "delta_AP"]].copy()
    elif {"AP_con_SpecialDay", "AP_sin_SpecialDay"}.issubset(ablacion.columns):
        delta = ablacion[["modelo"]].copy()
        delta["delta_AP"] = ablacion["AP_con_SpecialDay"] - ablacion["AP_sin_SpecialDay"]
    elif {"AP_con", "AP_sin"}.issubset(ablacion.columns):
        delta = ablacion[["modelo"]].copy()
        delta["delta_AP"] = ablacion["AP_con"] - ablacion["AP_sin"]
    else:
        raise ValueError(
            "specialday_ablacion.csv debe contener delta_AP_con_menos_sin, delta_AP "
            "o las columnas AP con y sin SpecialDay"
        )
    delta = delta.set_index("modelo")
    modelos = [m for m in MODELOS if m in delta.index]
    x = np.arange(len(modelos))
    fig, eje = plt.subplots(figsize=(6.8, 3.1))
    eje.bar(x, delta.loc[modelos, "delta_AP"], color="#756bb1")
    eje.axhline(0, color="black", linewidth=0.8)
    eje.set_xticks(x)
    eje.set_xticklabels([_etiqueta_modelo(m) for m in modelos], fontsize=7.5, rotation=15)
    eje.set_ylabel("Δ AP (con − sin)")
    eje.set_xlabel("Modelo")
    eje.set_title("Ablación: retirar SpecialDay")
    eje.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(3)))
    eje.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


def _etiqueta_representacion(metodo, componentes=None) -> str:
    texto = str(metodo).strip()
    minusculas = texto.lower()
    if "base" in minusculas:
        return "Base"
    if "pca" in minusculas:
        return "PCA"
    if "umap" in minusculas:
        if "2d" in minusculas or "umap2" in minusculas:
            return "UMAP 2D"
        if componentes is not None and not pd.isna(componentes):
            try:
                if int(float(componentes)) == 2:
                    return "UMAP 2D"
                return f"UMAP {int(float(componentes))}"
            except (TypeError, ValueError):
                pass
        return "UMAP"
    return texto


def _orden_representacion(etiqueta: str):
    if etiqueta == "Base":
        return (0, 0)
    if etiqueta == "PCA":
        return (1, 0)
    if etiqueta.startswith("UMAP 2D"):
        return (3, 0)
    if etiqueta.startswith("UMAP"):
        digitos = re.findall(r"\d+", etiqueta)
        return (2, int(digitos[0]) if digitos else 0)
    return (4, etiqueta)


def figura_comparacion_reduccion(
    metricas: pd.DataFrame,
    reduccion_test: pd.DataFrame,
    destino: Path,
    dpi: int = 200,
) -> Path:
    for columna in ("modelo", "AP"):
        if columna not in reduccion_test.columns:
            raise ValueError(f"reduccion_test.csv no contiene {columna}")
    columna_metodo = _columna(reduccion_test, ["metodo", "representacion", "variante"],
                              "metodo de reduccion")
    columna_componentes = None
    for candidata in ("componentes", "dimensiones", "dimensiones_solicitadas"):
        if candidata in reduccion_test.columns:
            columna_componentes = candidata
            break

    base = (metricas[metricas["particion"] == "test"]
            .set_index("modelo")["AP"].to_dict())
    valores: dict[str, dict[str, float]] = {}
    for fila in reduccion_test.itertuples(index=False):
        modelo = str(getattr(fila, "modelo"))
        if modelo not in base:
            continue
        componentes = getattr(fila, columna_componentes) if columna_componentes else None
        etiqueta = _etiqueta_representacion(getattr(fila, columna_metodo), componentes)
        valores.setdefault(etiqueta, {})
        if modelo not in valores[etiqueta]:
            valores[etiqueta][modelo] = float(getattr(fila, "AP"))

    for modelo in base:
        if modelo in reduccion_test["modelo"].astype(str).values and modelo in (
            "RandomForest", "MLP"
        ):
            valores.setdefault("Base", {})
            valores["Base"].setdefault(modelo, float(base[modelo]))

    if not valores:
        raise ValueError("reduccion_test.csv no contiene resultados de Random Forest o MLP")
    etiquetas = sorted(valores, key=_orden_representacion)
    modelos = [m for m in ("RandomForest", "MLP") if any(m in v for v in valores.values())]
    x = np.arange(len(etiquetas))
    ancho = 0.38
    fig, eje = plt.subplots(figsize=(7.6, 3.4))
    for indice, modelo in enumerate(modelos):
        alturas = [valores[etiqueta].get(modelo, np.nan) for etiqueta in etiquetas]
        eje.bar(x + (indice - 0.5) * ancho, alturas, width=ancho,
                color=COLORES.get(modelo, "C0"), label=_etiqueta_modelo(modelo))
    eje.set_xticks(x)
    eje.set_xticklabels(etiquetas, fontsize=7.5, rotation=15)
    eje.set_ylabel("AP en prueba")
    eje.set_xlabel("Representación")
    eje.set_title("Efecto de la reducción sobre el AP")
    eje.yaxis.set_major_formatter(FuncFormatter(_formateador_decimal(2)))
    eje.legend(fontsize=7)
    eje.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _guardar(fig, destino, dpi)


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------
def generar_figuras(resultados=DIR_RES, figuras_informe=None, rapido: bool = False) -> dict:
    """Genera las seis figuras del informe y las figuras del notebook.

    Parameters
    ----------
    resultados : Path
        Carpeta del flujo actual (por defecto, resultados/). Debe contener
        auditoria.json, cobertura_mensual_train.csv, cv_todos_los_candidatos.csv,
        comparacion_pagevalues.csv y las subcarpetas evaluacion_familias,
        variables, reduccion y specialday.
    figuras_informe : Path | None
        Destino de las figuras del informe. Si es None y rapido=False, usa
        informe_entregable_II_latex/figuras. Si es None y rapido=True, usa
        <resultados>/figuras_informe para no sobrescribir el informe.
    rapido : bool
        Reduce la resolucion y evita escribir en el directorio del informe.
    """
    resultados = Path(resultados)
    if not resultados.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de resultados {resultados}")
    if figuras_informe is None:
        figuras_informe = (resultados / "figuras_informe") if rapido else DIR_INFORME
    figuras_informe = Path(figuras_informe)
    figuras_notebook = resultados / "figuras"
    dpi = 120 if rapido else 200

    # Se cargan todas las entradas antes de escribir la primera figura para
    # fallar por datos incompletos sin dejar un lote parcial.
    auditoria = _leer_json(resultados, "auditoria.json")
    mensual = _leer_csv(resultados, "cobertura_mensual_train.csv",
                        requeridas=("Month", "sesiones", "tasa_compra"))
    cv = _leer_csv(resultados, "cv_todos_los_candidatos.csv",
                   requeridas=("modelo", "mean_test_score"))
    metricas = _leer_csv(resultados, "evaluacion_familias", "metricas.csv",
                         requeridas=("modelo", "particion", "AP"))
    intervalos = _leer_csv(resultados, "evaluacion_familias", "intervalos.csv",
                           requeridas=("metrica", "media_bootstrap", "ic_inferior",
                                       "ic_superior", "modelo", "particion"))
    curvas = _leer_json(resultados, "evaluacion_familias", "curvas.json")
    matrices = _leer_json(resultados, "evaluacion_familias", "matrices.json")
    variables = _leer_csv(resultados, "variables", "analisis_variables.csv",
                          requeridas=("variable",))
    pca_varianza = _leer_csv(resultados, "reduccion", "pca_varianza.csv",
                             requeridas=("componente", "varianza_acumulada"))
    pca_barrido = _leer_csv(resultados, "reduccion", "pca_barrido.csv",
                            requeridas=("modelo",))
    umap_barrido, etiqueta_ap_umap = _cargar_barrido_umap(resultados)
    umap_2d = _leer_json(resultados, "reduccion", "umap_2d.json")
    reduccion_test = _leer_csv(resultados, "reduccion", "reduccion_test.csv",
                               requeridas=("modelo", "AP"))
    protocolo = _leer_opcional(resultados, "reduccion", "reduccion_protocolo.json")
    ruta_specialday = _localizar(resultados, (
        ("specialday", "specialday_ablacion.csv"),
        ("specialday_ablacion.csv",),
    ))
    specialday = pd.read_csv(ruta_specialday)
    comparacion_pv = _leer_csv(resultados, "comparacion_pagevalues.csv",
                               requeridas=("modelo", "AP_con", "AP_sin"))

    prevalencia = float(auditoria.get("proporcion_compra", np.nan))
    rutas = {"informe": [], "notebook": []}
    rutas["informe"].append(figura_clases_cobertura(
        auditoria, mensual, figuras_informe / "fig01_clases_cobertura.png", dpi))
    rutas["informe"].append(figura_hiperparametros(
        cv, figuras_informe / "fig02_hiperparametros.png", dpi))
    rutas["informe"].append(figura_pr_roc(
        curvas, prevalencia, figuras_informe / "fig03_pr_roc_validacion.png", dpi))
    rutas["informe"].append(figura_variables(
        variables, figuras_informe / "fig04_variables_auc.png", dpi))
    rutas["informe"].append(figura_reduccion(
        pca_varianza, pca_barrido, umap_barrido, etiqueta_ap_umap, protocolo,
        figuras_informe / "fig05_reduccion.png", dpi))
    rutas["informe"].append(figura_umap(
        umap_2d, figuras_informe / "fig06_umap_2d.png", dpi))

    rutas["notebook"].append(figura_train_validacion(
        metricas, figuras_notebook / "fig04_train_vs_validacion.png", dpi))
    rutas["notebook"].append(figura_matrices_confusion(
        matrices, figuras_notebook / "fig07_matrices_confusion_test.png", dpi))
    rutas["notebook"].append(figura_ap_intervalos(
        intervalos, figuras_notebook / "fig08_ap_intervalos.png", dpi))
    rutas["notebook"].append(figura_pagevalues(
        comparacion_pv, figuras_notebook / "fig09_sensibilidad_pagevalues.png", dpi))
    rutas["notebook"].append(figura_specialday(
        specialday, figuras_notebook / "fig10_ablacion_specialday.png", dpi))
    rutas["notebook"].append(figura_comparacion_reduccion(
        metricas, reduccion_test, figuras_notebook / "fig13_comparacion_reduccion.png", dpi))
    return rutas


def main(argv=None) -> None:
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--resultados", type=Path, default=DIR_RES,
                            help="Carpeta de resultados del flujo actual.")
    analizador.add_argument("--figuras-informe", type=Path, default=None,
                            help="Destino de las figuras del informe.")
    analizador.add_argument("--rapido", action="store_true",
                            help="Baja resolucion y no toca el directorio del informe.")
    argumentos = analizador.parse_args(argv)
    rutas = generar_figuras(argumentos.resultados, argumentos.figuras_informe,
                            argumentos.rapido)
    print(f"Figuras del informe: {len(rutas['informe'])}")
    print(f"Figuras del notebook: {len(rutas['notebook'])}")


if __name__ == "__main__":
    main()
