"""Pruebas focalizadas del generador de figuras con datos sinteticos.

Las pruebas no leen resultados/ reales ni escriben en el directorio del
informe: construyen un conjunto minimo de salidas en un directorio temporal.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import figuras  # noqa: E402


MODELOS = ["RandomForest", "MLP", "SVM", "kNN", "Regresion_logistica"]


def _fila_cv(modelo, splits, splits_train, **params):
    fila = {"modelo": modelo, "mean_test_score": float(np.mean(splits)),
            "std_test_score": float(np.std(splits, ddof=1))}
    for indice in range(5):
        fila[f"split{indice}_test_score"] = float(splits[indice])
        fila[f"split{indice}_train_score"] = float(splits_train[indice])
    for nombre in ("param_clf__max_depth", "param_clf__min_samples_leaf",
                   "param_clf__alpha", "param_clf__learning_rate_init",
                   "param_clf__C", "param_clf__n_neighbors", "param_clf__kernel"):
        fila[nombre] = np.nan
    fila.update(params)
    return fila


def _resultados_sinteticos(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = [0.60, 0.62, 0.64, 0.66, 0.68]
    filas = [
        _fila_cv("Regresion_logistica", base, [v + 0.2 for v in base],
                 **{"param_clf__C": 0.1}),
        _fila_cv("Regresion_logistica", [0.65] * 5, [0.85] * 5,
                 **{"param_clf__C": 1.0}),
        _fila_cv("kNN", base, [v + 0.1 for v in base],
                 **{"param_clf__n_neighbors": 5}),
        _fila_cv("kNN", [0.65] * 5, [0.75] * 5,
                 **{"param_clf__n_neighbors": 15}),
        _fila_cv("RandomForest", base, [0.95] * 5,
                 **{"param_clf__max_depth": 4, "param_clf__min_samples_leaf": 1}),
        _fila_cv("RandomForest", [0.65] * 5, [0.99] * 5,
                 **{"param_clf__max_depth": np.nan, "param_clf__min_samples_leaf": 5}),
        _fila_cv("MLP", base, [0.9] * 5,
                 **{"param_clf__alpha": 1e-5, "param_clf__learning_rate_init": 0.001}),
        _fila_cv("MLP", [0.66] * 5, [0.9] * 5,
                 **{"param_clf__alpha": 0.1, "param_clf__learning_rate_init": 0.01}),
        _fila_cv("SVM", base, [0.8] * 5,
                 **{"param_clf__C": 0.1, "param_clf__kernel": "rbf"}),
        _fila_cv("SVM", [0.66] * 5, [0.8] * 5,
                 **{"param_clf__C": 10.0, "param_clf__kernel": "linear"}),
    ]
    pd.DataFrame(filas).to_csv(tmp_path / "cv_todos_los_candidatos.csv", index=False)

    (tmp_path / "auditoria.json").write_text(json.dumps({
        "clases": {"False": 10422, "True": 1908},
        "proporcion_compra": 0.15474,
    }), encoding="utf-8")
    pd.DataFrame([
        {"Month": "June", "sesiones": 164, "tasa_compra": 0.097561},
        {"Month": "Mar", "sesiones": 1159, "tasa_compra": 0.106126},
        {"Month": "Nov", "sesiones": 1752, "tasa_compra": 0.251712},
    ]).to_csv(tmp_path / "cobertura_mensual_train.csv", index=False)

    evaluacion = tmp_path / "evaluacion_familias"
    evaluacion.mkdir()
    metricas = []
    intervalos = []
    curvas = {}
    matrices = {}
    for indice, modelo in enumerate(MODELOS):
        for particion, ap in (("entrenamiento", 0.90 - 0.02 * indice),
                              ("validacion", 0.70 - 0.01 * indice),
                              ("test", 0.71 - 0.01 * indice)):
            metricas.append({"modelo": modelo, "particion": particion, "AP": ap})
        for particion, ap in (("validacion", 0.70 - 0.01 * indice),
                              ("test", 0.71 - 0.01 * indice)):
            intervalos.append({"metrica": "AP", "media_bootstrap": ap,
                               "ic_inferior": ap - 0.02, "ic_superior": ap + 0.02,
                               "modelo": modelo, "particion": particion})
        curvas[modelo] = {"validacion": {
            "pr_precision": [0.9, 0.6, 0.4], "pr_recall": [0.1, 0.5, 1.0],
            "roc_fpr": [0.0, 0.3, 1.0], "roc_tpr": [0.0, 0.7, 1.0]}}
        matrices[modelo] = {"test": [[500, 100], [50, 300]],
                            "test_umbral_f1_validacion": [[450, 150], [40, 310]]}
    pd.DataFrame(metricas).to_csv(evaluacion / "metricas.csv", index=False)
    pd.DataFrame(intervalos).to_csv(evaluacion / "intervalos.csv", index=False)
    (evaluacion / "curvas.json").write_text(json.dumps(curvas), encoding="utf-8")
    (evaluacion / "matrices.json").write_text(json.dumps(matrices), encoding="utf-8")

    variables = tmp_path / "variables"
    variables.mkdir()
    filas_variables = []
    for posicion, (nombre, auc, recomendacion) in enumerate([
        ("PageValues", 0.866, "mantener"),
        ("ExitRates", 0.710, "mantener"),
        ("ProductRelated", 0.670, "mantener"),
        ("BounceRates", 0.610, "revisar"),
        ("Region", 0.560, "revisar"),
        ("Weekend", 0.520, "eliminar o combinar"),
        ("VisitorType", 0.505, "eliminar o combinar"),
        ("SpecialDay", 0.480, "eliminar o combinar"),
        ("Month", 0.470, "eliminar o combinar"),
        ("Browser", 0.460, "eliminar o combinar"),
        ("OperatingSystems", 0.450, "eliminar o combinar"),
        ("TrafficType", 0.440, "eliminar o combinar"),
    ]):
        filas_variables.append({
            "variable": nombre, "tipo": "numérica", "metrica": "correlación",
            "valor": 0.4, "p_valor": 0.0, "auc_univariado": auc,
            "direccion": "positiva", "informacion_mutua": 0.16 - 0.01 * posicion,
            "ranking": posicion + 1, "recomendacion": recomendacion})
    pd.DataFrame(filas_variables).to_csv(variables / "analisis_variables.csv", index=False)

    reduccion = tmp_path / "reduccion"
    reduccion.mkdir()
    explained = [0.35, 0.25, 0.15, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.01]
    filas_varianza = [
        {"modelo": modelo, "componente": componente,
         "varianza_explicada": explicada,
         "varianza_acumulada": float(np.cumsum(explained)[indice])}
        for modelo in ("RandomForest", "MLP")
        for indice, (componente, explicada) in enumerate(zip(range(1, 11), explained))
    ]
    pd.DataFrame(filas_varianza).to_csv(reduccion / "pca_varianza.csv", index=False)
    pd.DataFrame([
        {"modelo": "RandomForest", "componentes": n, "dimensiones_solicitadas": n,
         "AP_validacion": 0.70 + 0.005 * n, "AP_CV": 0.69 + 0.005 * n,
         "std_CV": 0.02, "varianza_acumulada": 0.9, "seleccionado": False}
        for n in (5, 10, 20, 28, 35, 45)
    ] + [
        {"modelo": "MLP", "componentes": n, "dimensiones_solicitadas": n,
         "AP_validacion": 0.69 + 0.005 * n, "AP_CV": 0.68 + 0.005 * n,
         "std_CV": 0.02, "varianza_acumulada": 0.9, "seleccionado": False}
        for n in (5, 10, 20, 28, 35, 45)
    ]).to_csv(reduccion / "pca_barrido.csv", index=False)
    pd.DataFrame([
        {"modelo": "RandomForest", "dimensiones": d, "vecinos": 15,
         "AP_validacion": 0.68 + 0.005 * d, "AP_CV": 0.67 + 0.005 * d,
         "std_CV": 0.03, "seleccionado": False}
        for d in (2, 5, 10, 15, 20)
    ] + [
        {"modelo": "MLP", "dimensiones": d, "vecinos": 15,
         "AP_validacion": 0.65 + 0.004 * d, "AP_CV": 0.64 + 0.004 * d,
         "std_CV": 0.03, "seleccionado": False}
        for d in (2, 5, 10, 15, 20)
    ]).to_csv(reduccion / "umap_barrido.csv", index=False)
    (reduccion / "reduccion_protocolo.json").write_text(json.dumps({
        "seleccion": [
            {"modelo": "RandomForest", "metodo": "PCA", "componentes": 28, "AP_validacion": 0.74},
            {"modelo": "RandomForest", "metodo": "UMAP", "componentes": 10, "AP_validacion": 0.68},
            {"modelo": "MLP", "metodo": "PCA", "componentes": 28, "AP_validacion": 0.72},
            {"modelo": "MLP", "metodo": "UMAP", "componentes": 10, "AP_validacion": 0.67},
        ]}), encoding="utf-8")
    pd.DataFrame([
        {"modelo": "RandomForest", "metodo": "PCA", "componentes": 28, "AP": 0.740},
        {"modelo": "RandomForest", "metodo": "UMAP", "componentes": 10, "AP": 0.683},
        {"modelo": "MLP", "metodo": "PCA", "componentes": 28, "AP": 0.733},
        {"modelo": "MLP", "metodo": "UMAP", "componentes": 10, "AP": 0.622},
    ]).to_csv(reduccion / "reduccion_test.csv", index=False)
    (reduccion / "umap_2d.json").write_text(json.dumps({
        "test": [[0.1, 0.2], [0.3, -0.1], [-0.4, 0.5], [0.7, 0.9]],
        "y_test": [0, 1, 0, 1],
    }), encoding="utf-8")

    specialday = tmp_path / "specialday"
    specialday.mkdir()
    pd.DataFrame([
        {"modelo": modelo, "AP_con_SpecialDay": 0.70, "AP_sin_SpecialDay": 0.6996,
         "delta_AP_con_menos_sin": 0.0004, "params": "{}"}
        for modelo in MODELOS
    ]).to_csv(specialday / "specialday_ablacion.csv", index=False)

    pd.DataFrame([
        {"modelo": modelo, "AP_con": 0.70, "AP_sin": 0.38, "delta_AP": 0.32}
        for modelo in MODELOS
    ]).to_csv(tmp_path / "comparacion_pagevalues.csv", index=False)
    return tmp_path


def test_resumen_parametro_usa_desviacion_entre_folds(tmp_path):
    base = [0.60, 0.62, 0.64, 0.66, 0.68]
    filas = [
        _fila_cv("RandomForest", base, [0.9] * 5,
                 **{"param_clf__max_depth": 4}),
        _fila_cv("RandomForest", base, [0.9] * 5,
                 **{"param_clf__max_depth": 4}),
    ]
    tabla = figuras.resumen_parametro(
        pd.DataFrame(filas), "RandomForest", "param_clf__max_depth")
    assert len(tabla) == 1
    assert tabla.loc[0, "ap"] == pytest.approx(np.mean(base))
    assert tabla.loc[0, "desviacion"] == pytest.approx(np.std(base, ddof=1))


def test_resumen_parametro_ordena_valores_y_trata_sin_limite(tmp_path):
    filas = [
        _fila_cv("RandomForest", [0.6] * 5, [0.9] * 5,
                 **{"param_clf__max_depth": np.nan}),
        _fila_cv("RandomForest", [0.6] * 5, [0.9] * 5,
                 **{"param_clf__max_depth": 4}),
        _fila_cv("RandomForest", [0.6] * 5, [0.9] * 5,
                 **{"param_clf__max_depth": 12}),
    ]
    tabla = figuras.resumen_parametro(
        pd.DataFrame(filas), "RandomForest", "param_clf__max_depth")
    assert list(tabla["valor"]) == [4, 12, "sin límite"]
    assert figuras._etiqueta_valor("sin límite") == "sin límite"


def test_meses_en_espanol_y_orden_cronologico():
    assert figuras._mes_espanol("June") == "Jun"
    assert figuras._mes_espanol("Aug") == "Ago"
    assert figuras._orden_mes("Feb") < figuras._orden_mes("Nov")


def test_representaciones_de_reduccion_ordenadas():
    etiquetas = ["UMAP 10 d", "Base", "PCA", "UMAP 2 D (ref.)"]
    assert sorted(etiquetas, key=figuras._orden_representacion) == [
        "Base", "PCA", "UMAP 10 d", "UMAP 2 D (ref.)"
    ]
    assert figuras._etiqueta_representacion("PCA", 28) == "PCA"
    assert figuras._etiqueta_representacion("UMAP", 2) == "UMAP 2 d"
    assert figuras._etiqueta_representacion("UMAP", 10) == "UMAP 10 d"
    assert figuras._etiqueta_representacion("UMAP2D", 2) == "UMAP 2 D (ref.)"


def test_estrella_de_reduccion_usa_la_variante_elegida_no_el_maximo():
    tabla = pd.DataFrame([
        {"modelo": "RandomForest", "componentes": 10, "AP_validacion": 0.707,
         "seleccionado": True},
        {"modelo": "RandomForest", "componentes": 20, "AP_validacion": 0.718,
         "seleccionado": False},
        {"modelo": "MLP", "componentes": 15, "AP_validacion": 0.722,
         "seleccionado": "True"},
    ])
    assert figuras._fila_elegida(tabla, "RandomForest", "componentes", "AP_validacion") == (10.0, 0.707)
    assert figuras._fila_elegida(tabla, "MLP", "componentes", "AP_validacion") == (15.0, 0.722)
    vacia = tabla.assign(seleccionado=False)
    assert figuras._fila_elegida(vacia, "MLP", "componentes", "AP_validacion") is None


def test_es_verdadero_acepta_bool_texto_y_entero():
    assert figuras._es_verdadero(True)
    assert figuras._es_verdadero("true")
    assert figuras._es_verdadero(1)
    assert not figuras._es_verdadero(False)
    assert not figuras._es_verdadero("false")
    assert not figuras._es_verdadero(np.nan)


def test_generar_figuras_escribe_seis_de_informe_y_seis_de_notebook(tmp_path):
    resultados = _resultados_sinteticos(tmp_path / "resultados")
    informe = tmp_path / "informe"
    rutas = figuras.generar_figuras(resultados, figuras_informe=informe, rapido=True)
    assert len(rutas["informe"]) == 6
    assert len(rutas["notebook"]) == 6
    for ruta in rutas["informe"] + rutas["notebook"]:
        assert ruta.is_file()
        assert ruta.stat().st_size > 1000
    assert (informe / "fig01_clases_cobertura.png").is_file()
    assert (resultados / "figuras" / "fig13_comparacion_reduccion.png").is_file()


def test_faltan_resultados_produce_error_explicito(tmp_path):
    with pytest.raises(FileNotFoundError, match="auditoria.json"):
        figuras.generar_figuras(tmp_path, figuras_informe=tmp_path / "informe")
