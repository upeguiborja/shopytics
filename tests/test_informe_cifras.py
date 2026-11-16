"""Comprueba que las cifras clave del informe y del README salgan de los CSV.

Cada valor se formatea con cuatro decimales y coma decimal, como se escribe en el
informe LaTeX (``0{,}7414``) y en el README (``0,7414``). Si un CSV cambia y los
documentos no se actualizan, la prueba falla.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
INFORME = RAIZ / "informe_entregable_II_latex" / "main.tex"
README = RAIZ / "README.md"

COMPARACIONES = {"AP", "ROC_AUC"}


def _leer(ruta):
    if not ruta.is_file():
        pytest.skip(f"No existe {ruta.relative_to(RAIZ)}")
    return ruta.read_text(encoding="utf-8")


def _variantes(valor, decimales=4):
    texto = f"{abs(valor):.{decimales}f}"
    return {texto, texto.replace(".", "{,}"), texto.replace(".", ",")}


def _aparece(texto, valor, decimales=4):
    return any(variante in texto for variante in _variantes(valor, decimales))


def _metricas():
    ruta = RES / "evaluacion_familias" / "metricas.csv"
    if not ruta.is_file():
        pytest.skip("No existe metricas.csv")
    tabla = pd.read_csv(ruta)
    return tabla.set_index(["modelo", "particion"])["AP"]


def _comparacion(modelo_a, modelo_b, metrica):
    ruta = RES / "evaluacion_familias" / "comparaciones_test.csv"
    if not ruta.is_file():
        pytest.skip("No existe comparaciones_test.csv")
    tabla = pd.read_csv(ruta)
    fila = tabla[(tabla["modelo_a"] == modelo_a) & (tabla["modelo_b"] == modelo_b)
                 & (tabla["metrica"] == metrica)]
    assert len(fila) == 1
    return fila.iloc[0]


def _reduccion():
    ruta = RES / "reduccion" / "reduccion_test.csv"
    if not ruta.is_file():
        pytest.skip("No existe reduccion_test.csv")
    return pd.read_csv(ruta)


def _pagevalues():
    ruta = RES / "comparacion_pagevalues.csv"
    if not ruta.is_file():
        pytest.skip("No existe comparacion_pagevalues.csv")
    return pd.read_csv(ruta)


def test_ap_de_random_forest_y_mlp_en_informe_y_readme():
    metricas = _metricas()
    informe, readme = _leer(INFORME), _leer(README)
    for modelo in ("RandomForest", "MLP"):
        for particion in ("validacion", "test"):
            valor = float(metricas.loc[(modelo, particion)])
            for nombre, texto in (("informe", informe), ("README", readme)):
                assert _aparece(texto, valor), (
                    f"AP {particion} de {modelo} = {valor:.4f} no aparece en el {nombre}"
                )


def test_diferencia_random_forest_menos_mlp_con_su_intervalo():
    informe, readme = _leer(INFORME), _leer(README)
    for metrica in COMPARACIONES:
        fila = _comparacion("RandomForest", "MLP", metrica)
        for valor in (fila["diferencia_puntual"], fila["ic_inferior"], fila["ic_superior"]):
            for nombre, texto in (("informe", informe), ("README", readme)):
                assert _aparece(texto, float(valor)), (
                    f"La comparación RF-MLP en {metrica} ({valor:.4f}) no aparece en el {nombre}"
                )


def test_variantes_elegidas_de_pca_y_umap_en_el_informe():
    tabla = _reduccion()
    informe = _leer(INFORME)
    for modelo in ("RandomForest", "MLP"):
        pca = tabla[(tabla["modelo"] == modelo) & (tabla["metodo"] == "PCA")].iloc[0]
        umap = tabla[(tabla["modelo"] == modelo) & (tabla["metodo"] == "UMAP")].iloc[0]
        assert f"{int(pca['componentes'])} componentes" in informe, (
            f"PCA de {modelo} con {int(pca['componentes'])} componentes no se menciona"
        )
        assert f"{int(umap['componentes'])} dimensiones" in informe, (
            f"UMAP de {modelo} con {int(umap['componentes'])} dimensiones no se menciona"
        )
        assert _aparece(informe, float(pca["AP"])), f"AP de PCA de {modelo} no aparece"
        assert _aparece(informe, float(umap["AP"])), f"AP de UMAP de {modelo} no aparece"


def test_rango_sin_pagevalues_en_informe_y_readme():
    tabla = _pagevalues()
    minimo, maximo = float(tabla["AP_sin"].min()), float(tabla["AP_sin"].max())
    for nombre, texto in (("informe", _leer(INFORME)), ("README", _leer(README))):
        assert _aparece(texto, minimo), f"El mínimo sin PageValues no aparece en el {nombre}"
        assert _aparece(texto, maximo), f"El máximo sin PageValues no aparece en el {nombre}"


def test_el_informe_no_conserva_marcadores_de_trabajo():
    informe = _leer(INFORME)
    for marcador in ("PENDIENTE", "TODO", "FIXME", "XXX"):
        assert marcador not in informe
    protocolo = RES / "reduccion" / "reduccion_protocolo.json"
    if protocolo.is_file():
        datos = json.loads(protocolo.read_text(encoding="utf-8"))
        assert "regla_simplicidad" in datos
        assert "error_estandar" in datos["regla_simplicidad"]


def test_cada_etiqueta_de_figura_y_tabla_tiene_referencia():
    informe = _leer(INFORME)
    etiquetas = re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", informe)
    assert etiquetas, "El informe no declara etiquetas de figura o tabla"
    sin_cita = [etiqueta for etiqueta in etiquetas
                if f"\\ref{{{etiqueta}}}" not in informe]
    assert not sin_cita, f"Etiquetas sin referencia en el texto: {sin_cita}"


def test_porcentajes_de_reduccion_coinciden_con_el_csv():
    tabla = _reduccion()
    informe = _leer(INFORME)
    for fila in tabla.itertuples():
        if pd.isna(fila.componentes):
            continue
        porcentaje = (1 - float(fila.componentes) / 73) * 100
        texto = f"{porcentaje:.1f}".replace(".", ",")
        assert texto + "\\,\\%" in informe, (
            f"La reducción de {fila.modelo} {fila.metodo} ({texto} %) no aparece en el informe"
        )


def test_cifras_de_eliminacion_en_informe_y_readme():
    ruta = RES / "variables" / "eliminacion_validacion.csv"
    if not ruta.is_file():
        pytest.skip("No existe eliminacion_validacion.csv")
    tabla = pd.read_csv(ruta)
    for modelo in ("RandomForest", "MLP"):
        for variante in ("completo", "sin_candidatas"):
            valor = float(tabla[(tabla["modelo"] == modelo)
                                & (tabla["variante"] == variante)]["AP"].iloc[0])
            for nombre, texto in (("informe", _leer(INFORME)), ("README", _leer(README))):
                assert _aparece(texto, valor), (
                    f"AP de {modelo} en {variante} ({valor:.4f}) no aparece en el {nombre}"
                )


def test_valores_de_c_de_svm_coinciden_con_la_malla():
    import sys

    sys.path.insert(0, str(RAIZ / "src"))
    import busqueda
    import proyecto_ml

    _, mallas = busqueda.mallas_ampliadas(proyecto_ml.construir_modelos)
    informe = _leer(INFORME)
    for malla in mallas["SVM"]:
        valores = ";".join(f"{v:g}" for v in malla["clf__C"]).replace(".", "{,}")
        assert valores in informe, f"Los valores de C de SVM {valores} no aparecen en el informe"


def test_el_informe_no_supera_diez_paginas():
    log = RAIZ / "informe_entregable_II_latex" / "main.log"
    if not log.is_file():
        pytest.skip("No existe main.log; compile el informe")
    texto = log.read_text(encoding="utf-8", errors="ignore")
    coincidencia = re.search(r"Output written on main\.pdf \((\d+) pages?", texto)
    if not coincidencia:
        pytest.skip("El log no registra el número de páginas")
    assert int(coincidencia.group(1)) <= 10


def test_el_pdf_final_no_es_mas_antiguo_que_la_fuente():
    pdf = RAIZ / "informe_entregable_II.pdf"
    if not pdf.is_file():
        pytest.skip("No existe informe_entregable_II.pdf")
    assert pdf.stat().st_mtime >= INFORME.stat().st_mtime - 1, (
        "El PDF de la raíz es más antiguo que main.tex: recompile y cópielo"
    )
