"""Comprueba el notebook como archivo, sin ejecutarlo.

Lee el ``.ipynb`` como JSON y verifica que los archivos que cita existan, que los nombres
de archivo no lleven tildes, que no queden palabras dañadas y que las salidas guardadas
correspondan a una ejecución completa y sin errores.
"""

import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks" / "proyecto_intencion_compra.ipynb"
CARPETAS = (RAIZ / "resultados", RAIZ / "informe_entregable_II_latex" / "figuras")

ARCHIVO = re.compile(r"[\wáéíóúñÁÉÍÓÚÑ]+\.(?:csv|json|png)\b")
# Patrones con comodín que el notebook usa para buscar figuras, no nombres de archivo.
COMODINES = ("fig_nb_",)
FORMAS_DANADAS = ("ciónes", "siónes", "Sistemás", "Purchasíng", "clasífic")


@pytest.fixture(scope="module")
def celdas():
    if not NOTEBOOK.is_file():
        pytest.skip("No existe el notebook")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _fuente(celda):
    return "".join(celda["source"])


def _existentes():
    return {ruta.name for carpeta in CARPETAS if carpeta.is_dir()
            for ruta in carpeta.rglob("*") if ruta.is_file()}


def test_los_archivos_citados_existen(celdas):
    existentes = _existentes()
    faltantes = set()
    for celda in celdas:
        for nombre in ARCHIVO.findall(_fuente(celda)):
            if nombre.startswith(COMODINES):
                continue
            if nombre not in existentes:
                faltantes.add(nombre)
    assert not faltantes, f"El notebook cita archivos que no existen: {sorted(faltantes)}"


def test_los_nombres_de_archivo_no_llevan_tildes(celdas):
    con_tilde = set()
    for celda in celdas:
        for nombre in ARCHIVO.findall(_fuente(celda)):
            if re.search(r"[áéíóúñÁÉÍÓÚÑ]", nombre):
                con_tilde.add(nombre)
    assert not con_tilde, f"Nombres de archivo con tilde: {sorted(con_tilde)}"


def test_no_quedan_palabras_danadas(celdas):
    texto = "\n".join(_fuente(celda) for celda in celdas)
    presentes = [forma for forma in FORMAS_DANADAS if forma in texto]
    assert not presentes, f"Palabras dañadas en el notebook: {presentes}"


def test_las_salidas_no_tienen_errores_ni_avisos(celdas):
    for celda in celdas:
        if celda["cell_type"] != "code":
            continue
        for salida in celda.get("outputs", []):
            assert salida["output_type"] != "error", "Hay una celda con error guardado"
            texto = "".join(salida.get("text", []))
            assert "AVISO: falta" not in texto, "Una celda avisa de un archivo faltante"


def test_los_contadores_de_ejecucion_son_consecutivos(celdas):
    contadores = [celda.get("execution_count") for celda in celdas
                  if celda["cell_type"] == "code"]
    assert contadores == list(range(1, len(contadores) + 1))
