# Predicción de la intención de compra de usuarios en línea mediante aprendizaje supervisado

**Proyecto de aula — Modelos y Simulación de Sistemas II**
Departamento de Ingeniería de Sistemas, Universidad de Antioquia, Colombia

**Integrantes:** Mateo Upegui Borja, Juan P. Uribe González y Juan D. Rincón Bedoya.

## Descripción

Clasificación binaria supervisada de la intención de compra sobre el *Online Shoppers
Purchasing Intention Dataset* (UCI #468, Sakar y Kastro, 2018): 12 330 sesiones de
navegación, 17 atributos predictores y la etiqueta `Revenue` (15,47 % de compras).

El repositorio contiene el informe completo, que reúne todo lo realizado en el Entregable I
más lo realizado en el Entregable II. No es el Entregable II por sí solo.

El Entregable II corresponde a las siguientes secciones:

- **Sección 1 (30 %):** diseño experimental y resultados del entrenamiento y la evaluación
  de cinco familias de modelos (regresión logística, k-NN, Random Forest, MLP y SVM), con
  intervalos de confianza, comparaciones pareadas, análisis de umbral y sensibilidad a
  `PageValues` y `SpecialDay`.
- **Sección 2 (20 %):** análisis individual de las 17 variables, redundancia, PCA y UMAP,
  y comparación del AP con y sin reducción de dimensión.

El flujo es único: `src/ejecutar_todo.py` ejecuta las etapas en orden, escribe los
artefactos en `resultados/` y alimenta al notebook y al informe LaTeX. El informe se
compila con `pdflatex`.

## Estructura del repositorio

```
.
├── data/
│   └── online_shoppers_intention.csv        # datos de entrada
├── src/
│   ├── ejecutar_todo.py                     # orquestador del flujo único
│   ├── proyecto_ml.py                       # carga, auditoría, particiones y pipelines
│   ├── busqueda.py                          # validación cruzada y top-3 por familia
│   ├── seleccion.py                         # configuración final elegida en validación
│   ├── evaluacion.py                        # métricas, intervalos, umbrales y comparaciones
│   ├── sensibilidad.py                      # repetición del experimento sin PageValues
│   ├── specialday.py                        # ablación y codificación de SpecialDay
│   ├── analisis_variables.py                # análisis individual, redundancia y eliminación
│   ├── experimentos_reduccion.py            # PCA, UMAP y comparación con la línea base
│   └── figuras.py                           # figuras del informe y del notebook
├── notebooks/
│   └── proyecto_intencion_compra.ipynb      # notebook único de resultados
├── tests/
│   └── test_*.py                            # pruebas unitarias y de integridad
├── resultados/                              # salida vigente del flujo único
│   ├── evaluacion_familias/
│   ├── variables/
│   ├── reduccion/
│   └── figuras/
├── informe_entregable_II_latex/
│   ├── main.tex                             # fuente del informe IEEE
│   ├── IEEEtran.cls
│   └── figuras/
├── requirements.txt
└── README.md
```

El flujo vigente lee y escribe únicamente en `resultados/`; si se ejecuta con `--rapido`,
usa `resultados_rapido/`. Los notebooks y artefactos de exploraciones anteriores no forman
parte de este flujo.

## Instalación

Las versiones exactas de las dependencias están en `requirements.txt`. La reproducibilidad
documentada se verificó con Python 3.13.5.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux o macOS:

```bash
source .venv/bin/activate
```

Instalación de dependencias:

```bash
python -m pip install -r requirements.txt
```

Para compilar el informe se necesita una distribución de TeX con `pdflatex` (TeX Live,
MiKTeX o MacTeX). No se instala con `pip`.

## Flujo único

La ejecución completa se lanza desde la raíz del proyecto:

```bash
python src/ejecutar_todo.py
```

Las etapas se ejecutan en este orden y cada una consume las salidas de las anteriores:

| Etapa | Contenido | Salidas principales |
|---|---|---|
| `datos` | Auditoría, tabla de variables, particiones y cobertura mensual | `auditoria.json`, `tabla_variables.csv`, `resumen_particiones.csv` |
| `busqueda` | Validación cruzada de las cinco familias y top-3 por AP-CV | `cv_todos_los_candidatos.csv`, `validacion_top3.csv`, `mejores_configuraciones.csv`, `ganador.json` |
| `evaluacion` | Métricas por partición, intervalos CV y bootstrap, umbrales y comparaciones pareadas | `evaluacion_familias/metricas.csv`, `intervalos.csv`, `cv_intervalos.csv`, `umbrales.csv`, `comparaciones_test.csv` |
| `sensibilidad` | Repetición de la búsqueda sin `PageValues` | `mejores_sin_pagevalues.csv`, `comparacion_pagevalues.csv` |
| `specialday` | Ablación, comparación numérica/one-hot y AP mensual | `specialday_ablacion.csv`, `specialday_codificacion.csv`, `specialday_por_mes.csv` |
| `variables` | AUC univariado, información mutua, redundancia y eliminación | `variables/analisis_variables.csv`, `redundancia_numericas.csv`, `eliminacion_validacion.csv` |
| `reduccion` | PCA y UMAP para las dos mejores familias, con la regla 1-SE en validación | `reduccion/reduccion_test.csv`, `reduccion_validacion.csv`, `reduccion_cv.csv`, `reduccion_protocolo.json` |
| `figuras` | Figuras del informe y del notebook | `informe_entregable_II_latex/figuras/`, `resultados/figuras/` |

Al finalizar se escribe `resultados/manifiesto.json` con el SHA-256 de los datos, la semilla,
las versiones de las librerías, los tamaños de las particiones y la duración de cada etapa.

### Reanudar desde una etapa

`--desde` ejecuta desde la etapa indicada hasta `figuras` y reutiliza los artefactos de las
etapas anteriores. `--hasta` acota el recorrido y acepta las mismas etapas, de modo que
`--desde datos --hasta datos` ejecuta solo la auditoría y `--desde figuras` regenera solo
las figuras. Los valores admitidos son `datos`, `busqueda`, `evaluacion`, `sensibilidad`,
`specialday`, `variables`, `reduccion` y `figuras`.

```bash
python src/ejecutar_todo.py --desde evaluacion
python src/ejecutar_todo.py --desde reduccion
python src/ejecutar_todo.py --desde figuras
python src/ejecutar_todo.py --desde datos --hasta datos
```

Antes de usar `--desde`, compruebe que los archivos de las etapas previas existan en la
carpeta de salida correspondiente. La etapa `reduccion` depende de `mejores_configuraciones.csv`
y de los resultados de `evaluacion`; `figuras` depende de todas las anteriores.

### Prueba de humo

```bash
python src/ejecutar_todo.py --rapido
```

`--rapido` reduce folds, tamaño de muestra y réplicas bootstrap, y escribe en
`resultados_rapido/`. Sirve para comprobar que el flujo se ejecuta, no para producir cifras
del informe: sus métricas **no son comparables** con la corrida completa y no deben citarse
como resultados.

### Advertencia de costo

La corrida completa es costosa. La etapa `sensibilidad` repite la búsqueda por validación
cruzada de las cinco familias sin `PageValues`, y `reduccion` recorre la rejilla de PCA y
UMAP (quince configuraciones de UMAP por familia en el protocolo completo, con validación
cruzada y evaluación en validación). UMAP y la búsqueda de sensibilidad son las etapas que
más tiempo consumen; no hay un atajo equivalente para obtener resultados válidos. Para
ahorrar tiempo, ejecute por etapas con `--desde` y conserve `resultados/` entre ejecuciones.

## Pruebas

La suite usa `pytest` y no entrena la rejilla completa:

```bash
python -m pytest -q
```

Las pruebas de integridad comprueban dimensiones, particiones, ausencia de fuga, mallas,
preprocesador y contratos de las etapas. Las que dependen de `resultados/manifiesto.json`
se omiten si no existe una ejecución completa.

## Notebook

El notebook único es `notebooks/proyecto_intencion_compra.ipynb`. Por defecto
`RECALCULAR = False`: lee las tablas y figuras de `resultados/` sin reentrenar. Si se
establece `RECALCULAR = True`, llama a `ejecutar_todo.main()` y ejecuta el flujo completo.

Para ejecutarlo y guardar una copia ejecutada:

```bash
jupyter nbconvert --to notebook --execute notebooks/proyecto_intencion_compra.ipynb --output-dir notebooks --output proyecto_intencion_compra_ejecutado.ipynb
```

El notebook debe abrirse desde `notebooks/` o desde la raíz del proyecto; rechaza otros
directorios de trabajo. No transcribe cifras: todas las tablas se leen de los CSV y JSON de
`resultados/`. Si falta un artefacto, la sección correspondiente lo advierte y continúa.

## Informe LaTeX

Genere primero las figuras con la etapa `figuras`. Después, desde
`informe_entregable_II_latex/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

La segunda compilación actualiza referencias y tablas. En Windows con MiKTeX puede usarse la
ruta completa de `pdflatex.exe`.

El notebook y el informe consumen los mismos artefactos de `resultados/`. Si se regenera una
etapa, vuelva a ejecutar `figuras` y recompile el informe para mantener las cifras y las
figuras alineadas.

## Resultados principales

Cifras redondeadas a cuatro decimales, leídas de los CSV actuales del flujo. Si se
reejecuta el flujo, los CSV son la fuente de verdad y pueden cambiar estos valores.

| Modelo | AP validación | AP prueba | ROC-AUC prueba |
|---|---:|---:|---:|
| Random Forest | 0,7414 | 0,7496 | 0,9302 |
| MLP | 0,7225 | 0,7494 | 0,9284 |
| SVM | 0,7054 | 0,7272 | 0,9034 |
| k-NN | 0,7018 | 0,6976 | 0,9075 |
| Regresión logística | 0,6841 | 0,6956 | 0,9110 |

Fuente: `resultados/evaluacion_familias/metricas.csv`.

- La diferencia pareada Random Forest − MLP en AP de prueba es +0,0002, con IC 95 %
  [−0,0190; 0,0190]; en ROC-AUC es +0,0018, con IC [−0,0036; 0,0075]. Ambos intervalos
  incluyen cero: no se detecta una diferencia significativa, lo que no demuestra
  equivalencia. Fuente: `evaluacion_familias/comparaciones_test.csv`.
- Con el umbral de Random Forest elegido por F1 en validación (0,3195), el F1 de prueba es
  0,6904, con recall 0,7356 y precisión 0,6505. Fuente: `umbrales.csv` y
  `metricas_test_umbral.csv`.
- Retirar `PageValues` reduce el AP de validación entre 0,3345 y 0,3848 según la familia,
  una caída de 0,35 a 0,37 frente al 0,7414 del modelo completo. Fuente:
  `comparacion_pagevalues.csv`.
- La ablación de `SpecialDay` es marginal: el AP de validación pasa de 0,7414 a 0,7418
  (Δ = −0,0004). Fuente: `specialday_ablacion.csv`.
- Las candidatas a eliminación salen del propio análisis: las seis variables con AUC
  univariado menor que 0,55 (`Weekend`, `Region`, `Browser`, `VisitorType`, `SpecialDay` y
  `OperatingSystems`). Retirarlas deja el AP de validación en 0,7422 para Random Forest y
  0,7414 para la MLP, sin deterioro frente a 0,7414 y 0,7225. Fuente:
  `variables/eliminacion_validacion.csv`.
- Reducción de dimensión, con la rejilla completa del protocolo y la regla 1-SE: la base
  queda en AP de prueba 0,7496 (Random Forest) y 0,7494 (MLP). La regla eligió PCA con 10
  componentes para Random Forest (AP 0,7236) y 15 para la MLP (0,7408), y UMAP con 15
  dimensiones y 50 vecinos (0,6879) y con 5 dimensiones y 30 vecinos (0,6381),
  respectivamente; la referencia UMAP de dos dimensiones dio 0,6595 y 0,5815. En Random
  Forest, PCA cuesta −0,0260 de AP con IC [−0,0438; −0,0090] y UMAP cuesta −0,0616 con IC
  [−0,0893; −0,0353]. En la MLP, la caída con PCA es −0,0086 con IC [−0,0219; 0,0050], que
  incluye cero, mientras que UMAP cuesta −0,1113. Reducir dimensiones no mejora el
  resultado; UMAP sirve para visualizar, no para modelar. Fuentes:
  `reduccion/reduccion_test.csv`, `reduccion/reduccion_comparacion.csv` y
  `reduccion/reduccion_protocolo.json`.

### Limitaciones de la evaluación

La partición de prueba no participó en ninguna decisión de esta versión, pero sus métricas
se conocían de una primera versión del trabajo: se leen como descriptivas y no como una
evaluación independiente. Las decisiones de búsqueda y selección se tomaron sin usar la
prueba. Los intervalos bootstrap son
condicionales a los modelos y las predicciones fijos, y no incorporan la variabilidad del
entrenamiento ni de la selección del umbral. La ausencia de diferencia significativa entre
Random Forest y MLP no demuestra equivalencia. Las cifras de `--rapido` tampoco son comparables con
las de la corrida completa.

## Video de sustentación

## Datos y licencia

*Online Shoppers Purchasing Intention Dataset*, C. Sakar y Y. Kastro, UCI Machine Learning
Repository, 2018. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licencia CC BY 4.0.
