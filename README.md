# Predicción de la intención de compra de usuarios en línea mediante aprendizaje supervisado

**Proyecto de aula — Modelos y Simulación de Sistemas II**
Departamento de Ingeniería de Sistemas, Universidad de Antioquia, Colombia

**Integrantes:** Mateo Upegui Borja, Juan P. Uribe González y Juan D. Rincón Bedoya.

## Descripción

Clasificación binaria supervisada de la intención de compra en sesiones de navegación en
línea, sobre el *Online Shoppers Purchasing Intention Dataset* (UCI #468): 12 330 sesiones,
17 atributos predictores y la etiqueta `Revenue` (15,47 % de compras).

## Notebooks

Los tres notebooks incluyen las tablas, figuras y conclusiones de la ejecución guardada. Se pueden leer en orden sin entrenar modelos.

| Notebook | Contenido |
|---|---|
| [01 · Análisis exploratorio](notebooks/01_analisis_exploratorio.ipynb) | Audita las sesiones, elimina duplicados exactos, crea las particiones estratificadas y analiza los 17 predictores. |
| [02 · Entrenamiento y ablación](notebooks/02_entrenamiento_y_ablacion.ipynb) | Compara cinco familias con validación cruzada, evalúa los tres finalistas de cada familia en validación y mide el efecto de retirar variables. |
| [03 · Evaluación final, sensibilidad y reducción](notebooks/03_evaluacion_sensibilidad_y_reduccion.ipynb) | Calcula métricas, intervalos y comparaciones pareadas. Evalúa los umbrales, la sensibilidad a `PageValues` y `SpecialDay`, y la reducción con PCA y UMAP. |

## Reproducir los resultados

Desde la raíz del repositorio, instale el entorno con Python 3.13.5 y las dependencias fijadas en `uv.lock`. La ejecución utiliza el paquete local `shopytics`.

```bash
uv sync --locked
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/0[123]_*.ipynb
```

El comando ejecuta 01, 02 y 03 en ese orden y actualiza sus salidas en el repositorio. Para ejecutar las celdas de forma interactiva, abra los notebooks con `uv run jupyter lab`.

| Carpeta | Archivos que producen o utilizan los notebooks |
|---|---|
| `data/` | CSV original, datos deduplicados y particiones en pickle. El 02 guarda además las matrices codificadas y ajusta el preprocesador dentro de cada pliegue para validación cruzada. |
| `resultados/variables/` | El 01 guarda el análisis individual y los pares de variables redundantes en JSON. El 02 los usa para elegir las ablaciones. |
| `resultados/entrenamiento/` | El 02 guarda los candidatos de validación cruzada por familia, los finalistas, las configuraciones seleccionadas, el ganador y las ablaciones en JSON. |
| `resultados/evaluacion/` | El 03 reutiliza los 15 finalistas sin `PageValues` de `finalistas_sin_pagevalues.json` y los 46 candidatos PCA/UMAP de `barrido_reduccion.pkl`. Ajusta los modelos seleccionados y calcula las tablas y figuras en el propio notebook. |

### Repetir las búsquedas

La ejecución predeterminada reutiliza las búsquedas guardadas. Para regenerarlas desde el CSV original, ejecute el 01 y active las siguientes opciones antes de ejecutar los notebooks correspondientes.

| Notebook | Opción | Búsqueda que repite |
|---|---|---|
| 02 | `REENTRENAR = True` | Mallas de hiperparámetros de las cinco familias sobre los predictores completos. |
| 03 | `REPETIR_SIN_PAGEVALUES = True` | Mallas de las cinco familias después de retirar `PageValues`. |
| 03 | `REPETIR_REDUCCION = True` | Barrido de PCA y UMAP para las dos familias mejor clasificadas en validación. |

## Resultados principales

Random Forest obtuvo la mayor precisión promedio (AP) en validación, con 0,7414. En prueba alcanzó 0,7496 frente a 0,7494 de MLP. El intervalo bootstrap pareado del 95 % para la diferencia de AP fue de −0,0190 a 0,0190. Al repetir la selección sin `PageValues`, el AP de validación de Random Forest descendió a 0,3848. La regla de un error estándar seleccionó 10 componentes de PCA para Random Forest y 15 para MLP, con AP en prueba de 0,7236 y 0,7408, respectivamente.

Las búsquedas y la selección utilizaron entrenamiento y validación. Las métricas de prueba ya se conocían de una entrega anterior y se interpretan como descriptivas. Los intervalos bootstrap condicionan la incertidumbre a los modelos ajustados y las predicciones de esta partición.

## Datos y licencia

*Online Shoppers Purchasing Intention Dataset*, C. Sakar y Y. Kastro, UCI Machine Learning
Repository, 2018. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licencia CC BY 4.0.
