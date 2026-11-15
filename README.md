# Predicción de la intención de compra de usuarios en línea mediante aprendizaje supervisado

**Proyecto de aula — Modelos y Simulación de Sistemas II**
Departamento de Ingeniería de Sistemas, Universidad de Antioquia, Colombia

**Integrantes:** Mateo Upegui Borja, Juan P. Uribe González y Juan D. Rincón Bedoya.

## Descripción

Clasificación binaria supervisada de la intención de compra sobre el *Online Shoppers
Purchasing Intention Dataset* (UCI #468, Sakar y Kastro, 2018): 12 330 sesiones de
navegación, 17 atributos predictores y la etiqueta `Revenue` (15,47 % de compras).

El repositorio contiene el Entregable II completo:

- **Sección 1 (30 %):** diseño experimental y resultados del entrenamiento y la evaluación
  de cinco familias de modelos (regresión logística, k-NN, Random Forest, MLP y SVM), con
  intervalos de confianza bootstrap, comparaciones pareadas, análisis de umbral y
  sensibilidad a `PageValues` y `SpecialDay`.
- **Sección 2 (20 %):** análisis individual de las 17 variables, redundancia, PCA y UMAP,
  y comparación del AP con y sin reducción de dimensión.

El informe está en LaTeX (formato IEEE) y se compila con `pdflatex`.

## Estructura del repositorio

```
.
├── data/
│   └── online_shoppers_intention.csv        # datos de entrada
├── informe_entregable_II_latex/
│   ├── main.tex                             # fuente del informe IEEE
│   ├── main.pdf                             # PDF previo; puede no coincidir con la fuente
│   ├── IEEEtran.cls
│   └── figuras/                             # fig01–fig13
├── notebooks/
│   ├── 01_analisis_exploratorio.ipynb       # EDA y análisis individual
│   ├── 02_entrenamiento_evaluacion.ipynb    # sección 1 (30 %)
│   └── 03_reduccion_dimension.ipynb         # PCA/UMAP (20 %)
├── src/
│   ├── proyecto_ml.py                       # pipelines, métricas, IC, UMAP
│   ├── auditar.py                           # auditoría y particiones
│   ├── entrenar.py                          # búsqueda y evaluación
│   ├── configuracion_final.py                # configuraciones históricas finales
│   ├── evaluar_final.py                     # reproducción de configuraciones finales
│   ├── figuras.py                           # figuras del flujo original
│   ├── figuras_corregidas.py                # figuras del informe corregido
│   ├── analisis_variables.py                # análisis individual y redundancia
│   ├── reduccion.py                         # PCA, UMAP y comparación
│   ├── figuras_reduccion.py                 # figuras de reducción
│   └── informe.py                           # generador DOCX legado
├── resultados/                              # archivo histórico de la exploración
├── resultados_corregidos/                   # resultados usados en el informe
├── requirements.txt
└── requirements-informe-windows.txt         # flujo DOCX opcional (Windows)
```

## Cómo reproducir

### Entorno

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
pip install -r requirements.txt
```

Las ejecuciones históricas del informe usaron Python 3.13.5, scikit-learn 1.7.2 y
semilla `SEMILLA = 42`. Los requisitos no fijan todas las versiones; otras versiones
pueden producir diferencias.

El flujo DOCX legado requiere además, solo en Windows:

```bash
pip install -r requirements-informe-windows.txt
```

### Notebooks

Ejecute en orden desde `notebooks/`:

1. `01_analisis_exploratorio.ipynb`: auditoría, distribución de clases, cobertura
   mensual, correlaciones y análisis individual de variables.
2. `02_entrenamiento_evaluacion.ipynb`: sección 1. Con `RECALCULAR = False` carga los
   resultados guardados; con `True` reentrena los cinco modelos (≈12 minutos).
3. `03_reduccion_dimension.ipynb`: PCA, UMAP y comparación con la línea base.

### Línea de comandos

```bash
python src/auditar.py              # auditoría y particiones
python src/evaluar_final.py        # configura y evalúa los modelos finales
python src/figuras_corregidas.py   # figuras del informe corregido
python src/analisis_variables.py   # análisis individual y redundancia
python src/reduccion.py            # PCA, UMAP y comparación
python src/figuras_reduccion.py    # figuras de reducción
```

`resultados/` y `resultados_corregidos/mallas_ext/` conservan los artefactos históricos
versionados. No ejecute `entrenar.py` para consultar el informe: regenera la búsqueda y
las ablaciones en `resultados/` y puede sobrescribir ese archivo. Solo es necesario si
se desea repetir la búsqueda, preferiblemente en una copia aislada.

`evaluar_final.py` reajusta las configuraciones finales y escribe
`resultados_corregidos/`, reutilizando las ablaciones archivadas. Ejecútelo antes de
generar las figuras; `entrenar.py` es opcional y solo sirve para repetir la búsqueda.
Los scripts de figuras escriben en `informe_entregable_II_latex/figuras/`.

### Informe LaTeX

Desde `informe_entregable_II_latex/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

En Windows con MiKTeX:

```powershell
& "C:\Users\<usuario>\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" `
  -interaction=nonstopmode -halt-on-error main.tex
```

Para Overleaf, suba un ZIP con `main.tex`, `IEEEtran.cls` y `figuras/`, no el PDF
suelto. El enlace de edición no se publica en este repositorio.

## Resultados principales

| Métrica | Valor |
|---|---|
| AP de prueba, Random Forest | 0,7486 |
| AP de prueba, MLP | 0,7377 |
| ROC-AUC de prueba, MLP | 0,9270 |
| ROC-AUC de prueba, Random Forest | 0,9242 |
| Media bootstrap pareada AP RF−MLP | +0,0104, IC 95 % [−0,0100; 0,0321] |
| Umbral de RF por F1 | 0,3078; F1 0,6691; recall 0,7173; precisión 0,6270 |
| Caída de AP sin `PageValues` | 0,34–0,39 |
| PCA | 28 componentes; 61,64 % de reducción; AP 0,7352 (RF) y 0,7211 (MLP) |
| UMAP 10 componentes | 86,30 % de reducción; AP 0,6650 (RF) y 0,6236 (MLP) |
| UMAP 2 componentes | AP 0,6489 (RF) y 0,5762 (MLP) |

No se detectó una diferencia estadísticamente significativa entre Random Forest y la MLP
en AP ni ROC-AUC; esto no demuestra equivalencia, solo ausencia de evidencia de diferencia
con la potencia disponible. `PageValues` presenta el mayor AUC univariado (0,866). PCA
muestra una caída puntual de AP de hasta 1,7 puntos con 61,64 % menos columnas; las
caídas puntuales de UMAP son de 8 a 16 puntos. Los IC pareados de ΔAP y ΔROC-AUC
incluyen cero para PCA y quedan por debajo de cero para UMAP en esta partición.
Los IC bootstrap disponibles son condicionales a modelos y predicciones fijos y no
incluyen la variabilidad del entrenamiento ni de la selección del umbral. La prueba
histórica ya fue consultada y no constituye una evaluación independiente.

## Datos y licencia

*Online Shoppers Purchasing Intention Dataset*, C. Sakar y Y. Kastro, UCI Machine Learning
Repository, 2018. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licencia CC BY 4.0.
