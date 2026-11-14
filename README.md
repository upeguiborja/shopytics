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

El informe final está en LaTeX (formato IEEE, 10 páginas) y se compila con `pdflatex`.

## Estructura del repositorio

```
.
├── data/
│   └── online_shoppers_intention.csv        # datos de entrada
├── informe_entregable_II_latex/
│   ├── main.tex                             # informe IEEE (10 páginas)
│   ├── main.pdf                             # PDF compilado
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
│   ├── figuras.py                           # figuras del flujo original
│   ├── figuras_corregidas.py                # figuras del informe corregido
│   ├── analisis_variables.py                # análisis individual y redundancia
│   ├── reduccion.py                         # PCA, UMAP y comparación
│   ├── figuras_reduccion.py                 # figuras de reducción
│   └── informe.py                           # generador DOCX legado
├── resultados/                              # resultados del flujo original
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
python src/entrenar.py             # entrenamiento y evaluación (≈12 minutos)
python src/figuras_corregidas.py   # figuras del informe corregido
python src/analisis_variables.py   # análisis individual y redundancia
python src/reduccion.py            # PCA, UMAP y comparación
python src/figuras_reduccion.py    # figuras de reducción
```

`entrenar.py` escribe en `resultados/`; las cifras del informe se conservan en
`resultados_corregidos/`. Los scripts de figuras leen de `resultados_corregidos/` y
escriben en `informe_entregable_II_latex/figuras/`.

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

El PDF también se puede subir a Overleaf junto con `IEEEtran.cls` y `figuras/`. El enlace
de edición no se publica en este repositorio.

## Resultados principales

| Métrica | Valor |
|---|---|
| AP de prueba, Random Forest | 0,7486 |
| AP de prueba, MLP | 0,7377 |
| ROC-AUC de prueba, MLP | 0,9270 |
| ROC-AUC de prueba, Random Forest | 0,9242 |
| Diferencia pareada AP RF−MLP | +0,0104, IC 95 % [−0,0100; 0,0321] |
| Umbral de RF por F1 | 0,3078; F1 0,6691; recall 0,7173; precisión 0,6270 |
| Caída de AP sin `PageValues` | 0,34–0,39 |
| PCA | 28 componentes; 61,64 % de reducción; AP 0,7352 (RF) y 0,7211 (MLP) |
| UMAP 10 componentes | 86,30 % de reducción; AP 0,6650 (RF) y 0,6236 (MLP) |
| UMAP 2 componentes | AP 0,6489 (RF) y 0,5762 (MLP) |

Random Forest y la MLP son estadísticamente equivalentes en AP y ROC-AUC. `PageValues`
concentra la mayor señal (AUC univariado 0,866); PCA conserva el desempeño con 61,64 %
menos columnas, mientras UMAP lo degrada entre 8 y 16 puntos.

## Datos y licencia

*Online Shoppers Purchasing Intention Dataset*, C. Sakar y Y. Kastro, UCI Machine Learning
Repository, 2018. DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licencia CC BY 4.0.
