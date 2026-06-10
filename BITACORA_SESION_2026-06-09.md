# Bitácora de Sesión — Selección de Características (Feature Selection)

**Fecha:** 2026-06-09  
**Dataset:** `dataA_binario.csv` (Target binario — 4,424 filas × 38 columnas)  
**Ubicación:** `/Users/julianolazco/Downloads/GitHub/dataA/`

---

## 1. Preparación del Entorno

- Se identificó que el entorno virtual `./venv` no contaba con `scikit-learn`.
- Se instaló correctamente usando `venv/bin/python -m pip install scikit-learn pandas numpy`.
- Se verificó que todas las importaciones del script funcionaran correctamente.

---

## 2. Corrección del Script `script.py`

El script estaba hardcodeado para eliminar las columnas `["target", "dropout.semester"]`, pero `dataA_binario.csv` utiliza `Target` (mayúscula) y no contiene `dropout.semester`.

### Modificación realizada
Se reemplazó el bloque de carga de datos por una detección robusta de la columna target:

```python
# Detectar columna target y columnas a eliminar de forma robusta
target_col = next((c for c in data.columns if c.lower() == "target"), None)
if target_col is None:
    raise ValueError("No se encontró una columna 'target' o 'Target' en los datos.")
drop_cols = [c for c in data.columns if c.lower() in ("target", "dropout.semester", "target_original")]
X = data.drop(columns=drop_cols)
y = data[target_col]
```

**Impacto:** El script ahora es compatible con `dataA.csv`, `dataA_binario.csv` y datasets futuros que usen cualquier variante de mayúsculas/minúsculas para la columna target.

---

## 3. Experimentos de Selección de Características

Se ejecutó el script `script.py` con el dataset `dataA_binario.csv` bajo diferentes números de repeticiones (runs) para evaluar la estabilidad estadística de cada método.

### Métodos evaluados
| # | Método | Tipo | Descripción |
|---|--------|------|-------------|
| 1 | `baseline` | Referencia | Todas las 36 características |
| 2 | `variance_threshold` | Filtro | Umbral de varianza (>0.01) |
| 3 | `chi2` | Filtro | Chi-cuadrado, top 10 features |
| 4 | `mutual_info` | Filtro | Información mutua, top 10 features |
| 5 | `rfe` | Wrapper | Recursive Feature Elimination (10 features) |
| 6 | `rfecv` | Wrapper | RFE con validación cruzada (5-fold) |
| 7 | `l1` | Embedded | L1 Regularización (Logistic Regression) |
| 8 | `rf_topk` | Embedded | Random Forest — top 10 importancias |
| 9 | `rf_median` | Embedded | Random Forest — umbral mediana |

### Métricas reportadas
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Número de features seleccionadas
- Tiempo de entrenamiento (segundos)

---

## 4. Resultados por Número de Runs

### 15 runs
| Top 3 | Accuracy | Features |
|-------|----------|----------|
| rfecv | 87.86% | ~30 |
| l1 | 87.74% | ~27 |
| baseline | 87.78% | 36 |

### 30 runs
| Top 3 | Accuracy | Features |
|-------|----------|----------|
| l1 | 87.74% | ~27 |
| rfecv | 87.71% | ~30 |
| baseline | 87.69% | 36 |

### 60 runs
| Top 3 | Accuracy | Features |
|-------|----------|----------|
| baseline | 87.55% | 36 |
| variance_threshold | 87.55% | 36 |
| rfecv | 87.51% | ~30 |

### 90 runs (Resultados finales estabilizados)
| Posición | Método | Accuracy | Features | Observación |
|----------|--------|----------|----------|-------------|
| 🥇 | baseline | **87.55%** | 36 | Referencia completa |
| 🥇 | variance_threshold | **87.55%** | 36 | Equivalente al baseline |
| 🥈 | l1 | **87.50%** | ~27 | Mejor balance reducción/rendimiento |
| 🥉 | rfecv | **87.49%** | ~30 | Mayor precision (87.91%) |
| 4 | rf_median | 87.06% | 18 | Reducción del 50% |
| 5 | rfe | 86.71% | 10 | Mejor opción con 10 features |
| 6 | rf_topk | 85.89% | 10 | Rápido pero inestable |
| 7 | mutual_info | 85.81% | 10 | Información mutua |
| 8 | chi2 | 84.82% | 10 | Más rápido, menor accuracy |

---

## 5. Archivos Generados en esta Sesión

### Resultados crudos
| Archivo | Descripción |
|---------|-------------|
| `experiment_results.csv` | Resultados de 15 runs |
| `experiment_results_60runs.csv` | Resultados de 60 runs |
| `experiment_results_90runs.csv` | Resultados de 90 runs |
| `experiment_results_summary.csv` | Resumen de 15 runs |
| `experiment_results_60runs_summary.csv` | Resumen de 60 runs |
| `experiment_results_90runs_summary.csv` | Resumen de 90 runs |

### Tablas comparativas visuales
| Archivo | Formato | Descripción |
|---------|---------|-------------|
| `resultados_comparativos.md` | Markdown | 15 runs |
| `resultados_comparativos.html` | HTML+CSS | 15 runs (visual) |
| `resultados_comparativos_30runs.md` | Markdown | 30 runs |
| `resultados_comparativos_30runs.html` | HTML+CSS | 30 runs (visual) |
| `resultados_comparativos_60runs.md` | Markdown | 60 runs |
| `resultados_comparativos_60runs.html` | HTML+CSS | 60 runs (visual) |
| `resultados_comparativos_90runs.md` | Markdown | 90 runs |
| `resultados_comparativos_90runs.html` | HTML+CSS | 90 runs (visual) |

### Script modificado
| Archivo | Cambio |
|---------|--------|
| `script.py` | Detección robusta de columna target (`Target`/`target`) y eliminación automática de columnas auxiliares (`Target_Original`, `dropout.semester`) |

---

## 6. Hallazgos Clave

1. **Estabilidad estadística:** Los resultados se estabilizaron a partir de los 60 runs. Con 90 runs, las diferencias entre el top 3 son <0.06%.
2. **El baseline es difícil de superar:** Con 36 features, el baseline logra el mejor accuracy promedio (87.55%).
3. **L1 es la mejor alternativa de reducción:** Mantiene 87.50% de accuracy con solo ~27 features (25% de reducción).
4. **RFECV ofrece la mayor precision:** 87.91% de precision con ~30 features seleccionadas automáticamente.
5. **RFE con 10 features es viable:** 86.71% de accuracy con máxima reducción dimensional.
6. **Chi2 es el más rápido:** 0.0037s por experimento, pero con el menor accuracy (84.82%).
7. **Variance threshold no elimina features:** En este dataset, todas las 36 features superan el umbral de varianza de 0.01.

---

## 7. Comparativa de Evolución por Runs

| Método | 15 runs | 30 runs | 60 runs | 90 runs |
|--------|---------|---------|---------|---------|
| baseline | 87.78% | 87.69% | 87.55% | **87.55%** |
| l1 | 87.74% | 87.74% | 87.50% | **87.50%** |
| rfecv | 87.86% | 87.71% | 87.51% | **87.49%** |
| rf_median | 87.27% | 87.19% | 87.01% | **87.06%** |
| rfe | 86.95% | 86.90% | 86.70% | **86.71%** |

---

## 8. Próximos Pasos Sugeridos

1. **Optimización de hiperparámetros:** Ajustar `C` en L1, `n_estimators` en Random Forest, o `step` en RFE/RFECV.
2. **Modelos adicionales:** Evaluar los subsets seleccionados con XGBoost, SVM o Redes Neuronales.
3. **Validación estadística:** Realizar pruebas t-pareadas o Wilcoxon para confirmar diferencias significativas entre métodos.
4. **Visualización de convergencia:** Graficar la evolución de accuracy vs. número de runs para cada método.
5. **Análisis de features seleccionadas:** Identificar qué features son consistentemente elegidas por L1 y RFECV a través de los 90 runs.

---

*Bitácora generada al finalizar la sesión de experimentación de selección de características.*
