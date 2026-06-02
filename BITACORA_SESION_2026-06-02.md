# Bitácora de Sesión — Análisis Exploratorio de Datos (EDA)

**Fecha:** 2026-06-02  
**Dataset:** `dataA.csv` (Conjunto de datos de estudiantes — Predicción de deserción/retención)  
**Ubicación:** `/Users/julianolazco/Downloads/GitHub/data/`

---

## 1. Exploración Inicial del Dataset

- **Dimensiones:** 4,424 filas × 37 columnas
- **Valores faltantes:** Ninguno
- **Tipos de datos:** 30 enteros, 7 flotantes
- **Variable objetivo original:** `Target` con 3 clases:
  - `0` — Graduado (2,209 registros, 49.93%)
  - `1` — Desertó (1,421 registros, 32.12%)
  - `2` — Matriculado (794 registros, 17.95%)

### Variables identificadas
- Demográficas: edad, género, nacionalidad, estado civil
- Académicas: calificaciones, unidades aprobadas/inscritas/evaluadas (1er y 2do semestre)
- Económicas: tasa de desempleo, inflación, PIB
- Familiares: ocupación y calificación de padres y madres
- Administrativas: modo de aplicación, beca, deudor, colegiatura al día

---

## 2. Análisis Exploratorio de Datos (EDA) — Target Original (3 clases)

### Gráficas generadas (`./graficas/`)
| # | Gráfica | Descripción |
|---|---------|-------------|
| 01 | `01_distribucion_target.png` | Distribución de las 3 clases del target |
| 02 | `02_histogramas_variables.png` | Histogramas de 8 variables numéricas clave |
| 03 | `03_boxplots_por_target.png` | Boxplots de edad, nota de admisión y calificaciones por target |
| 04 | `04_matriz_correlacion_completa.png` | Mapa de calor triangular de la matriz de correlación completa |
| 05 | `05_correlacion_con_target.png` | Barras horizontales de correlación de cada variable con Target |
| 06 | `06_top_correlaciones.png` | Top 15 correlaciones más fuertes entre pares de variables |
| 07 | `07_pairplot_correlaciones.png` | Pairplot de las 6 variables más correlacionadas con Target |
| 08 | `08_edad_por_target.png` | Distribución de edad de inscripción por cada clase |
| 09 | `09_calificaciones_por_semestre.png` | Histogramas de calificaciones del 1er y 2do semestre por target |
| 10 | `10_indicadores_economicos.png` | Distribución de desempleo, inflación y PIB |

### Hallazgos principales (Target original)
- Las variables más correlacionadas con Target fueron académicas:
  - `Curricular units 2nd sem (approved)`: -0.410
  - `Curricular units 1st sem (approved)`: -0.354
  - `Curricular units 2nd sem (grade)`: -0.271
- Las correlaciones eran negativas porque a mayor valor de variable, menor valor de Target (más cercano a graduación).
- Los indicadores económicos mostraron correlaciones muy débiles con el target.
- Alta multicolinearidad entre métricas del 1er y 2do semestre (r > 0.9).

---

## 3. Transformación del Target a Binario

**Decisión tomada:** Convertir el problema en clasificación binaria **Retención vs Deserción**.

### Regla de transformación
| Target Original | Significado | Target Binario | Significado |
|-----------------|-------------|----------------|-------------|
| 0 | Graduado | 1 | Retenido |
| 1 | Desertó | 0 | Desertó |
| 2 | Matriculado | 1 | Retenido |

### Resultado de la transformación
| Clase | Cantidad | Porcentaje |
|-------|----------|------------|
| 1 — Retenido | 3,003 | 67.88% |
| 0 — Desertó | 1,421 | 32.12% |

- Se preservó el target original en la columna `Target_Original`.
- **Archivo generado:** `dataA_binario.csv` (4,424 filas × 38 columnas).

---

## 4. Análisis Exploratorio de Datos (EDA) — Target Binario

### Gráficas generadas (`./graficas_binario/`)
| # | Gráfica | Descripción |
|---|---------|-------------|
| 01 | `01_distribucion_target_binario.png` | Distribución binaria: Desertó vs Retenido |
| 02 | `02_histogramas_variables.png` | Histogramas de variables numéricas clave |
| 03 | `03_boxplots_por_target.png` | Boxplots comparativos por grupo binario |
| 04 | `04_matriz_correlacion_completa.png` | Mapa de calor de correlaciones (Target binario) |
| 05 | `05_correlacion_con_target.png` | Correlación de todas las variables con Target binario |
| 06 | `06_top_correlaciones.png` | Top 15 correlaciones entre pares de variables |
| 07 | `07_pairplot_correlaciones.png` | Pairplot de variables más correlacionadas |
| 08 | `08_edad_por_target.png` | Distribución de edad por grupo binario |
| 09 | `09_calificaciones_por_semestre.png` | Calificaciones 1er y 2do semestre por grupo |
| 10 | `10_comparacion_medias.png` | Comparación de medias: Top 10 variables más correlacionadas |

### Hallazgos principales (Target binario)
Las correlaciones con el target mejoraron significativamente en magnitud e interpretabilidad:

| Variable | Correlación con Target Binario |
|----------|-------------------------------|
| `Curricular units 2nd sem (grade)` | **+0.572** |
| `Curricular units 2nd sem (approved)` | **+0.570** |
| `Curricular units 1st sem (grade)` | **+0.481** |
| `Curricular units 1st sem (approved)` | **+0.479** |
| `Tuition fees up to date` | **+0.429** |
| `Age at enrollment` | **-0.254** |
| `Scholarship holder` | **+0.245** |
| `Debtor` | **-0.229** |
| `Gender` | **-0.204** |
| `Application mode` | **-0.199** |

### Comparación de medias por grupo (Desertó vs Retenido)

| Variable | Desertó | Retenido | Diferencia |
|----------|---------|----------|------------|
| Calificación 2do sem | 5.90 | 12.28 | **+6.38** |
| Calificación 1er sem | 7.26 | 12.24 | **+4.99** |
| Unidades aprobadas 2do sem | 1.94 | 5.62 | **+3.68** |
| Unidades aprobadas 1er sem | 2.55 | 5.73 | **+3.18** |
| Colegiatura al día | 0.68 | 0.98 | **+0.30** |
| Edad al inscribirse | 26.07 | 21.94 | **-4.13** |
| Becario | 0.09 | 0.32 | **+0.23** |
| Deudor | 0.22 | 0.06 | **-0.16** |
| Género (masculino) | 0.49 | 0.29 | **-0.21** |
| Modo de aplicación | 23.71 | 16.28 | **-7.43** |

### Insights clave
1. **Desempeño académico es el predictor más fuerte:** Los estudiantes retenidos tienen calificaciones aproximadamente el doble de altas.
2. **Edad es un factor de riesgo:** Los desertores se inscriben en promedio 4 años más tarde.
3. **Unidades aprobadas:** Los retenidos aprueban casi 3 veces más unidades curriculares por semestre.
4. **Becas protegen:** El 32.1% de retenidos son becarios vs solo 9.4% de desertores.
5. **Deuda académica:** El 22% de desertores tienen deudas vs 6.4% de retenidos.
6. **Indicadores económicos:** Desempleo, inflación y PIB tienen correlaciones muy débiles (< 0.05) con la retención.
7. **Alta multicolinearidad:** Las métricas del 1er y 2do semestre están fuertemente correlacionadas (r > 0.7), lo cual debe considerarse al seleccionar características para modelado.

---

## 5. Entorno y Dependencias

Se creó un entorno virtual Python e instalaron las siguientes librerías:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

**Ubicación del entorno:** `./venv/`

---

## 6. Archivos de Código Generados

| Archivo | Descripción |
|---------|-------------|
| `analisis_eda.py` | Script de análisis para target original (3 clases) |
| `transformar_target_binario.py` | Script de transformación de target a binario |
| `analisis_eda_binario.py` | Script de análisis para target binario (2 clases) |

---

## 7. Resumen de Entregables

### Datasets
- [x] `dataA.csv` — Dataset original
- [x] `dataA_binario.csv` — Dataset con target binario + columna `Target_Original`

### Gráficas (target original)
- [x] 10 gráficas en `./graficas/`

### Gráficas (target binario)
- [x] 10 gráficas en `./graficas_binario/`

### Scripts
- [x] 3 scripts de Python documentados

---

## 8. Próximos Pasos Sugeridos

1. **Selección de características:** Eliminar variables redundantes por multicolinearidad (ej. mantener solo métricas del 2do semestre o crear índices compuestos).
2. **Balanceo de clases:** Evaluar si el desbalance 68%-32% requiere técnicas como SMOTE o ponderación de clases.
3. **Modelado predictivo:** Entrenar modelos de clasificación binaria (Regresión Logística, Random Forest, XGBoost).
4. **Evaluación:** Usar métricas como Accuracy, Precision, Recall, F1-Score y AUC-ROC.
5. **Interpretabilidad:** Generar SHAP values o importancia de características del modelo.

---

*Bitácora generada automáticamente al finalizar la sesión de análisis.*
