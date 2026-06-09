# 📊 Resultados Comparativos — Selección de Características
**Dataset:** `dataA_binario.csv` | **Runs:** 15 | **Features totales:** 36

---

## Tabla Comparativa (Promedio ± Desv. Estándar)

| Método | Accuracy | Precision | Recall | F1 | ROC AUC | N° Features | Tiempo (s) |
|--------|----------|-----------|--------|----|---------|-------------|------------|
| **baseline** | 0.8778 ± 0.0067 | 0.8790 ± 0.0085 | 0.9511 ± 0.0069 | 0.9136 ± 0.0044 | 0.9160 ± 0.0122 | 36.0 ± 0.0 | 0.0512 ± 0.0062 |
| **variance_threshold** | 0.8778 ± 0.0067 | 0.8790 ± 0.0085 | 0.9511 ± 0.0069 | 0.9136 ± 0.0044 | 0.9160 ± 0.0122 | 36.0 ± 0.0 | 0.0508 ± 0.0058 |
| **chi2** | 0.8511 ± 0.0076 | 0.8615 ± 0.0089 | 0.9306 ± 0.0102 | 0.8946 ± 0.0053 | 0.8925 ± 0.0122 | 10.0 ± 0.0 | 0.0036 ± 0.0001 |
| **mutual_info** | 0.8591 ± 0.0118 | 0.8687 ± 0.0115 | 0.9338 ± 0.0164 | 0.9000 ± 0.0086 | 0.8959 ± 0.0127 | 10.0 ± 0.0 | 0.0058 ± 0.0003 |
| **rfe** | 0.8695 ± 0.0095 | 0.8769 ± 0.0091 | 0.9399 ± 0.0109 | 0.9072 ± 0.0067 | 0.9103 ± 0.0122 | 10.0 ± 0.0 | 0.0037 ± 0.0002 |
| **rfecv** | **0.8786 ± 0.0072** | **0.8813 ± 0.0084** | 0.9493 ± 0.0074 | **0.9140 ± 0.0049** | 0.9153 ± 0.0122 | 29.5 ± 3.6 | 0.0086 ± 0.0011 |
| **l1** | 0.8780 ± 0.0058 | 0.8795 ± 0.0079 | 0.9507 ± 0.0070 | 0.9137 ± 0.0039 | 0.9158 ± 0.0122 | 27.2 ± 1.1 | 0.0073 ± 0.0004 |
| **rf_topk** | 0.8596 ± 0.0118 | 0.8588 ± 0.0103 | 0.9495 ± 0.0164 | 0.9018 ± 0.0086 | 0.8947 ± 0.0137 | 10.0 ± 0.0 | 0.0041 ± 0.0001 |
| **rf_median** | 0.8727 ± 0.0085 | 0.8705 ± 0.0092 | **0.9547 ± 0.0065** | 0.9106 ± 0.0057 | 0.9074 ± 0.0124 | 18.0 ± 0.0 | 0.0057 ± 0.0002 |

---

## 🏆 Ranking por Accuracy

| Posición | Método | Accuracy Promedio | Features Usadas |
|----------|--------|-------------------|-----------------|
| 🥇 | **rfecv** | **0.8786** | ~30 |
| 🥈 | **l1** | **0.8780** | ~27 |
| 🥉 | **baseline / variance_threshold** | **0.8778** | 36 |
| 4 | **rf_median** | 0.8727 | 18 |
| 5 | **rfe** | 0.8695 | 10 |
| 6 | **mutual_info** | 0.8591 | 10 |
| 7 | **rf_topk** | 0.8596 | 10 |
| 8 | **chi2** | 0.8511 | 10 |

---

## 💡 Conclusiones Rápidas

- **Mejor desempeño:** `rfecv` (Accuracy: 87.86%) seleccionando dinámicamente ~30 features.
- **Mejor balance reducción/rendimiento:** `l1` mantiene 87.80% de accuracy con solo ~27 features.
- **Mayor reducción con buen rendimiento:** `rfe` logra 86.95% usando solo 10 features.
- **Más rápido:** `chi2` (0.0036s) y `rfe` (0.0037s) son los métodos de filtro más eficientes.
