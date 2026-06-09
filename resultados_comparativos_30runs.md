# 📊 Resultados Comparativos — Selección de Características (30 runs)
**Dataset:** `dataA_binario.csv` | **Runs:** 30 | **Features totales:** 36

---

## Tabla Comparativa (Promedio ± Desv. Estándar)

| Método | Accuracy | Precision | Recall | F1 | ROC AUC | N° Features | Tiempo (s) |
|--------|----------|-----------|--------|----|---------|-------------|------------|
| **baseline** | 0.8769 ± 0.0081 | 0.8790 ± 0.0105 | 0.9496 ± 0.0087 | 0.9129 ± 0.0054 | 0.9171 ± 0.0103 | 36.0 ± 0.0 | 0.0519 ± 0.0074 |
| **variance_threshold** | 0.8769 ± 0.0081 | 0.8790 ± 0.0105 | 0.9496 ± 0.0087 | 0.9129 ± 0.0054 | 0.9171 ± 0.0103 | 36.0 ± 0.0 | 0.0514 ± 0.0071 |
| **chi2** | 0.8496 ± 0.0094 | 0.8622 ± 0.0098 | 0.9268 ± 0.0123 | 0.8932 ± 0.0067 | 0.8942 ± 0.0121 | 10.0 ± 0.0 | 0.0036 ± 0.0002 |
| **mutual_info** | 0.8650 ± 0.0109 | 0.8678 ± 0.0114 | 0.9456 ± 0.0164 | 0.9049 ± 0.0079 | 0.9007 ± 0.0133 | 10.0 ± 0.0 | 0.0063 ± 0.0006 |
| **rfe** | 0.8690 ± 0.0090 | 0.8763 ± 0.0101 | 0.9399 ± 0.0114 | 0.9069 ± 0.0063 | 0.9112 ± 0.0113 | 10.0 ± 0.0 | 0.0038 ± 0.0002 |
| **rfecv** | **0.8771 ± 0.0090** | **0.8808 ± 0.0107** | 0.9474 ± 0.0090 | 0.9128 ± 0.0061 | 0.9165 ± 0.0103 | 30.1 ± 3.6 | 0.0087 ± 0.0011 |
| **l1** | **0.8774 ± 0.0077** | 0.8798 ± 0.0103 | 0.9494 ± 0.0086 | **0.9132 ± 0.0052** | 0.9169 ± 0.0103 | 27.1 ± 1.8 | 0.0073 ± 0.0005 |
| **rf_topk** | 0.8604 ± 0.0103 | 0.8591 ± 0.0108 | 0.9506 ± 0.0137 | 0.9024 ± 0.0073 | 0.8962 ± 0.0125 | 10.0 ± 0.0 | 0.0041 ± 0.0002 |
| **rf_median** | 0.8719 ± 0.0089 | 0.8702 ± 0.0098 | **0.9539 ± 0.0080** | 0.9101 ± 0.0059 | 0.9082 ± 0.0112 | 18.0 ± 0.0 | 0.0057 ± 0.0003 |

---

## 🏆 Ranking por Accuracy

| Posición | Método | Accuracy Promedio | Features Usadas |
|----------|--------|-------------------|-----------------|
| 🥇 | **l1** | **0.8774** | ~27 |
| 🥈 | **rfecv** | **0.8771** | ~30 |
| 🥉 | **baseline / variance_threshold** | **0.8769** | 36 |
| 4 | **rf_median** | 0.8719 | 18 |
| 5 | **rfe** | 0.8690 | 10 |
| 6 | **mutual_info** | 0.8650 | 10 |
| 7 | **rf_topk** | 0.8604 | 10 |
| 8 | **chi2** | 0.8496 | 10 |

---

## 💡 Conclusiones Rápidas

- **Mejor desempeño:** `l1` lidera con **87.74%** de accuracy, reduciendo de 36 a ~27 features.
- **Mayor precisión:** `rfecv` obtiene el mejor precision (**88.08%**) y prácticamente el mismo accuracy que `l1`.
- **Mayor reducción con buen rendimiento:** `rfe` logra **86.90%** usando solo **10 features**, siendo muy eficiente computacionalmente (0.0038s).
- **Más rápido:** `chi2` (0.0036s) es el método más veloz, aunque con el menor accuracy.
- **Consistencia:** Con 30 runs los resultados son estables; `l1` y `rfecv` superan marginalmente al baseline completo.
