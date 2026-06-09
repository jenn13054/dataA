# 📊 Resultados Comparativos — Selección de Características (60 runs)
**Dataset:** `dataA_binario.csv` | **Runs:** 60 | **Features totales:** 36

---

## Tabla Comparativa (Promedio ± Desv. Estándar)

| Método | Accuracy | Precision | Recall | F1 | ROC AUC | N° Features | Tiempo (s) |
|--------|----------|-----------|--------|----|---------|-------------|------------|
| **baseline** | 0.8755 ± 0.0092 | 0.8783 ± 0.0108 | 0.9482 ± 0.0095 | 0.9118 ± 0.0063 | 0.9161 ± 0.0111 | 36.0 ± 0.0 | 0.0517 ± 0.0063 |
| **variance_threshold** | 0.8755 ± 0.0092 | 0.8783 ± 0.0108 | 0.9483 ± 0.0096 | 0.9118 ± 0.0063 | 0.9161 ± 0.0111 | 36.0 ± 0.2 | 0.0514 ± 0.0062 |
| **chi2** | 0.8472 ± 0.0109 | 0.8596 ± 0.0114 | 0.9266 ± 0.0122 | 0.8918 ± 0.0075 | 0.8925 ± 0.0131 | 10.0 ± 0.0 | 0.0037 ± 0.0001 |
| **mutual_info** | 0.8570 ± 0.0129 | 0.8646 ± 0.0131 | 0.9365 ± 0.0187 | 0.8989 ± 0.0093 | 0.8969 ± 0.0134 | 10.0 ± 0.0 | 0.0061 ± 0.0003 |
| **rfe** | 0.8670 ± 0.0102 | 0.8750 ± 0.0113 | 0.9384 ± 0.0107 | 0.9055 ± 0.0070 | 0.9104 ± 0.0121 | 10.0 ± 0.0 | 0.0038 ± 0.0002 |
| **rfecv** | 0.8751 ± 0.0100 | **0.8791 ± 0.0111** | 0.9465 ± 0.0094 | 0.9115 ± 0.0069 | 0.9155 ± 0.0112 | 30.0 ± 3.6 | 0.0088 ± 0.0011 |
| **l1** | 0.8750 ± 0.0099 | 0.8785 ± 0.0113 | 0.9473 ± 0.0098 | 0.9115 ± 0.0067 | 0.9159 ± 0.0111 | 27.0 ± 1.7 | 0.0074 ± 0.0006 |
| **rf_topk** | 0.8581 ± 0.0122 | 0.8580 ± 0.0117 | 0.9482 ± 0.0163 | 0.9007 ± 0.0087 | 0.8944 ± 0.0138 | 10.0 ± 0.0 | 0.0041 ± 0.0002 |
| **rf_median** | 0.8701 ± 0.0105 | 0.8691 ± 0.0113 | **0.9523 ± 0.0094** | 0.9087 ± 0.0071 | 0.9070 ± 0.0127 | 18.0 ± 0.0 | 0.0059 ± 0.0005 |

---

## 🏆 Ranking por Accuracy

| Posición | Método | Accuracy Promedio | Features Usadas |
|----------|--------|-------------------|-----------------|
| 🥇 | **baseline / variance_threshold** | **0.8755** | 36 |
| 🥈 | **rfecv** | **0.8751** | ~30 |
| 🥉 | **l1** | **0.8750** | ~27 |
| 4 | **rf_median** | 0.8701 | 18 |
| 5 | **rfe** | 0.8670 | 10 |
| 6 | **mutual_info** | 0.8570 | 10 |
| 7 | **rf_topk** | 0.8581 | 10 |
| 8 | **chi2** | 0.8472 | 10 |

---

## 💡 Conclusiones Rápidas

- **Con 60 runs el baseline recupera la primera posición** con **87.55%**, aunque la diferencia con `rfecv` (87.51%) y `l1` (87.50%) es marginal (<0.05%).
- **`rfecv` mantiene la mayor precision** (87.91%) con selección automática por validación cruzada.
- **`l1` ofrece el mejor balance** de reducción y desempeño: ~27 features con prácticamente el mismo accuracy que el baseline.
- **`rfe` es la mejor opción con 10 features**: 86.70% de accuracy con máxima reducción dimensional.
- **`rf_median`** logra 87.01% con solo 18 features, una reducción del 50% con pérdida mínima.
- **Estabilidad**: A mayor número de runs, los métodos wrapper (`rfecv`, `l1`) se consolidan como alternativas sólidas al baseline completo.
