# 📊 Resultados Comparativos — Selección de Características (90 runs)
**Dataset:** `dataA_binario.csv` | **Runs:** 90 | **Features totales:** 36

---

## Tabla Comparativa (Promedio ± Desv. Estándar)

| Método | Accuracy | Precision | Recall | F1 | ROC AUC | N° Features | Tiempo (s) |
|--------|----------|-----------|--------|----|---------|-------------|------------|
| **baseline** | 0.8755 ± 0.0092 | 0.8784 ± 0.0103 | 0.9480 ± 0.0098 | 0.9118 ± 0.0064 | 0.9163 ± 0.0104 | 36.0 ± 0.0 | 0.0520 ± 0.0070 |
| **variance_threshold** | 0.8755 ± 0.0092 | 0.8784 ± 0.0103 | 0.9480 ± 0.0098 | 0.9118 ± 0.0064 | 0.9163 ± 0.0104 | 36.0 ± 0.1 | 0.0517 ± 0.0069 |
| **chi2** | 0.8482 ± 0.0107 | 0.8605 ± 0.0109 | 0.9268 ± 0.0117 | 0.8924 ± 0.0075 | 0.8924 ± 0.0124 | 10.0 ± 0.0 | 0.0037 ± 0.0002 |
| **mutual_info** | 0.8581 ± 0.0117 | 0.8653 ± 0.0128 | 0.9372 ± 0.0170 | 0.8997 ± 0.0085 | 0.8971 ± 0.0124 | 10.0 ± 0.0 | 0.0062 ± 0.0004 |
| **rfe** | 0.8671 ± 0.0099 | 0.8751 ± 0.0105 | 0.9383 ± 0.0109 | 0.9056 ± 0.0069 | 0.9103 ± 0.0114 | 10.0 ± 0.0 | 0.0038 ± 0.0002 |
| **rfecv** | 0.8749 ± 0.0099 | **0.8791 ± 0.0107** | 0.9461 ± 0.0100 | 0.9113 ± 0.0069 | 0.9156 ± 0.0106 | 30.0 ± 3.6 | 0.0087 ± 0.0011 |
| **l1** | 0.8750 ± 0.0098 | 0.8785 ± 0.0108 | 0.9472 ± 0.0100 | 0.9115 ± 0.0067 | 0.9162 ± 0.0105 | 27.0 ± 1.7 | 0.0074 ± 0.0006 |
| **rf_topk** | 0.8589 ± 0.0117 | 0.8579 ± 0.0111 | 0.9498 ± 0.0161 | 0.9014 ± 0.0084 | 0.8944 ± 0.0127 | 10.0 ± 0.0 | 0.0042 ± 0.0001 |
| **rf_median** | 0.8706 ± 0.0100 | 0.8696 ± 0.0106 | **0.9523 ± 0.0097** | 0.9091 ± 0.0068 | 0.9072 ± 0.0120 | 18.0 ± 0.0 | 0.0058 ± 0.0002 |

---

## 🏆 Ranking por Accuracy

| Posición | Método | Accuracy Promedio | Features Usadas |
|----------|--------|-------------------|-----------------|
| 🥇 | **baseline / variance_threshold** | **0.8755** | 36 |
| 🥈 | **l1** | **0.8750** | ~27 |
| 🥉 | **rfecv** | **0.8749** | ~30 |
| 4 | **rf_median** | 0.8706 | 18 |
| 5 | **rfe** | 0.8671 | 10 |
| 6 | **rf_topk** | 0.8589 | 10 |
| 7 | **mutual_info** | 0.8581 | 10 |
| 8 | **chi2** | 0.8482 | 10 |

---

## 💡 Conclusiones Rápidas

- **El baseline se consolida en la primera posición** con **87.55%**, empatado con `variance_threshold`.
- **`l1` sube al segundo lugar** (87.50%), superando a `rfecv` (87.49%) por una diferencia mínima.
- **`rfecv` mantiene la mayor precision** (87.91%) con selección automática robusta.
- **`l1` es la mejor opción de reducción/rendimiento**: 25% menos de features con pérdida imperceptible.
- **`rfe` sigue siendo la mejor con 10 features**: 86.71% de accuracy.
- **`rf_median`** ofrece 87.06% con 18 features, reduciendo a la mitad la dimensionalidad.
- **Con 90 runs los resultados están plenamente estabilizados**: las diferencias entre los top 3 son <0.06%.
