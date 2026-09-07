"""
Análisis del trade-off entre accuracy y tiempo de ejecución.
Resultados de 90 runs sobre data_final.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

RESULTS_PATH = "experiment_results_90runs.csv"
OUTPUT_PREFIX = "comparacion_90runs"


def is_pareto_efficient(costs, return_mask=True):
    """
    Identifica los puntos no dominados.
    costs: array (n_points, 2) donde menor es mejor en ambas dimensiones.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient if return_mask else costs[is_efficient]


def main():
    df = pd.read_csv(RESULTS_PATH)
    print(f"Resultados cargados: {df.shape[0]} filas, {df['run_id'].nunique()} runs")

    os.makedirs(OUTPUT_PREFIX, exist_ok=True)

    # Resumen por método
    summary = df.groupby("method").agg({
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "n_features": "mean",
        "training_time": ["mean", "std"],
    }).reset_index()
    summary.columns = ["method", "accuracy_mean", "accuracy_std", "f1_mean", "f1_std",
                       "roc_auc_mean", "roc_auc_std", "n_features_mean", "time_mean", "time_std"]

    # Normalizar métricas para scores combinados
    summary["accuracy_norm"] = (summary["accuracy_mean"] - summary["accuracy_mean"].min()) / (summary["accuracy_mean"].max() - summary["accuracy_mean"].min())
    summary["time_inv_norm"] = 1 - ((summary["time_mean"] - summary["time_mean"].min()) / (summary["time_mean"].max() - summary["time_mean"].min()))
    summary["roc_auc_norm"] = (summary["roc_auc_mean"] - summary["roc_auc_mean"].min()) / (summary["roc_auc_mean"].max() - summary["roc_auc_mean"].min())
    summary["features_inv_norm"] = 1 - ((summary["n_features_mean"] - summary["n_features_mean"].min()) / (summary["n_features_mean"].max() - summary["n_features_mean"].min()))

    # Scores combinados con diferentes pesos
    summary["score_acc_time"] = 0.7 * summary["accuracy_norm"] + 0.3 * summary["time_inv_norm"]
    summary["score_acc_roc_time"] = 0.5 * summary["accuracy_norm"] + 0.3 * summary["roc_auc_norm"] + 0.2 * summary["time_inv_norm"]
    summary["score_all"] = (0.4 * summary["accuracy_norm"] +
                            0.3 * summary["roc_auc_norm"] +
                            0.2 * summary["time_inv_norm"] +
                            0.1 * summary["features_inv_norm"])

    # Pareto frontier: minimizar tiempo y maximizar accuracy -> (time, -accuracy)
    costs = np.column_stack([summary["time_mean"].values, -summary["accuracy_mean"].values])
    summary["pareto_efficient"] = is_pareto_efficient(costs, return_mask=True)

    # Ordenar por score combinado
    summary_sorted = summary.sort_values("score_all", ascending=False)

    # Guardar tabla
    output_csv = os.path.join(OUTPUT_PREFIX, "tradeoff_accuracy_tiempo_90runs.csv")
    summary_sorted.to_csv(output_csv, index=False)
    print(f"\nGuardado: {output_csv}")
    print(summary_sorted[["method", "accuracy_mean", "roc_auc_mean", "n_features_mean",
                          "time_mean", "score_acc_time", "score_acc_roc_time", "score_all", "pareto_efficient"]].to_string(index=False))

    # Visualización 1: Scatter plot accuracy vs tiempo
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("tab10", n_colors=len(summary))

    for idx, row in summary.iterrows():
        ax.scatter(row["time_mean"], row["accuracy_mean"], s=row["n_features_mean"] * 15,
                   c=[colors[idx]], alpha=0.75, edgecolors="black", linewidth=1.2, label=row["method"])
        # Anotar método
        offset = (0.02, 0.00005) if row["method"] != "baseline" else (-0.15, 0.00005)
        ax.annotate(row["method"], (row["time_mean"], row["accuracy_mean"]),
                    xytext=offset, textcoords="offset points", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Resaltar frontera de Pareto
    pareto_points = summary[summary["pareto_efficient"]].sort_values("time_mean")
    ax.plot(pareto_points["time_mean"], pareto_points["accuracy_mean"], "r--", alpha=0.6, linewidth=2, label="Frontera de Pareto")

    ax.set_xlabel("Tiempo promedio de ejecución (segundos)")
    ax.set_ylabel("Accuracy promedio")
    ax.set_title("Trade-off Accuracy vs Tiempo de ejecución (90 runs)\nTamaño del punto = número de features")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PREFIX, "tradeoff_accuracy_vs_tiempo_90runs.png"), bbox_inches="tight")
    plt.close()
    print(f"Guardado: {os.path.join(OUTPUT_PREFIX, 'tradeoff_accuracy_vs_tiempo_90runs.png')}")

    # Visualización 2: Barras del score combinado
    fig, ax = plt.subplots(figsize=(11, 7))
    summary_plot = summary_sorted.sort_values("score_all", ascending=True)
    colors_score = ["#2ecc71" if pe else "#3498db" for pe in summary_plot["pareto_efficient"]]
    bars = ax.barh(summary_plot["method"], summary_plot["score_all"], color=colors_score)

    for bar, score in zip(bars, summary_plot["score_all"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Score combinado (accuracy + ROC-AUC + velocidad + parsimonia)")
    ax.set_title("Ranking de métodos por trade-off (90 runs)\nVerde = en frontera de Pareto")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PREFIX, "ranking_tradeoff_90runs.png"), bbox_inches="tight")
    plt.close()
    print(f"Guardado: {os.path.join(OUTPUT_PREFIX, 'ranking_tradeoff_90runs.png')}")

    # Visualización 3: Heatmap de métricas normalizadas
    metrics_heatmap = summary.set_index("method")[["accuracy_norm", "roc_auc_norm", "time_inv_norm", "features_inv_norm"]]
    metrics_heatmap.columns = ["Accuracy", "ROC-AUC", "Velocidad", "Parsimonia"]
    metrics_heatmap = metrics_heatmap.loc[summary_sorted["method"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(metrics_heatmap, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Score normalizado"}, ax=ax)
    ax.set_title("Perfil de cada método en 4 dimensiones (90 runs)")
    ax.set_xlabel("Dimensión")
    ax.set_ylabel("Método")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PREFIX, "heatmap_tradeoff_perfil_90runs.png"), bbox_inches="tight")
    plt.close()
    print(f"Guardado: {os.path.join(OUTPUT_PREFIX, 'heatmap_tradeoff_perfil_90runs.png')}")

    print(f"\nTodos los resultados guardados en: {OUTPUT_PREFIX}/")


if __name__ == "__main__":
    main()
