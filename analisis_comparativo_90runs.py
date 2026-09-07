"""
Comparación estadística y visual de métodos de selección de características
Resultados de 90 runs sobre data_final.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

RESULTS_PATH = "experiment_results_90runs.csv"
OUTPUT_PREFIX = "comparacion_90runs"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "n_features", "training_time"]
ALPHA = 0.05


def load_results(path):
    df = pd.read_csv(path)
    print(f"Resultados cargados: {df.shape[0]} filas, {df['run_id'].nunique()} runs, {df['method'].nunique()} métodos")
    return df


def paired_ttest(df, metric, baseline="baseline"):
    """Prueba t pareada de cada método contra el baseline."""
    pivot = df.pivot(index="run_id", columns="method", values=metric)
    baseline_vals = pivot[baseline].values
    methods = [c for c in pivot.columns if c != baseline]

    records = []
    for method in methods:
        method_vals = pivot[method].values
        diff = method_vals - baseline_vals
        t_stat, p_value = stats.ttest_rel(method_vals, baseline_vals)
        mean_diff = np.mean(diff)
        ci_low, ci_high = stats.t.interval(
            confidence=1 - ALPHA,
            df=len(diff) - 1,
            loc=mean_diff,
            scale=stats.sem(diff),
        )
        records.append({
            "method": method,
            "metric": metric,
            "baseline_mean": np.mean(baseline_vals),
            "method_mean": np.mean(method_vals),
            "mean_diff": mean_diff,
            "std_diff": np.std(diff, ddof=1),
            "t_statistic": t_stat,
            "p_value": p_value,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "significant": p_value < ALPHA,
            "n_runs": len(diff),
        })
    return pd.DataFrame(records)


def plot_metric_boxplots(df, metrics, output_path):
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        order = df.groupby("method")[metric].median().sort_values(ascending=False).index.tolist()
        sns.boxplot(data=df, x="method", y=metric, hue="method", hue_order=order, order=order, ax=ax, palette="Set2", legend=False)
        ax.set_title(f"{metric.upper()}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Distribución de métricas por método (90 runs)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def plot_mean_comparison(df, metric, output_path):
    summary = df.groupby("method")[metric].agg(["mean", "std"]).sort_values("mean", ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(summary["method"], summary["mean"], yerr=summary["std"], capsize=4, color=sns.color_palette("Set2"))
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} promedio ± desv. est. por método (90 runs)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # Anotar valores
    for bar, mean_val in zip(bars, summary["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean_val:.4f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def plot_significance_heatmap(ttest_df, metric, output_path):
    pivot = ttest_df[ttest_df["metric"] == metric].set_index("method")[["p_value"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", vmin=0, vmax=0.1,
                cbar_kws={"label": "p-value"}, ax=ax)
    ax.set_title(f"p-values prueba t pareada vs baseline ({metric}, 90 runs)")
    ax.set_ylabel("Método")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def main():
    df = load_results(RESULTS_PATH)

    os.makedirs(OUTPUT_PREFIX, exist_ok=True)

    # 1) Tabla resumen compacta
    summary = df.groupby("method")[METRICS].agg(["mean", "std"]).round(4)
    summary_path = os.path.join(OUTPUT_PREFIX, "resumen_metricas_90runs.csv")
    summary.to_csv(summary_path)
    print(f"Guardado resumen: {summary_path}")
    print(summary)

    # 2) Pruebas t pareadas contra baseline
    ttest_all = []
    for metric in ["accuracy", "f1", "roc_auc"]:
        ttest_df = paired_ttest(df, metric, baseline="baseline")
        ttest_all.append(ttest_df)

    ttest_combined = pd.concat(ttest_all, ignore_index=True)
    ttest_path = os.path.join(OUTPUT_PREFIX, "ttest_pareado_vs_baseline_90runs.csv")
    ttest_combined.to_csv(ttest_path, index=False)
    print(f"\nGuardado t-test: {ttest_path}")
    print(ttest_combined[["method", "metric", "mean_diff", "p_value", "significant"]].to_string(index=False))

    # 3) Visualizaciones
    plot_metric_boxplots(df, ["accuracy", "precision", "recall", "f1", "roc_auc", "n_features"],
                         os.path.join(OUTPUT_PREFIX, "boxplots_metricas_90runs.png"))

    plot_mean_comparison(df, "accuracy", os.path.join(OUTPUT_PREFIX, "accuracy_promedio_90runs.png"))
    plot_mean_comparison(df, "roc_auc", os.path.join(OUTPUT_PREFIX, "roc_auc_promedio_90runs.png"))
    plot_mean_comparison(df, "n_features", os.path.join(OUTPUT_PREFIX, "n_features_promedio_90runs.png"))

    plot_significance_heatmap(ttest_combined, "accuracy", os.path.join(OUTPUT_PREFIX, "ttest_accuracy_heatmap_90runs.png"))
    plot_significance_heatmap(ttest_combined, "roc_auc", os.path.join(OUTPUT_PREFIX, "ttest_roc_auc_heatmap_90runs.png"))

    print(f"\nTodos los resultados guardados en: {OUTPUT_PREFIX}/")


if __name__ == "__main__":
    main()
