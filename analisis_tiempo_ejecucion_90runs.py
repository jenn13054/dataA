"""
Análisis estadístico del tiempo de ejecución (training_time) entre métodos.
Prueba t pareada vs baseline usando los 90 runs de experiment_results_90runs.csv.
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
ALPHA = 0.05


def paired_ttest_time(df, baseline="baseline"):
    """Prueba t pareada del training_time de cada método contra el baseline."""
    pivot = df.pivot(index="run_id", columns="method", values="training_time")
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
        speedup = np.mean(baseline_vals) / np.mean(method_vals)
        records.append({
            "method": method,
            "baseline_mean_sec": np.mean(baseline_vals),
            "method_mean_sec": np.mean(method_vals),
            "mean_diff_sec": mean_diff,
            "std_diff_sec": np.std(diff, ddof=1),
            "t_statistic": t_stat,
            "p_value": p_value,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "significant": p_value < ALPHA,
            "speedup_vs_baseline": speedup,
            "n_runs": len(diff),
        })
    return pd.DataFrame(records)


def plot_training_time(df, ttest_df, output_path):
    summary = df.groupby("method")["training_time"].agg(["mean", "std"]).reset_index()
    summary = summary.merge(ttest_df[["method", "p_value"]], on="method", how="left")
    summary["p_text"] = summary["p_value"].apply(lambda p: f"p={p:.2e}" if pd.notna(p) else "baseline")
    summary = summary.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#2ecc71" if p < ALPHA else "#e74c3c" for p in summary["p_value"].fillna(1)]
    bars = ax.barh(summary["method"], summary["mean"], xerr=summary["std"], capsize=4, color=colors, alpha=0.8)

    for i, (bar, p_text, mean_val) in enumerate(zip(bars, summary["p_text"], summary["mean"])):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{mean_val:.3f}s | {p_text}", va="center", fontsize=9)

    ax.set_xlabel("Tiempo de entrenamiento promedio (segundos)")
    ax.set_title("Tiempo de ejecución por método vs baseline (90 runs)\nVerde = significativamente más rápido que baseline")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def plot_boxplot_training_time(df, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby("method")["training_time"].median().sort_values(ascending=True).index.tolist()
    sns.boxplot(data=df, x="method", y="training_time", hue="method", hue_order=order, order=order,
                ax=ax, palette="viridis", legend=False)
    ax.set_yscale("log")
    ax.set_ylabel("Tiempo de entrenamiento (segundos, escala log)")
    ax.set_xlabel("Método")
    ax.set_title("Distribución del tiempo de ejecución por método (90 runs, escala log)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def main():
    df = pd.read_csv(RESULTS_PATH)
    print(f"Resultados cargados: {df.shape[0]} filas, {df['run_id'].nunique()} runs, {df['method'].nunique()} métodos")

    os.makedirs(OUTPUT_PREFIX, exist_ok=True)

    ttest_df = paired_ttest_time(df, baseline="baseline")
    ttest_path = os.path.join(OUTPUT_PREFIX, "ttest_tiempo_ejecucion_vs_baseline_90runs.csv")
    ttest_df.to_csv(ttest_path, index=False)
    print(f"\nGuardado t-test de tiempos: {ttest_path}")
    print(ttest_df[["method", "method_mean_sec", "mean_diff_sec", "speedup_vs_baseline", "p_value", "significant"]].to_string(index=False))

    plot_training_time(df, ttest_df, os.path.join(OUTPUT_PREFIX, "tiempo_ejecucion_vs_baseline_90runs.png"))
    plot_boxplot_training_time(df, os.path.join(OUTPUT_PREFIX, "boxplot_tiempo_ejecucion_90runs.png"))

    print(f"\nTodos los resultados guardados en: {OUTPUT_PREFIX}/")


if __name__ == "__main__":
    main()
