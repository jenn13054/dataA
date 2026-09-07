"""
Análisis del desbalance de clases y estrategias de mitigación.
Compara el baseline actual con técnicas de balanceo sobre data_final.csv.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report,
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

DATA_PATH = "data_final.csv"
OUTPUT_PREFIX = "desbalance_clases"
N_RUNS = 15
SEED_BASE = 42


def evaluate_model(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "avg_precision": average_precision_score(y_true, y_proba) if y_proba is not None else np.nan,
    }


def run_experiment(run_id, seed):
    print(f"\nRUN {run_id:02d} | seed={seed}")
    data = pd.read_csv(DATA_PATH)
    target_col = next((c for c in data.columns if c.lower() == "target"), None)
    drop_cols = [c for c in data.columns if c.lower() in ("target", "dropout.semester", "target_original")]
    X = data.drop(columns=drop_cols)
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    records = []
    models = {}

    # 1) Baseline actual
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="sag", random_state=seed))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_proba)
    records.append({"run_id": run_id, "strategy": "baseline", **metrics})
    models["baseline"] = (y_test, y_proba)

    # 2) Baseline con class_weight='balanced'
    pipe_bal = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="sag", class_weight="balanced", random_state=seed))
    ])
    pipe_bal.fit(X_train, y_train)
    y_pred = pipe_bal.predict(X_test)
    y_proba = pipe_bal.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_proba)
    records.append({"run_id": run_id, "strategy": "class_weight_balanced", **metrics})
    models["class_weight_balanced"] = (y_test, y_proba)

    # 3) Random Oversampling + LogisticRegression
    pipe_ros = ImbPipeline([
        ("scaler", StandardScaler()),
        ("ros", RandomOverSampler(random_state=seed)),
        ("clf", LogisticRegression(max_iter=5000, solver="sag", random_state=seed))
    ])
    pipe_ros.fit(X_train, y_train)
    y_pred = pipe_ros.predict(X_test)
    y_proba = pipe_ros.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_proba)
    records.append({"run_id": run_id, "strategy": "random_oversampling", **metrics})
    models["random_oversampling"] = (y_test, y_proba)

    # 4) SMOTE + LogisticRegression
    pipe_smote = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", LogisticRegression(max_iter=5000, solver="sag", random_state=seed))
    ])
    pipe_smote.fit(X_train, y_train)
    y_pred = pipe_smote.predict(X_test)
    y_proba = pipe_smote.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_proba)
    records.append({"run_id": run_id, "strategy": "smote", **metrics})
    models["smote"] = (y_test, y_proba)

    # 5) Baseline con umbral ajustado (threshold tuning para maximizar F2)
    pipe_thr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="sag", random_state=seed))
    ])
    pipe_thr.fit(X_train, y_train)
    y_proba = pipe_thr.predict_proba(X_test)[:, 1]

    # Encontrar umbral que maximice F2 en validación
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)
    pipe_thr.fit(X_tr, y_tr)
    val_proba = pipe_thr.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_f2 = 0
    best_thr = 0.5
    for thr in thresholds:
        y_val_pred = (val_proba >= thr).astype(int)
        f2 = fbeta_score(y_val, y_val_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = thr

    y_pred = (y_proba >= best_thr).astype(int)
    metrics = evaluate_model(y_test, y_pred, y_proba)
    metrics["best_threshold"] = best_thr
    records.append({"run_id": run_id, "strategy": "threshold_tuning_f2", **metrics})
    models["threshold_tuning_f2"] = (y_test, y_proba)

    # 6) rf_topk + class_weight balanced (mejor trade-off previo)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    selected = list(X.columns[top_idx])

    pipe_rf_topk = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=seed))
    ])
    pipe_rf_topk.fit(X_train[selected], y_train)
    y_pred = pipe_rf_topk.predict(X_test[selected])
    y_proba = pipe_rf_topk.predict_proba(X_test[selected])[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_proba)
    records.append({"run_id": run_id, "strategy": "rf_topk_balanced", **metrics})
    models["rf_topk_balanced"] = (y_test, y_proba)

    return pd.DataFrame(records), models


def plot_metric_comparison(df, output_path):
    metrics = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "avg_precision"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        order = df.groupby("strategy")[metric].median().sort_values(ascending=False).index.tolist()
        sns.boxplot(data=df, x="strategy", y=metric, hue="strategy", hue_order=order, order=order,
                    ax=ax, palette="Set2", legend=False)
        ax.set_title(metric.upper())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    plt.suptitle("Comparación de estrategias ante desbalance de clases (15 runs)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def plot_curves(models, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curves
    ax = axes[0]
    for strategy, (y_true, y_proba) in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{strategy} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Precision-Recall curves
    ax = axes[1]
    for strategy, (y_true, y_proba) in models.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, label=f"{strategy} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Curvas ROC y Precision-Recall (último run)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def plot_confusion_matrices(models, output_path):
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for idx, (strategy, (y_true, y_proba)) in enumerate(models.items()):
        y_pred = (y_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
        axes[idx].set_title(strategy)
        axes[idx].set_xlabel("Predicho")
        axes[idx].set_ylabel("Real")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Matrices de confusión (último run, umbral 0.5)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {output_path}")


def main():
    os.makedirs(OUTPUT_PREFIX, exist_ok=True)

    all_records = []
    last_models = None

    for run_id in range(N_RUNS):
        seed = SEED_BASE + run_id
        df_run, models = run_experiment(run_id, seed)
        all_records.append(df_run)
        last_models = models

    df_all = pd.concat(all_records, ignore_index=True)
    results_path = os.path.join(OUTPUT_PREFIX, "resultados_desbalance_15runs.csv")
    df_all.to_csv(results_path, index=False)
    print(f"\nGuardado: {results_path}")

    # Resumen
    summary = df_all.groupby("strategy").agg(["mean", "std"]).round(4)
    summary_path = os.path.join(OUTPUT_PREFIX, "resumen_desbalance_15runs.csv")
    summary.to_csv(summary_path)
    print(f"Guardado: {summary_path}")
    print(summary)

    # Visualizaciones
    plot_metric_comparison(df_all, os.path.join(OUTPUT_PREFIX, "metricas_desbalance_15runs.png"))
    plot_curves(last_models, os.path.join(OUTPUT_PREFIX, "curvas_roc_pr_desbalance.png"))
    plot_confusion_matrices(last_models, os.path.join(OUTPUT_PREFIX, "matrices_confusion_desbalance.png"))

    print(f"\nTodos los resultados guardados en: {OUTPUT_PREFIX}/")


if __name__ == "__main__":
    main()
