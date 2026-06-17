"""
Prueba t de Student pareada unilateral (cola izquierda) sobre los resultados de 60 runs.
Hipotesis alternativa: metodo < baseline.
Compara cada metodo de seleccion de caracteristicas contra el baseline.
"""

import pandas as pd
import numpy as np
from scipy import stats

# Cargar resultados
df = pd.read_csv('experiment_results_60runs.csv')

# Verificar que tenemos 60 runs
runs = df['run_id'].unique()
print(f"Runs encontrados: {sorted(runs)} (total: {len(runs)})")
assert len(runs) == 60, f"Se esperaban 60 runs, se encontraron {len(runs)}"

methods = df['method'].unique()
print(f"Metodos: {list(methods)}\n")

# Metricas a comparar
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Separar baseline
baseline = df[df['method'] == 'baseline']

# Metodos a comparar contra baseline (excluyendo baseline mismo)
methods_to_compare = [m for m in methods if m != 'baseline']

alpha = 0.05
# Correccion de Bonferroni: numero de comparaciones independientes
n_comparisons = len(methods_to_compare) * len(metrics)
alpha_bonferroni = alpha / n_comparisons

print("=" * 100)
print("PRUEBA T DE STUDENT PAREADA UNILATERAL (IZQUIERDA): CADA METODO VS BASELINE (60 runs)")
print("=" * 100)
print(f"Nivel de significancia alpha = {alpha}")
print(f"Correccion de Bonferroni: {n_comparisons} comparaciones -> alpha_adj = {alpha_bonferroni:.6f}")
print(f"Hipotesis nula H0: mu_metodo >= mu_baseline")
print(f"Hipotesis alternativa H1: mu_metodo < mu_baseline (metodo es inferior al baseline)\n")

results = []

for metric in metrics:
    print(f"\n--- Metrica: {metric.upper()} ---")
    print(f"{'Metodo':<20} {'Media':>8} {'+-':>3} {'DE':>6} {'Diff':>8} {'t':>8} {'gl':>4} {'p-value':>12} {'IC 95%':>22} {'Sig.':>4} {'Bonf.':>5}")
    print("-" * 110)
    
    baseline_values = baseline.sort_values('run_id')[metric].values
    baseline_mean = baseline_values.mean()
    baseline_std = baseline_values.std(ddof=1)
    
    for method in methods_to_compare:
        method_df = df[df['method'] == method].sort_values('run_id')
        method_values = method_df[metric].values
        
        # Verificar que los run_id coincidan
        assert np.array_equal(method_df['run_id'].values, baseline['run_id'].values), \
            f"Run IDs no coinciden para {method}"
        
        # Prueba t pareada unilateral: H1: metodo < baseline
        diff = method_values - baseline_values
        if np.all(diff == 0):
            t_stat = 0.0
            p_value = 1.0
            ci_low, ci_high = 0.0, 0.0
        else:
            t_stat, p_value = stats.ttest_rel(method_values, baseline_values, alternative='less')
            # Intervalo de confianza unilateral (cola izquierda) para la diferencia
            n = len(diff)
            se = np.std(diff, ddof=1) / np.sqrt(n)
            t_crit = stats.t.ppf(1 - alpha, df=n-1)  # unilateral
            ci_low = -np.inf
            ci_high = diff.mean() + t_crit * se
        
        method_mean = method_values.mean()
        method_std = method_values.std(ddof=1)
        diff_mean = diff.mean()
        gl = len(diff) - 1
        
        significativo = "SI" if p_value < alpha else "NO"
        significativo_bonf = "SI" if p_value < alpha_bonferroni else "NO"
        
        results.append({
            'metric': metric,
            'method': method,
            'mean_method': method_mean,
            'std_method': method_std,
            'mean_baseline': baseline_mean,
            'std_baseline': baseline_std,
            'diff_mean': diff_mean,
            't_statistic': t_stat,
            'df': gl,
            'p_value': p_value,
            'ci_95_high': ci_high,
            'significant_alpha_05': p_value < alpha,
            'significant_bonferroni': p_value < alpha_bonferroni,
            'alternative_hypothesis': 'method < baseline'
        })
        
        print(f"{method:<20} {method_mean:>8.4f} {'+-':>3} {method_std:>6.4f} {diff_mean:>8.4f} {t_stat:>8.3f} {gl:>4} {p_value:>12.6f} [-inf, {ci_high:>8.4f}] {significativo:>4} {significativo_bonf:>5}")

# Convertir a DataFrame y guardar
results_df = pd.DataFrame(results)
results_df.to_csv('prueba_t_student_60runs_unilateral_menor_resultados.csv', index=False)

# Resumen por metrica: metodos significativamente inferiores al baseline
print("\n\n" + "=" * 100)
print("RESUMEN: METODOS SIGNIFICATIVAMENTE INFERIORES AL BASELINE")
print("=" * 100)
for metric in metrics:
    print(f"\nMetrica: {metric.upper()}")
    subset = results_df[results_df['metric'] == metric]
    
    sig_05 = subset[subset['p_value'] < alpha]
    sig_bonf = subset[subset['p_value'] < alpha_bonferroni]
    
    print(f"  alpha = 0.05: {len(sig_05)} metodos significativamente inferiores")
    for _, row in sig_05.iterrows():
        print(f"    - {row['method']}: diff={row['diff_mean']:+.4f}, t={row['t_statistic']:.3f}, p={row['p_value']:.6f}")
    
    if len(sig_05) == 0:
        print("    Ningun metodo es significativamente inferior al baseline.")
    
    print(f"  alpha Bonferroni = {alpha_bonferroni:.6f}: {len(sig_bonf)} metodos significativamente inferiores")
    for _, row in sig_bonf.iterrows():
        print(f"    - {row['method']}: diff={row['diff_mean']:+.4f}, t={row['t_statistic']:.3f}, p={row['p_value']:.6f}")

print("\n" + "=" * 100)
print("RESULTADOS GUARDADOS EN: prueba_t_student_60runs_unilateral_menor_resultados.csv")
print("=" * 100)
