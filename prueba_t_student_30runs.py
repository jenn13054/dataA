"""
Prueba t de Student pareada sobre los resultados de 30 runs.
Compara cada método de selección de características contra el baseline.
"""

import pandas as pd
import numpy as np
from scipy import stats

# Cargar resultados
df = pd.read_csv('experiment_results.csv')

# Verificar que tenemos 30 runs
runs = df['run_id'].unique()
print(f"Runs encontrados: {sorted(runs)} (total: {len(runs)})")
assert len(runs) == 30, f"Se esperaban 30 runs, se encontraron {len(runs)}"

methods = df['method'].unique()
print(f"Métodos: {list(methods)}\n")

# Métricas a comparar
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Separar baseline
baseline = df[df['method'] == 'baseline']

# Métodos a comparar contra baseline (excluyendo baseline mismo)
methods_to_compare = [m for m in methods if m != 'baseline']

alpha = 0.05
# Corrección de Bonferroni: número de comparaciones independientes
# (métodos × métricas)
n_comparisons = len(methods_to_compare) * len(metrics)
alpha_bonferroni = alpha / n_comparisons

print("=" * 90)
print("PRUEBA T DE STUDENT PAREADA: CADA MÉTODO VS BASELINE (30 runs)")
print("=" * 90)
print(f"Nivel de significancia α = {alpha}")
print(f"Corrección de Bonferroni: {n_comparisons} comparaciones → α_adj = {alpha_bonferroni:.6f}")
print(f"Hipótesis nula H₀: μ_método = μ_baseline (no hay diferencia)")
print(f"Hipótesis alternativa H₁: μ_método ≠ μ_baseline (diferencia bilateral)\n")

results = []

for metric in metrics:
    print(f"\n--- Métrica: {metric.upper()} ---")
    print(f"{'Método':<20} {'Media':>8} {'±':>3} {'DE':>6} {'Diff':>7} {'t':>8} {'gl':>4} {'p-value':>12} {'IC 95%':>22} {'Signif.':>9}")
    print("-" * 105)
    
    baseline_values = baseline.sort_values('run_id')[metric].values
    baseline_mean = baseline_values.mean()
    baseline_std = baseline_values.std(ddof=1)
    
    for method in methods_to_compare:
        method_df = df[df['method'] == method].sort_values('run_id')
        method_values = method_df[metric].values
        
        # Verificar que los run_id coincidan
        assert np.array_equal(method_df['run_id'].values, baseline['run_id'].values), \
            f"Run IDs no coinciden para {method}"
        
        # Prueba t pareada
        # Si las varianzas de las diferencias son cero, evitar warning
        diff = method_values - baseline_values
        if np.all(diff == 0):
            t_stat = 0.0
            p_value = 1.0
            ci_low, ci_high = 0.0, 0.0
        else:
            t_stat, p_value = stats.ttest_rel(method_values, baseline_values)
            # Intervalo de confianza para la diferencia de medias
            n = len(diff)
            se = np.std(diff, ddof=1) / np.sqrt(n)
            t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
            ci_low = diff.mean() - t_crit * se
            ci_high = diff.mean() + t_crit * se
        
        method_mean = method_values.mean()
        method_std = method_values.std(ddof=1)
        diff_mean = diff.mean()
        gl = len(diff) - 1
        
        significativo = "SÍ" if p_value < alpha else "NO"
        significativo_bonf = "SÍ" if p_value < alpha_bonferroni else "NO"
        
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
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'significant_alpha_05': p_value < alpha,
            'significant_bonferroni': p_value < alpha_bonferroni
        })
        
        print(f"{method:<20} {method_mean:>8.4f} {'±':>3} {method_std:>6.4f} {diff_mean:>7.4f} {t_stat:>8.3f} {gl:>4} {p_value:>12.6f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {significativo:>4} ({'Sí' if significativo_bonf=='SÍ' else 'No'} Bonf.)")

# Convertir a DataFrame y guardar
results_df = pd.DataFrame(results)
results_df.to_csv('prueba_t_student_30runs_resultados.csv', index=False)

# Resumen por métrica: métodos significativamente diferentes del baseline
print("\n\n" + "=" * 90)
print("RESUMEN: MÉTODOS CON DIFERENCIA SIGNIFICATIVA RESPECTO AL BASELINE")
print("=" * 90)
for metric in metrics:
    print(f"\nMétrica: {metric.upper()}")
    subset = results_df[results_df['metric'] == metric]
    
    sig_05 = subset[subset['p_value'] < alpha]
    sig_bonf = subset[subset['p_value'] < alpha_bonferroni]
    
    print(f"  α = 0.05: {len(sig_05)} métodos significativamente diferentes")
    for _, row in sig_05.iterrows():
        direction = "superior" if row['diff_mean'] > 0 else "inferior"
        print(f"    - {row['method']}: diff={row['diff_mean']:+.4f}, t={row['t_statistic']:.3f}, p={row['p_value']:.6f} ({direction})")
    
    if len(sig_05) == 0:
        print("    Ningún método difiere significativamente del baseline.")
    
    print(f"  α Bonferroni = {alpha_bonferroni:.6f}: {len(sig_bonf)} métodos significativamente diferentes")
    for _, row in sig_bonf.iterrows():
        direction = "superior" if row['diff_mean'] > 0 else "inferior"
        print(f"    - {row['method']}: diff={row['diff_mean']:+.4f}, t={row['t_statistic']:.3f}, p={row['p_value']:.6f} ({direction})")

# Comparaciones adicionales de interés: l1 vs rfecv y l1 vs baseline (ya incluida)
print("\n\n" + "=" * 90)
print("COMPARACIONES ADICIONALES DE INTERÉS (pareadas, α = 0.05)")
print("=" * 90)

comparisons = [
    ('l1', 'rfecv'),
    ('l1', 'rfe'),
    ('rfecv', 'rfe'),
    ('l1', 'baseline'),
]

for m1, m2 in comparisons:
    print(f"\n{m1} vs {m2}:")
    vals1 = df[df['method'] == m1].sort_values('run_id')
    vals2 = df[df['method'] == m2].sort_values('run_id')
    assert np.array_equal(vals1['run_id'].values, vals2['run_id'].values)
    
    for metric in ['accuracy', 'f1']:
        v1 = vals1[metric].values
        v2 = vals2[metric].values
        t, p = stats.ttest_rel(v1, v2)
        d = (v1 - v2).mean()
        print(f"  {metric:>10}: diff={d:+.4f}, t={t:>7.3f}, p={p:>10.6f}, {'SÍ' if p < alpha else 'NO'} significativo")

print("\n" + "=" * 90)
print("RESULTADOS GUARDADOS EN: prueba_t_student_30runs_resultados.csv")
print("=" * 90)
