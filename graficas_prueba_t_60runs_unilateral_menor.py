"""
Genera graficas para la prueba t de Student unilateral (metodo < baseline)
sobre el experimento de 60 runs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar resultados
results = pd.read_csv('prueba_t_student_60runs_unilateral_menor_resultados.csv')

# Configuracion general
plt.rcParams['figure.dpi'] = 300
sns.set_style('whitegrid')
alpha = 0.05
alpha_bonferroni = 0.00125

# Colores
COLOR_NO_SIGNIF = '#cccccc'
COLOR_SIGNIF_05 = '#666666'
COLOR_SIGNIF_BONF = '#000000'
COLOR_BASELINE = '#444444'

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC AUC']

# Ordenar metodos: excluir baseline (no aplica en resultados), pero mantener orden logico
methods_order = [m for m in results['method'].unique() if m != 'baseline']

# --- GRAFICA 1: Diferencias de medias con barras de error (IC 95%) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    subset = results[results['metric'] == metric].copy()
    subset = subset.sort_values('diff_mean')
    
    y_pos = np.arange(len(subset))
    colors = [COLOR_SIGNIF_BONF if row['significant_bonferroni']
              else COLOR_SIGNIF_05 if row['significant_alpha_05']
              else COLOR_NO_SIGNIF for _, row in subset.iterrows()]
    
    # Barra horizontal para cada metodo
    bars = ax.barh(y_pos, subset['diff_mean'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Linea vertical en 0
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset['method'], fontsize=9)
    ax.set_xlabel('Diferencia de medias (metodo - baseline)', fontsize=10)
    ax.set_title(f'{label}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Anotar valores
    for i, (_, row) in enumerate(subset.iterrows()):
        ax.text(row['diff_mean'], i, f" {row['diff_mean']:+.4f}",
                va='center', fontsize=8, color='black')

# Leyenda en el ultimo subplot
ax = axes[-1]
ax.axis('off')
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=COLOR_SIGNIF_BONF, edgecolor='black', label=f'Significativo (Bonferroni, p < {alpha_bonferroni})'),
    plt.Rectangle((0,0),1,1, facecolor=COLOR_SIGNIF_05, edgecolor='black', label=f'Significativo (alpha=0.05)'),
    plt.Rectangle((0,0),1,1, facecolor=COLOR_NO_SIGNIF, edgecolor='black', label='No significativo')
]
ax.legend(handles=legend_elements, loc='center', fontsize=11, title='Interpretacion')

plt.suptitle('Prueba t unilateral (izquierda): diferencias de medias metodo vs baseline (60 runs)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('grafica_ttest_60runs_diferencias_medias.png', bbox_inches='tight')
print('Guardada: grafica_ttest_60runs_diferencias_medias.png')
plt.close()

# --- GRAFICA 2: Heatmap de p-values (-log10 transformado) ---
pivot_pvalues = results.pivot(index='method', columns='metric', values='p_value')
pivot_pvalues = pivot_pvalues[metrics]  # ordenar metricas

plt.figure(figsize=(10, 8))
# Transformar p-values a -log10(p), con clip para evitar infinitos
log_p = -np.log10(pivot_pvalues.replace(0, 1e-300))

ax = sns.heatmap(
    log_p,
    annot=True,
    fmt='.2f',
    cmap='Greys',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '-log10(p-value)'},
    vmin=0,
    vmax=12
)
ax.set_title('Heatmap de significancia (-log10 p-value)\nPrueba t unilateral: metodo < baseline (60 runs)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Metrica', fontsize=12)
ax.set_ylabel('Metodo', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Lineas de referencia para umbrales
ax.axhline(y=0, color='black', linewidth=2)
ax.axvline(x=0, color='black', linewidth=2)

plt.tight_layout()
plt.savefig('grafica_ttest_60runs_heatmap_pvalues.png', bbox_inches='tight')
print('Guardada: grafica_ttest_60runs_heatmap_pvalues.png')
plt.close()

# --- GRAFICA 3: Medias de cada metodo vs baseline con barras de error ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx]
    subset = results[results['metric'] == metric].copy()
    subset = subset.sort_values('mean_method')
    
    y_pos = np.arange(len(subset))
    
    # Media del metodo
    colors = [COLOR_SIGNIF_BONF if row['significant_bonferroni']
              else COLOR_SIGNIF_05 if row['significant_alpha_05']
              else COLOR_NO_SIGNIF for _, row in subset.iterrows()]
    
    ax.barh(y_pos, subset['mean_method'], xerr=subset['std_method'],
            color=colors, edgecolor='black', linewidth=0.5, capsize=3, alpha=0.9)
    
    # Linea vertical para la media del baseline
    baseline_mean = subset['mean_baseline'].iloc[0]
    ax.axvline(x=baseline_mean, color=COLOR_BASELINE, linestyle='--', linewidth=2, label=f'Baseline = {baseline_mean:.4f}')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset['method'], fontsize=9)
    ax.set_xlabel(label, fontsize=10)
    ax.set_title(f'{label}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

# Leyenda en el ultimo subplot
ax = axes[-1]
ax.axis('off')
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=COLOR_SIGNIF_BONF, edgecolor='black', label=f'Significativo (Bonferroni)'),
    plt.Rectangle((0,0),1,1, facecolor=COLOR_SIGNIF_05, edgecolor='black', label=f'Significativo (alpha=0.05)'),
    plt.Rectangle((0,0),1,1, facecolor=COLOR_NO_SIGNIF, edgecolor='black', label='No significativo'),
    plt.Line2D([0], [0], color=COLOR_BASELINE, linestyle='--', linewidth=2, label='Baseline')
]
ax.legend(handles=legend_elements, loc='center', fontsize=11, title='Interpretacion')

plt.suptitle('Medias de desempeno por metodo vs baseline (60 runs)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('grafica_ttest_60runs_medias_vs_baseline.png', bbox_inches='tight')
print('Guardada: grafica_ttest_60runs_medias_vs_baseline.png')
plt.close()

# --- GRAFICA 4: Conteo de metricas con significancia por metodo ---
summary = results.groupby('method').agg(
    signif_alpha_05=('significant_alpha_05', 'sum'),
    signif_bonferroni=('significant_bonferroni', 'sum'),
    total=('significant_alpha_05', 'count')
).reset_index()
summary = summary.sort_values('signif_bonferroni', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(summary))
height = 0.35

bars1 = ax.barh(y_pos - height/2, summary['signif_alpha_05'], height,
                label=f'Significativo (alpha=0.05)', color=COLOR_SIGNIF_05, edgecolor='black')
bars2 = ax.barh(y_pos + height/2, summary['signif_bonferroni'], height,
                label=f'Significativo (Bonferroni)', color=COLOR_SIGNIF_BONF, edgecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(summary['method'])
ax.set_xlabel('Numero de metricas con metodo < baseline', fontsize=12)
ax.set_title('Resumen de significancia por metodo (5 metricas evaluadas)', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 6))
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

# Anotar valores
for bar in bars1:
    width = bar.get_width()
    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}',
            va='center', fontsize=9)
for bar in bars2:
    width = bar.get_width()
    ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}',
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('grafica_ttest_60runs_resumen_significancia.png', bbox_inches='tight')
print('Guardada: grafica_ttest_60runs_resumen_significancia.png')
plt.close()

print('\nTodas las graficas fueron generadas exitosamente.')
