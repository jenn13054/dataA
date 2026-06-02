import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Cargar datos
print("=" * 70)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 70)
df = pd.read_csv('dataA.csv')
print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas\n")

# ============================================================
# 1. INFORMACIÓN GENERAL Y TIPOS DE DATOS
# ============================================================
print("-" * 70)
print("1. INFORMACIÓN GENERAL Y TIPOS DE DATOS")
print("-" * 70)
print(df.info())
print("\n")

# ============================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================
print("-" * 70)
print("2. ESTADÍSTICAS DESCRIPTIVAS (Variables Numéricas)")
print("-" * 70)
desc = df.describe().T
desc['missing'] = df.isnull().sum()
desc['missing_%'] = (df.isnull().sum() / len(df)) * 100
print(desc.round(3).to_string())
print("\n")

# ============================================================
# 3. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# ============================================================
print("-" * 70)
print("3. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (Target)")
print("-" * 70)
target_counts = df['Target'].value_counts().sort_index()
print(target_counts.to_string())
print(f"\nProporciones:")
print((target_counts / len(df) * 100).round(2).to_string())
print("\n")

# ============================================================
# 4. ANÁLISIS DE VALORES FALTANTES
# ============================================================
print("-" * 70)
print("4. VALORES FALTANTES POR COLUMNA")
print("-" * 70)
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing.to_string())
else:
    print("No hay valores faltantes en el dataset.")
print("\n")

# ============================================================
# 5. SELECCIÓN DE VARIABLES NUMÉRICAS PARA ANÁLISIS
# ============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Variables numéricas identificadas: {len(numeric_cols)}")
print(numeric_cols)
print("\n")

# Crear carpeta de salida
os.makedirs('graficas', exist_ok=True)

# ============================================================
# GRÁFICA 1: Distribución de la Variable Objetivo
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
colors_target = ['#2ecc71', '#f39c12', '#e74c3c']
target_counts.plot(kind='bar', color=colors_target, ax=ax, edgecolor='black')
ax.set_title('Distribución de la Variable Objetivo (Target)', fontsize=14, fontweight='bold')
ax.set_xlabel('Target', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.set_xticklabels(['Graduado (0)', 'Desertó (1)', 'Matriculado (2)'], rotation=0)
for i, v in enumerate(target_counts.values):
    ax.text(i, v + 20, str(v), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/01_distribucion_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 1: Distribución de Target guardada.")

# ============================================================
# GRÁFICA 2: Histogramas de Variables Numéricas Clave
# ============================================================
key_numeric = ['Age at enrollment', 'Previous qualification (grade)', 
               'Admission grade', 'Curricular units 1st sem (grade)',
               'Curricular units 2nd sem (grade)', 'Unemployment rate', 
               'Inflation rate', 'GDP']

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
for idx, col in enumerate(key_numeric):
    if col in df.columns:
        axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribución: {col}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frecuencia')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df[col].mean():.2f}')
        axes[idx].legend()
plt.suptitle('Histogramas de Variables Numéricas Principales', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('graficas/02_histogramas_variables.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 2: Histogramas guardados.")

# ============================================================
# GRÁFICA 3: Boxplots por Target (variables clave)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
box_vars = ['Age at enrollment', 'Admission grade', 
            'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']
for idx, col in enumerate(box_vars):
    ax = axes[idx // 2, idx % 2]
    sns.boxplot(data=df, x='Target', y=col, ax=ax, palette=colors_target)
    ax.set_title(f'{col} por Target', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target')
plt.suptitle('Boxplots de Variables Clave por Target', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('graficas/03_boxplots_por_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 3: Boxplots por Target guardados.")

# ============================================================
# GRÁFICA 4: Mapa de Calor de Correlación
# ============================================================
print("-" * 70)
print("5. ANÁLISIS DE CORRELACIÓN")
print("-" * 70)

# Calcular matriz de correlación
corr_matrix = df[numeric_cols].corr()

# Guardar correlaciones con Target
print("\nCorrelaciones con Target (ordenadas por magnitud):")
target_corr = corr_matrix['Target'].drop('Target').sort_values(key=abs, ascending=False)
print(target_corr.round(4).to_string())
print("\n")

# Mapa de calor completo
fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matriz de Correlación (Variables Numéricas)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/04_matriz_correlacion_completa.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 4: Matriz de correlación completa guardada.")

# ============================================================
# GRÁFICA 5: Correlaciones con Target (barras)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))
colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in target_corr.values]
target_corr.plot(kind='barh', color=colors_bar, ax=ax, edgecolor='black')
ax.set_title('Correlación de Variables con Target', fontsize=14, fontweight='bold')
ax.set_xlabel('Coeficiente de Correlación de Pearson', fontsize=12)
ax.axvline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('graficas/05_correlacion_con_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 5: Correlación con Target guardada.")

# ============================================================
# GRÁFICA 6: Top 15 correlaciones más fuertes (entre variables)
# ============================================================
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        var1 = corr_matrix.columns[i]
        var2 = corr_matrix.columns[j]
        corr_val = corr_matrix.iloc[i, j]
        corr_pairs.append((var1, var2, corr_val))

corr_pairs_df = pd.DataFrame(corr_pairs, columns=['Variable 1', 'Variable 2', 'Correlación'])
corr_pairs_df['Abs_Correlación'] = corr_pairs_df['Correlación'].abs()
top_corr = corr_pairs_df.nlargest(15, 'Abs_Correlación')

print("Top 15 correlaciones más fuertes entre variables:")
print(top_corr[['Variable 1', 'Variable 2', 'Correlación']].to_string(index=False))
print("\n")

fig, ax = plt.subplots(figsize=(12, 10))
y_pos = range(len(top_corr))
colors_top = ['#e74c3c' if v < 0 else '#3498db' for v in top_corr['Correlación'].values]
labels = [f"{row['Variable 1'][:25]}\nvs\n{row['Variable 2'][:25]}" for _, row in top_corr.iterrows()]
ax.barh(y_pos, top_corr['Correlación'].values, color=colors_top, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_title('Top 15 Correlaciones más Fuertes entre Variables', fontsize=14, fontweight='bold')
ax.set_xlabel('Coeficiente de Correlación de Pearson', fontsize=12)
ax.axvline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('graficas/06_top_correlaciones.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 6: Top correlaciones guardada.")

# ============================================================
# GRÁFICA 7: Pairplot de variables más correlacionadas con Target
# ============================================================
top_vars = target_corr.head(6).index.tolist() + ['Target']
pairplot_data = df[top_vars].copy()

# Muestra para pairplot (máx 1500 puntos para no saturar)
if len(pairplot_data) > 1500:
    pairplot_data = pairplot_data.sample(1500, random_state=42)

g = sns.pairplot(pairplot_data, hue='Target', palette=colors_target, diag_kind='kde', 
                 plot_kws={'alpha': 0.5, 's': 20})
g.fig.suptitle('Pairplot: Variables más Correlacionadas con Target', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('graficas/07_pairplot_correlaciones.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 7: Pairplot guardado.")

# ============================================================
# GRÁFICA 8: Distribución de Edad por Target
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for t, color, label in zip([0, 1, 2], colors_target, ['Graduado', 'Desertó', 'Matriculado']):
    subset = df[df['Target'] == t]['Age at enrollment'].dropna()
    ax.hist(subset, bins=25, alpha=0.6, label=label, color=color, edgecolor='black')
ax.set_title('Distribución de Edad de Inscripción por Target', fontsize=14, fontweight='bold')
ax.set_xlabel('Edad al Inscribirse', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('graficas/08_edad_por_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 8: Distribución de edad guardada.")

# ============================================================
# GRÁFICA 9: Notas por semestre por Target
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, col in enumerate(['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']):
    for t, color, label in zip([0, 1, 2], colors_target, ['Graduado', 'Desertó', 'Matriculado']):
        subset = df[df['Target'] == t][col].dropna()
        axes[idx].hist(subset, bins=25, alpha=0.6, label=label, color=color, edgecolor='black')
    axes[idx].set_title(f'{col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Calificación')
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].legend()
plt.suptitle('Distribución de Calificaciones por Semestre y Target', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/09_calificaciones_por_semestre.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 9: Calificaciones por semestre guardada.")

# ============================================================
# GRÁFICA 10: Indicadores económicos
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, col in enumerate(['Unemployment rate', 'Inflation rate', 'GDP']):
    df[col].hist(bins=30, ax=axes[idx], color='teal', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribución: {col}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df[col].mean():.2f}')
    axes[idx].legend()
plt.suptitle('Indicadores Económicos', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('graficas/10_indicadores_economicos.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Gráfica 10: Indicadores económicos guardada.")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("=" * 70)
print("RESUMEN DEL ANÁLISIS")
print("=" * 70)
print(f"Total de registros: {len(df)}")
print(f"Total de variables: {len(df.columns)}")
print(f"Variables numéricas: {len(numeric_cols)}")
print(f"Variables categóricas: {len(df.columns) - len(numeric_cols)}")
print(f"\nDistribución de Target:")
for t, label in zip([0, 1, 2], ['Graduado', 'Desertó', 'Matriculado']):
    pct = (df['Target'] == t).mean() * 100
    print(f"  {label} ({t}): {target_counts.get(t, 0)} ({pct:.2f}%)")

print(f"\nVariables más correlacionadas con Target:")
for var, corr in target_corr.head(5).items():
    print(f"  {var}: {corr:.4f}")

print(f"\nTodas las gráficas se guardaron en la carpeta: ./graficas/")
print("=" * 70)
