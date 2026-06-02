import pandas as pd

# Cargar datos originales
df = pd.read_csv('dataA.csv')

print("=" * 60)
print("TRANSFORMACIÓN DE TARGET A BINARIO")
print("=" * 60)

print("\nDistribución ORIGINAL de Target:")
print(df['Target'].value_counts().sort_index().to_string())
print(f"  0 = Graduado")
print(f"  1 = Desertó")
print(f"  2 = Matriculado")

# Transformación: Retención (0, 2) -> 1, Deserción (1) -> 0
df['Target_Binario'] = df['Target'].apply(lambda x: 1 if x in [0, 2] else 0)

# Renombrar para claridad: 1 = Retenido, 0 = Desertó
# Opcionalmente podemos reemplazar la columna original
df['Target_Original'] = df['Target']
df['Target'] = df['Target_Binario']
df.drop(columns=['Target_Binario'], inplace=True)

print("\n" + "-" * 60)
print("Distribución NUEVA de Target (Binario):")
print("-" * 60)
new_counts = df['Target'].value_counts().sort_index()
print(new_counts.to_string())
print(f"\n  1 = Retenido (Graduado o Matriculado): {new_counts.get(1, 0)} ({new_counts.get(1,0)/len(df)*100:.2f}%)")
print(f"  0 = Desertó: {new_counts.get(0, 0)} ({new_counts.get(0,0)/len(df)*100:.2f}%)")

# Guardar nuevo dataset
df.to_csv('dataA_binario.csv', index=False)
print("\n[✓] Dataset binario guardado como: dataA_binario.csv")
print(f"[✓] Total de registros: {len(df)}")
print(f"[✓] Total de columnas: {len(df.columns)}")
