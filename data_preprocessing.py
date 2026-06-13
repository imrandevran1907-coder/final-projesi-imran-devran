import pandas as pd
import numpy as np

print("Çift veri kaynağı işleme adımı başladı...")

# 1. EN AZ 2 VERI KAYNAĞINI OKUMA (Yönerge Şartı)
df21 = pd.read_csv("players_21.csv", low_memory=False)
df22 = pd.read_csv("players_22.csv", low_memory=False)

# İki farklı sezon verisini tek bir dev veri havuzunda birleştirme
df = pd.concat([df21, df22], ignore_index=True)
print(f"Toplam birleştirilmiş veri satır sayısı: {len(df)}")

# 2. Gereksiz sütunları eleme
columns_to_keep = ['age', 'value_eur', 'wage_eur', 'pace', 'shooting', 
                    'passing', 'dribbling', 'defending', 'physic', 'overall']
df_cleaned = df[columns_to_keep].copy()

# 3. Eksik (Null) verileri medyan ile doldurma
for col in df_cleaned.columns:
    if df_cleaned[col].isnull().sum() > 0:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# 4. Temizlenmiş veriyi kaydetme
df_cleaned.to_csv("cleaned_football_players.csv", index=False)
print("Çift kaynaklı veri başarıyla temizlendi ve 'cleaned_football_players.csv' olarak kaydedildi!")