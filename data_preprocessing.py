import pandas as pd
import numpy as np
import os

# Yükleyeceğimiz dosya adı
dosya_adi = 'players_22.csv' 

if not os.path.exists(dosya_adi):
    print(f"HATA: Klasörde '{dosya_adi}' adında bir dosya bulunamadı!")
    exit()

print("Veri seti yükleniyor, lütfen bekleyin...")
df = pd.read_csv(dosya_adi, low_memory=False)
print("Veri seti başarıyla yüklendi! Toplam oyuncu sayısı:", df.shape[0])

# Projemiz için en önemli temel özellikleri seçiyoruz
ana_ozellikler = ['overall', 'age', 'height_cm', 'weight_kg', 'value_eur', 'wage_eur', 
                  'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'club_position']

# Sadece bu sütunlar veri setinde varsa filtreleyelim
mevcut_sutunlar = [col for col in ana_ozellikler if col in df.columns]
df = df[mevcut_sutunlar]

# Boş (Null) olan yerleri sütunların ortalamasıyla dolduruyoruz
sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns
df[sayisal_sutunlar] = df[sayisal_sutunlar].fillna(df[sayisal_sutunlar].mean())

# Kulüp pozisyonu boş olanları 'SUB' (Yedek) yapalım
if 'club_position' in df.columns:
    df['club_position'] = df['club_position'].fillna('SUB')
    # Pozisyonları sayısallaştırıyoruz (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['club_position'], drop_first=True)

# Temizlenmiş veriyi kaydediyoruz
df.to_csv('cleaned_football_players.csv', index=False)
print("1. AŞAMA TAMAMLANDI! 'cleaned_football_players.csv' başarıyla oluşturuldu.")
