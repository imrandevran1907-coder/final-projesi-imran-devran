import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

print("1. Temizlenmiş veri seti yükleniyor...")
# Bir önceki adımda oluşturduğumuz temiz veriyi okuyoruz
df = pd.read_csv('cleaned_football_players.csv')

# Tahmin etmek istediğimiz hedef değişken (Y): Oyuncunun Gücü (overall)
# Yapay zekaya tüyo olmasın diye overall sütununu girdilerden (X) çıkarıyoruz
X = df.drop(columns=['overall'])
y = df['overall']

# Veriyi %80 Eğitim, %20 Test olarak ikiye bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1907)

print(f"Eğitim verisi: {X_train.shape[0]} oyuncu, Test verisi: {X_test.shape[0]} oyuncu.")
print("-" * 50)

# ==========================================
# MODEL 1: RANDOM FOREST EĞİTİMİ
# ==========================================
print("2. Random Forest modeli eğitiliyor (Bu işlem 10-15 saniye sürebilir)...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=1907, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Test verisiyle tahmin yapalım
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print(f"-> Random Forest Ortalama Hata (MAE): {rf_mae:.2f} (Skor bazında ne kadar saptığı)")
print(f"-> Random Forest Başarı Oranı (R2 Score): %{rf_r2*100:.2f}")
print("-" * 50)

# ==========================================
# MODEL 2: XGBOOST EĞİTİMİ
# ==========================================
print("3. XGBoost modeli eğitiliyor...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=1907, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# Test verisiyle tahmin yapalım
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_r2 = r2_score(y_test, xgb_preds)

print(f"-> XGBoost Ortalama Hata (MAE): {xgb_mae:.2f}")
print(f"-> XGBoost Başarı Oranı (R2 Score): %{xgb_r2*100:.2f}")
print("-" * 50)

# ==========================================
# EN İYİ MODELİ KAYDETME
# ==========================================
# Hangisinin R2 skoru daha yüksekse (başarılıysa) onu kaydediyoruz
if xgb_r2 > rf_r2:
    en_iyi_model = xgb_model
    model_adi = "XGBoost"
else:
    en_iyi_model = rf_model
    model_adi = "Random Forest"

print(f"Sonuç: En yüksek başarıyı {model_adi} verdi!")
print(f"Model 'best_football_model.pkl' adıyla klasöre kaydediliyor...")

# Modeli bir dosya olarak bilgisayara gömüyoruz (Streamlit arayüzünde direkt çağıracağız)
with open('best_football_model.pkl', 'wb') as f:
    pickle.dump(en_iyi_model, f)

print("2. AŞAMA BAŞARIYLA TAMAMLANDI! Model eğitildi ve kaydedildi.")
