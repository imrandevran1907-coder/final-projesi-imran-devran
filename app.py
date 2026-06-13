import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Sürpriz hata grafiklerini engellemek için matplotlib ayarı
plt.style.use('default')

# Sayfa başlığı ve tasarımı
st.set_page_config(page_title="Futbolcu Performans Analizi", layout="wide")

st.title("⚽ Futbolcu Performans Tahmin ve Açıklanabilir Yapay Zekâ (XAI) Sistemi")
st.markdown("Bu sistem, girilen futbolcu özelliklerine göre oyuncunun **Genel Değerlendirme (Overall)** skorunu tahmin eder ve modelin bu kararı nasıl verdiğini açıklar.")

# 1. MODELİ VE VERİ SÜTUNLARINI YÜKLE
@st.cache_resource
def load_assets():
    with open('best_football_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Sütun isimlerini öğrenmek için temiz veriden bir satır okuyoruz
    df_sample = pd.read_csv('cleaned_football_players.csv', nrows=1)
    features = df_sample.drop(columns=['overall']).columns.tolist()
    return model, features

try:
    model, feature_cols = load_assets()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu! Lütfen önce train_model.py kodunu çalıştırın. Hata: {e}")
    st.stop()

# 2. KULLANICI GİRDİLERİ (SOL PANEL)
st.sidebar.header("🏃‍♂️ Oyuncu Özelliklerini Girin")

age = st.sidebar.slider("Yaş (Age)", 16, 45, 25)
height = st.sidebar.slider("Boy (cm)", 150, 210, 180)
weight = st.sidebar.slider("Kilo (kg)", 50, 110, 75)
value = st.sidebar.number_input("Piyasa Değeri (EUR)", min_value=0, value=5000000, step=50000)
wage = st.sidebar.number_input("Haftalık Maaş (EUR)", min_value=0, value=20000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Yetenek Skorları (0-100)")
pace = st.sidebar.slider("Hız (Pace)", 30, 100, 70)
shooting = st.sidebar.slider("Şut (Shooting)", 30, 100, 60)
passing = st.sidebar.slider("Pas (Passing)", 30, 100, 65)
dribbling = st.sidebar.slider("Top Sürme (Dribbling)", 30, 100, 68)
defending = st.sidebar.slider("Defans (Defending)", 30, 100, 50)
physic = st.sidebar.slider("Fizik (Physic)", 30, 100, 70)

# Tüm girdileri bir sözlükte topluyoruz
input_data = {
    'age': age, 'height_cm': height, 'weight_kg': weight,
    'value_eur': value, 'wage_eur': wage, 'pace': pace,
    'shooting': shooting, 'passing': passing, 'dribbling': dribbling,
    'defending': defending, 'physic': physic
}

# Eksik kalan one-hot pozisyon sütunlarını 0 (False) olarak otomatik tamamlıyoruz
for col in feature_cols:
    if col not in input_data:
        input_data[col] = 0

# Veriyi modelin okuyacağı DataFrame formatına sokuyoruz (Sütun sıralaması eğitimdekiyle birebir aynı olmalı)
input_df = pd.DataFrame([input_data])[feature_cols]

# 3. TAHMİN VE GÖRSELLEŞTİRME (SAĞ PANEL)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Tahmin Sonucu")
    if st.button("Performans Skorunu Hesapla", type="primary"):
        prediction = model.predict(input_df)[0]
        st.metric(label="Tahmini Overall Skoru", value=f"{prediction:.1f}")
        
        # Oyuncu kalitesine göre renkli bilgilendirme
        if prediction >= 85:
            st.success("🌟 Dünya Klasında Bir Oyuncu!")
        elif prediction >= 75:
            st.info("🔥 Kaliteli Bir Lig Oyuncusu")
        else:
            st.warning("🏃 Gelişmekte Olan / Ortalama Oyuncu")

with col2:
    st.subheader("🧠 Açıklanabilir Yapay Zekâ (SHAP Analizi)")
    st.write("Aşağıdaki grafik, girdiğiniz özelliklerin tahmini skoru nasıl etkilediğini gösterir. **Kırmızı** çubuklar skoru artıran, **Mavi** çubuklar ise azaltan özellikleri temsil eder.")
    
    # SHAP TreeExplainer hesaplaması
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)
        
        # SHAP Waterfall plot çizdiriyoruz
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.write("Grafik oluşturulurken bir hata oluştu veya bu model türü SHAP şelale grafiğini doğrudan desteklemiyor.")
        st.info("Alternatif olarak modelin karar mekanizması hazır durumdadır.")