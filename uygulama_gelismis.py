import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# =========================================================
# 1. SAYFA AYARLARI
# =========================================================

st.set_page_config(
    page_title="Fener Analytics | Futbolcu Performans Analizi",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Matplotlib koyu tema uyumluluğu
plt.style.use("dark_background")


# =========================================================
# 2. FENERBAHÇE TEMALI CSS TASARIMI
# =========================================================

st.markdown(
    """
    <style>
        /* Ana sayfa arka plan */
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255, 237, 0, 0.12), transparent 35%),
                linear-gradient(135deg, #051024 0%, #0a1c3a 50%, #030a16 100%);
        }

        /* Sayfa üst boşluğunu azalt */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        /* Sol menü özelleştirme */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #030b1a 0%, #071935 100%);
            border-right: 3px solid #ffed00;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #ffffff !important;
        }

        /* Ana başlık hero alanı */
        .hero-container {
            background: linear-gradient(120deg, rgba(6, 23, 50, 0.95), rgba(10, 37, 79, 0.90));
            border: 1px solid rgba(255, 237, 0, 0.4);
            border-left: 8px solid #ffed00;
            border-radius: 14px;
            padding: 20px 25px;
            margin-bottom: 25px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        }

        .hero-title {
            color: #ffed00 !important;
            font-size: 2.4rem;
            font-weight: 900;
            margin: 0;
            letter-spacing: 0.5px;
        }

        .hero-subtitle {
            color: #ffffff !important;
            font-size: 1.1rem;
            margin-top: 6px;
            margin-bottom: 0;
            opacity: 0.9;
        }

        /* Sağ paneldeki bilgi kartları */
        .info-box {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .info-box-title {
            font-size: 0.85rem;
            color: #a0aec0;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        .info-box-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
        }

        /* Sonuç kartı (Büyük Skorbord) */
        .prediction-card {
            background: linear-gradient(135deg, #ffed00 0%, #d4c100 100%);
            color: #051024 !important;
            border-radius: 16px;
            text-align: center;
            padding: 25px 15px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
            border: 2px solid #ffffff;
        }

        .prediction-label {
            font-size: 0.95rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #051024;
        }

        .prediction-number {
            font-size: 4.8rem;
            line-height: 1;
            font-weight: 900;
            margin: 10px 0;
            color: #051024;
        }

        .prediction-comment {
            font-size: 1.15rem;
            font-weight: 800;
            color: #051024;
        }

        /* Streamlit Form ve Buton revizyonları */
        .stButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            background: linear-gradient(90deg, #ffed00, #d4c100) !important;
            color: #051024 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.7rem 1rem !important;
            font-weight: 800 !important;
            font-size: 1rem !important;
            box-shadow: 0 5px 15px rgba(255, 237, 0, 0.15) !important;
            transition: all 0.2s ease-in-out !important;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #ffffff !important;
            color: #051024 !important;
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(255, 255, 255, 0.2) !important;
        }

        /* Sekmeler */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 6px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: #a0aec0 !important;
            padding: 8px 20px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ffed00 !important;
            color: #051024 !important;
            font-weight: 800 !important;
        }

        /* İçerik metin renkleri düzeltme */
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. DOSYA YOLLARI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_football_model.pkl"
DATA_PATH = BASE_DIR / "cleaned_football_players.csv"


# =========================================================
# 4. MODEL VE SÜTUNLARI YÜKLEME
# =========================================================

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH.name} dosyası bulunamadı.")

    with open(MODEL_PATH, "rb") as model_file:
        loaded_model = pickle.load(model_file)

    if hasattr(loaded_model, "feature_names_in_"):
        loaded_features = list(loaded_model.feature_names_in_)
    else:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"{DATA_PATH.name} dosyası bulunamadı.")

        sample_df = pd.read_csv(DATA_PATH, nrows=1)
        unnamed_columns = [col for col in sample_df.columns if col.lower().startswith("unnamed")]
        sample_df = sample_df.drop(columns=unnamed_columns, errors="ignore")

        if "overall" not in sample_df.columns:
            raise ValueError("CSV dosyasında 'overall' hedef sütunu bulunamadı.")

        loaded_features = sample_df.drop(columns=["overall"]).columns.tolist()

    return loaded_model, loaded_features


@st.cache_resource
def create_explainer(_model):
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return shap.Explainer(_model)


try:
    model, feature_cols = load_assets()
    explainer = create_explainer(model)
except Exception as error:
    st.error("Model veya veri dosyaları yüklenemedi.")
    st.code(str(error))
    st.info("Dosyaların ana dizinde (app.py ile aynı yerde) olduğunu kontrol edin.")
    st.stop()


# =========================================================
# 5. YARDIMCI SÖZLÜKLER VE FONKSİYONLAR
# =========================================================

POSITION_CODES = ["GK", "CB", "LB", "RB", "LWB", "RWB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "CF", "ST"]

POSITION_NAMES = {
    "GK": "Kaleci", "CB": "Stoper", "LB": "Sol Bek", "RB": "Sağ Bek",
    "LWB": "Sol Kanat Bek", "RWB": "Sağ Kanat Bek", "CDM": "Defansif Orta Saha",
    "CM": "Merkez Orta Saha", "CAM": "Ofansif Orta Saha", "LM": "Sol Orta Saha",
    "RM": "Sağ Orta Saha", "LW": "Sol Kanat", "RW": "Sağ Kanat",
    "CF": "İkinci Forvet", "ST": "Santrafor"
}

FEATURE_TRANSLATIONS = {
    "age": "Yaş",
    "height_cm": "Boy (cm)",
    "weight_kg": "Kilo (kg)",
    "value_eur": "Piyasa Değeri (€)",
    "wage_eur": "Haftalık Maaş (€)",
    "pace": "Hız (Pace)",
    "shooting": "Şut (Shooting)",
    "passing": "Pas (Passing)",
    "dribbling": "Top Sürme (Dribbling)",
    "defending": "Defans (Defending)",
    "physic": "Fizik (Physic)"
}

for code, name in POSITION_NAMES.items():
    FEATURE_TRANSLATIONS[code] = f"Mevki: {name}"
    FEATURE_TRANSLATIONS[f"position_{code}"] = f"Mevki: {name}"
    FEATURE_TRANSLATIONS[f"pos_{code}"] = f"Mevki: {name}"
    FEATURE_TRANSLATIONS[f"player_positions_{code}"] = f"Mevki: {name}"

def find_position_columns(columns):
    position_mapping = {}
    for position in POSITION_CODES:
        position_upper = position.upper()
        for column in columns:
            normalized_column = str(column).upper().replace("-", "_").replace(" ", "_")
            accepted_names = {
                position_upper, f"POSITION_{position_upper}", f"POSITIONS_{position_upper}",
                f"PLAYER_POSITION_{position_upper}", f"PLAYER_POSITIONS_{position_upper}", f"POS_{position_upper}"
            }
            if normalized_column in accepted_names:
                position_mapping[position] = column
                break
    return position_mapping

# 🌟 GÜNCELLEME: Çeviri hatası üreten "Bin" (Milyar/Bin karışıklığı) kelimesi kaldırıldı.
# Doğrudan Türkçe "Milyon €" veya tam sayı formatı ("20.000 €") getirildi.
def format_currency(val):
    val = float(val)
    if val >= 1_000_000_000: return f"{val / 1_000_000_000:.1f} Milyar €"
    if val >= 1_000_000: return f"{val / 1_000_000:.1f} Milyon €"
    return f"{val:,.0f} €".replace(",", ".")

def get_player_level(prediction):
    if prediction >= 90: return "👑 Efsane Çubuklu Seviyesi"
    if prediction >= 85: return "🌟 Dünya Klasında Kadro Lideri"
    if prediction >= 80: return "🔥 Üst Düzey Yıldız"
    if prediction >= 75: return "💪 Kaliteli Lig Oyuncusu"
    if prediction >= 70: return "📈 Rotasyon / Gelecek Vaadeden"
    if prediction >= 65: return "🏃 Ortalama Alternatif"
    return "🌱 Geliştirilmesi Gereken Genç Yetenek"

def create_radar_chart(player_values):
    categories = ["Hız", "Şut", "Pas", "Top Sürme", "Defans", "Fizik"]
    values = [
        player_values["pace"], player_values["shooting"], player_values["passing"],
        player_values["dribbling"], player_values["defending"], player_values["physic"]
    ]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"polar": True})
    fig.patch.set_alpha(0)
    
    ax.set_facecolor((7/255, 21/255, 47/255, 0.6))

    ax.plot(angles, values, linewidth=2.5, color="#ffed00", linestyle="solid")
    ax.fill(angles, values, color="#ffed00", alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="white", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#a0aec0", fontsize=8)
    
    ax.grid(color="#ffffff", alpha=0.15, linestyle="--")
    ax.spines["polar"].set_color("#ffed00")
    ax.spines["polar"].set_alpha(0.7)
    
    fig.tight_layout()
    return fig

position_columns = find_position_columns(feature_cols)
available_positions = list(position_columns.keys())


# =========================================================
# 6. ÜST BAŞLIK (HERO SECTION)
# =========================================================

st.markdown(
    """
    <div class="hero-container">
        <p class="hero-title">⚽ Fener Analytics Scout Panel</p>
        <p class="hero-subtitle">Yapay Zekâ Destekli Futbolcu Performans Tahmini ve Açıklanabilir XAI Sistemi</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 7. SOL MENÜ – FORM YAPISI
# =========================================================

with st.sidebar:
    st.markdown("### 🟡🔵 Scout Parametreleri")
    
    with st.form("player_input_form"):
        player_name = st.text_input("Oyuncu Adı", value="Yeni Transfer", placeholder="Örn: Alex de Souza")
        
        if available_positions:
            selected_position = st.selectbox(
                "Mevki (Position)", 
                options=available_positions,
                format_func=lambda pos: f"{pos} - {POSITION_NAMES.get(pos, pos)}",
                index=available_positions.index("ST") if "ST" in available_positions else 0
            )
        else:
            selected_position = st.selectbox(
                "Mevki (Position)", 
                options=POSITION_CODES,
                format_func=lambda pos: f"{pos} - {POSITION_NAMES.get(pos, pos)}",
                index=POSITION_CODES.index("ST")
            )
            
        st.markdown("---")
        st.markdown("⚙️ **Fiziksel & Mali Veriler**")
        age = st.slider("Yaş", 16, 45, 25)
        height = st.slider("Boy (cm)", 150, 210, 180)
        weight = st.slider("Kilo (kg)", 50, 110, 75)
        value = st.number_input("Piyasa Değeri (€)", 0, 500_000_000, 5_000_000, step=250_000)
        wage = st.number_input("Haftalık Maaş (€)", 0, 5_000_000, 20_000, step=1_000)
        
        st.markdown("---")
        st.markdown("📊 **Teknik Özellik Karakteristiği**")
        pace = st.slider("⚡ Hız (Pace)", 0, 100, 70)
        shooting = st.slider("🥅 Şut (Shooting)", 0, 100, 60)
        passing = st.slider("🎯 Pas (Passing)", 0, 100, 65)
        dribbling = st.slider("⚽ Top Sürme (Dribbling)", 0, 100, 68)
        defending = st.slider("🛡️ Defans (Defending)", 0, 100, 50)
        physic = st.slider("💪 Fizik (Physic)", 0, 100, 70)
        
        submit_button = st.form_submit_button("🔍 Performansı Hesapla")


# =========================================================
# 8. VERİ HAZIRLAMA VE SESSION STATE YÖNETİMİ
# =========================================================

if submit_button:
    input_data = {
        "age": age, "height_cm": height, "weight_kg": weight,
        "value_eur": value, "wage_eur": wage, "pace": pace,
        "shooting": shooting, "passing": passing, "dribbling": dribbling,
        "defending": defending, "physic": physic
    }

    for column in feature_cols:
        if column not in input_data:
            input_data[column] = 0

    if selected_position in position_columns:
        input_data[position_columns[selected_position]] = 1

    input_df = pd.DataFrame([input_data]).reindex(columns=feature_cols, fill_value=0)

    try:
        prediction = float(model.predict(input_df)[0])
        
        st.session_state["analysis_result"] = {
            "prediction": prediction,
            "player_name": player_name.strip() if player_name.strip() else "Bilinmeyen Oyuncu",
            "position": selected_position,
            "input_df": input_df,
            "metrics": {
                "Piyasa Değeri": format_currency(value),
                "Haftalık Maaş": format_currency(wage),
                "Fiziksel": f"{height} cm / {weight} kg"
            },
            "player_values": {
                "pace": pace, "shooting": shooting, "passing": passing,
                "dribbling": dribbling, "defending": defending, "physic": physic
            }
        }
        st.toast(f"📊 {player_name} için analiz başarıyla tamamlandı!", icon="⚽")
    except Exception as e:
        st.error(f"Tahmin motorunda hata oluştu: {e}")


# =========================================================
# 9. SAĞ PANEL - SONUÇLAR VE GÖRSELLEŞTİRME
# =========================================================

if "analysis_result" in st.session_state:
    res = st.session_state["analysis_result"]
    
    col_score, col_meta = st.columns([1, 2])
    
    with col_score:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">{res['player_name']} ({res['position']})</div>
                <div class="prediction-number">{res['prediction']:.1f}</div>
                <div class="prediction-comment">{get_player_level(res['prediction'])}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col_meta:
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f'<div class="info-box"><div class="info-box-title">Değer</div><div class="info-box-value">{res["metrics"]["Piyasa Değeri"]}</div></div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="info-box"><div class="info-box-title">Maaş</div><div class="info-box-value">{res["metrics"]["Haftalık Maaş"]}</div></div>', unsafe_allow_html=True)
        with m_col3:
            st.markdown(f'<div class="info-box"><div class="info-box-title">Boy / Kilo</div><div class="info-box-value" style="font-size:1.15rem; padding-top:4px;">{res["metrics"]["Fiziksel"]}</div></div>', unsafe_allow_html=True)
            
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🧠 Model Karar Mekanizması (SHAP)", "🕸️ Yetenek Profil Grafiği", "📋 Ham Etki Tablosu"])
    
    with tab1:
        st.subheader("🔍 Yapay Zekâ Karar Analizi (XAI)")
        st.markdown(
            "Aşağıdaki şelale grafiği, modelin baz değerden başlayarak "
            "verdiğiniz özelliklere göre tahmini skoru nasıl şekillendirdiğini gösterir. "
            "<span style='color:#ff4b4b; font-weight:bold;'>Kırmızı</span> barlar olumlu, "
            "<span style='color:#00a3ff; font-weight:bold;'>Mavi</span> barlar olumsuz etkileri temsil eder.", 
            unsafe_allow_html=True
        )
        
        try:
            shap_values = explainer(res["input_df"])
            
            translated_names = [FEATURE_TRANSLATIONS.get(name, name) for name in shap_values.feature_names]
            shap_values.feature_names = translated_names
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            
            plt.gcf().axes[0].set_xlabel("SHAP Değeri (Skora Etki)", color="white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP grafiği çizilirken bir uyarı oluştu. Alternatif olarak 'Ham Etki Tablosu' sekmesini inceleyebilirsiniz.")
            
    with tab2:
        radar_fig = create_radar_chart(res["player_values"])
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.pyplot(radar_fig)
        plt.close(radar_fig)
        
    with tab3:
        st.subheader("📊 Özelliklerin Skora Doğrudan Katkı Listesi")
        try:
            shap_values_raw = explainer(res["input_df"])
            
            turkish_features = [FEATURE_TRANSLATIONS.get(name, name) for name in shap_values_raw.feature_names]
            
            shap_df = pd.DataFrame({
                "Özellik Modülü": turkish_features,
                "Girilen Değer": np.asarray(shap_values_raw.data).flatten(),
                "Skora Net Etkisi (Puan)": np.asarray(shap_values_raw.values).flatten()
            })
            shap_df["Mutlak Önem"] = shap_df["Skora Net Etkisi (Puan)"].abs()
            shap_df = shap_df.sort_values(by="Mutlak Önem", ascending=False).drop(columns=["Mutlak Önem"])
            
            st.dataframe(
                shap_df.style.format({"Girilen Değer": "{:,.0f}", "Skora Net Etkisi (Puan)": "{:+.2f}"}),
                use_container_width=True,
                hide_index=True
            )
        except Exception as e:
            st.error(f"Tablo oluşturulamadı: {e}")

else:
    st.markdown(
        """
        <div style="text-align: center; padding: 80px 20px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.1);">
            <p style="font-size: 2.5rem; margin-bottom: 10px;">📋</p>
            <p style="font-size: 1.3rem; color: #a0aec0; font-weight: 600;">Analize Hazır!</p>
            <p style="color: #718096; max-width: 500px; margin: 0 auto;">Sol taraftaki menüden oyuncu verilerini düzenleyin ve ardından <b>"Performansı Hesapla"</b> butonuna basarak scout raporunu oluşturun.</p>
        </div>
        """,
        unsafe_allow_html=True
    )