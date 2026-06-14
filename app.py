"""
==========================================================================================
PROJE ADI: ScoutIntel Pro Enterprise Suite v5.0 (Unified Master Edition)
GELİŞTİRİCİ: İmran Devran
AKADEMİK KURUM: Manisa Celal Bayar Üniversitesi (MCBÜ) - Veri Bilimi ve Analitiği
==========================================================================================
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime

# PDF Raporlama Eklentisi Bileşenleri (ReportLab Enterprise)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# =========================================================
# SAYFA AYARLARI & GÜVENLİK
# =========================================================
st.set_page_config(
    page_title="Fener Analytics | Pro Scout Panel",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_football_model.pkl"
DATA_PATH = BASE_DIR / "cleaned_football_players.csv"

# =========================================================
# GELİŞMİŞ FENERBAHÇE TEMALI CSS TASARIMI (MODERN)
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background: 
                radial-gradient(circle at top right, rgba(255, 237, 0, 0.08), transparent 40%),
                linear-gradient(135deg, #040d1a 0%, #08162e 50%, #020710 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1600px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020712 0%, #061630 100%);
            border-right: 3px solid #ffed00;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] span {
            color: #ffffff !important;
        }
        .hero-container {
            background: linear-gradient(120deg, rgba(4, 18, 41, 0.95), rgba(8, 30, 66, 0.90));
            border: 1px solid rgba(255, 237, 0, 0.3);
            border-left: 8px solid #ffed00;
            border-radius: 14px;
            padding: 22px 28px;
            margin-bottom: 25px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
        }
        .hero-title {
            color: #ffed00 !important;
            font-size: 2.6rem;
            font-weight: 900;
            margin: 0;
            letter-spacing: 0.5px;
        }
        .hero-subtitle {
            color: #ffffff !important;
            font-size: 1.15rem;
            margin-top: 6px;
            margin-bottom: 0;
            opacity: 0.85;
        }
        .info-box {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .info-box:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 237, 0, 0.3);
        }
        .info-box-title {
            font-size: 0.85rem;
            color: #a0aec0;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }
        .info-box-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #ffffff;
        }
        .prediction-card {
            background: linear-gradient(135deg, #ffed00 0%, #c4b100 100%);
            color: #040d1a !important;
            border-radius: 16px;
            text-align: center;
            padding: 25px 15px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
            border: 2px solid #ffffff;
        }
        .prediction-label {
            font-size: 1rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #040d1a;
        }
        .prediction-number {
            font-size: 5rem;
            line-height: 1;
            font-weight: 900;
            margin: 12px 0;
            color: #040d1a;
        }
        .prediction-comment {
            font-size: 1.1rem;
            font-weight: 800;
            color: #040d1a;
        }
        .stButton > button, .stFormSubmitButton > button {
            width: 100%;
            background: linear-gradient(90deg, #ffed00, #c4b100) !important;
            color: #040d1a !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.8rem 1rem !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
            box-shadow: 0 6px 18px rgba(255, 237, 0, 0.2) !important;
            transition: all 0.25s ease-in-out !important;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background: #ffffff !important;
            color: #040d1a !important;
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(255, 255, 255, 0.25) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 12px;
            padding: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: #cbd5e0 !important;
            padding: 10px 24px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffed00 !important;
            color: #040d1a !important;
            font-weight: 800 !important;
        }
        .similar-card {
            background: rgba(255, 255, 255, 0.03);
            border-left: 4px solid #ffed00;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# VERİ VE MODEL YÜKLEME (GÜVENLİ VE ÖNBELLEKLİ)
# =========================================================
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        st.error("Kritik Hata: Model veya veri dosyası bulunamadı.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    df = pd.read_csv(DATA_PATH)
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")
    loaded_features = list(loaded_model.feature_names_in_) if hasattr(loaded_model, "feature_names_in_") else [c for c in df.columns if c != "overall"]
    return loaded_model, loaded_features, df

try:
    model, feature_cols, raw_dataset = load_assets()
    explainer = shap.Explainer(model)
except Exception as error:
    st.error("Sistem başlatılırken hata oluştu.")
    st.code(str(error))
    st.stop()

POSITION_CODES = ["GK", "CB", "LB", "RB", "LWB", "RWB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "CF", "ST"]
POSITION_NAMES = {
    "GK": "Kaleci", "CB": "Stoper", "LB": "Sol Bek", "RB": "Sağ Bek",
    "LWB": "Sol Kanat Bek", "RWB": "Sağ Kanat Bek", "CDM": "Defansif Orta Saha",
    "CM": "Merkez Orta Saha", "CAM": "Ofansif Orta Saha", "LM": "Sol Orta Saha",
    "RM": "Sağ Orta Saha", "LW": "Sol Kanat", "RW": "Sağ Kanat", "CF": "İkinci Forvet", "ST": "Santrafor"
}
FEATURE_TRANSLATIONS = {
    "age": "Yaş", "height_cm": "Boy (cm)", "weight_kg": "Kilo (kg)",
    "value_eur": "Piyasa Değeri (€)", "wage_eur": "Haftalık Maaş (€)",
    "pace": "Hız (Pace)", "shooting": "Şut (Shooting)", "passing": "Pas (Passing)",
    "dribbling": "Top Sürme (Dribbling)", "defending": "Defans (Defending)", "physic": "Fizik (Physic)"
}

def find_position_columns(columns):
    mapping = {}
    for pos in POSITION_CODES:
        pos_upper = pos.upper()
        for col in columns:
            norm = str(col).upper().replace("-", "_").replace(" ", "_")
            if norm in {pos_upper, f"POSITION_{pos_upper}", f"POSITIONS_{pos_upper}", f"PLAYER_POSITIONS_{pos_upper}", f"POS_{pos_upper}"}:
                mapping[pos] = col
                break
    return mapping

position_columns = find_position_columns(feature_cols)
available_positions = list(position_columns.keys()) if position_columns else POSITION_CODES

def format_currency(val):
    val = float(val)
    if val >= 1_000_000_000: return f"{val / 1_000_000_000:.1f} Milyar €"
    if val >= 1_000_000: return f"{val / 1_000_000:.1f} Milyon €"
    return f"{val:,.0f} €".replace(",", ".")

def get_player_level(pred):
    if pred >= 90: return "👑 Efsane Çubuklu Seviyesi"
    if pred >= 85: return "🌟 Dünya Klasında Kadro Lideri"
    if pred >= 80: return "🔥 Üst Düzey Yıldız"
    if pred >= 75: return "💪 Kaliteli Lig Oyuncusu"
    if pred >= 70: return "📈 Rotasyon / Gelecek Vaadeden"
    if pred >= 65: return "🏃 Ortalama Alternatif"
    return "🌱 Geliştirilmesi Gereken Genç Yetenek"

def find_similar_players(input_features, current_pos, top_n=3):
    try:
        df_copy = raw_dataset.copy()
        if position_columns and current_pos in position_columns:
            pos_col = position_columns[current_pos]
            if pos_col in df_copy.columns: df_copy = df_copy[df_copy[pos_col] == 1]
        if df_copy.empty or len(df_copy) < top_n: df_copy = raw_dataset.copy()
        base_skills = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
        valid_skills = [s for s in base_skills if s in df_copy.columns and s in input_features]
        if not valid_skills: return pd.DataFrame()
        target_vector = np.array([input_features[s] for s in valid_skills])
        matrix = df_copy[valid_skills].values
        distances = np.linalg.norm(matrix - target_vector, axis=1)
        df_copy["Similarity_Score"] = 100 / (1 + distances)
        name_col = "short_name" if "short_name" in df_copy.columns else ("player_name" if "player_name" in df_copy.columns else "generated_name")
        if name_col == "generated_name": df_copy["generated_name"] = "Profil-" + df_copy.index.astype(str)
        target_cols = [name_col, "overall", "Similarity_Score"]
        if "value_eur" in df_copy.columns: target_cols.append("value_eur")
        if "age" in df_copy.columns: target_cols.append("age")
        return df_copy.sort_values(by="Similarity_Score", ascending=False).head(top_n)[target_cols]
    except:
        return pd.DataFrame()

def draw_plotly_radar(player_data_dict, comparison_mode=False, pair_data_dict=None):
    categories = ["Hız", "Şut", "Pas", "Top Sürme", "Defans", "Fizik"]
    keys = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
    fig = go.Figure()
    val1 = [player_data_dict[k] for k in keys]
    val1.append(val1[0])
    fig.add_trace(go.Scatterpolar(
        r=val1, theta=categories + [categories[0]], fill='toself', name=player_data_dict.get("name", "Oyuncu A"),
        line=dict(color='#ffed00', width=3), fillcolor='rgba(255, 237, 0, 0.25)'
    ))
    if comparison_mode and pair_data_dict:
        val2 = [pair_data_dict[k] for k in keys]
        val2.append(val2[0])
        fig.add_trace(go.Scatterpolar(
            r=val2, theta=categories + [categories[0]], fill='toself', name=pair_data_dict.get("name", "Oyuncu B"),
            line=dict(color='#00a3ff', width=3), fillcolor='rgba(0, 163, 255, 0.25)'
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='#a0aec0')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='#ffffff', size=12)),
            bgcolor='rgba(6, 22, 46, 0.6)'
        ),
        showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#ffffff')), margin=dict(t=30, b=30, l=40, r=40)
    )
    return fig

def export_scout_pdf(name, pos, score, val, wage, h, w, skills, desc):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#0f1c2e'), spaceAfter=15)
    body_s = ParagraphStyle('Body', parent=styles['Normal'], fontSize=11, leading=14, spaceAfter=10)
    story.append(Paragraph(f"FENER ANALYTICS SCOUTING REPORT", title_s))
    story.append(Paragraph(f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}", body_s))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"<b>Futbolcu Kimliği:</b> {name} | <b>Mevki:</b> {pos} | <b>Fizik:</b> {h} cm / {w} kg", body_s))
    story.append(Paragraph(f"<b>Yapay Zekâ Güç Puanı (OVR):</b> {score:.1f} / 100", body_s))
    story.append(Paragraph(f"<b>Finansal Değerleme:</b> Bonservis: {format_currency(val)} | Maaş: {format_currency(wage)}/Hafta", body_s))
    story.append(Paragraph(f"<b>Moneyball Bütçe Analizi:</b> {desc}", body_s))
    story.append(Spacer(1, 15))
    table_data = [
        ['Öznitelik', 'Skor', 'Öznitelik', 'Skor'],
        ['Hız (PAC)', str(skills['pace']), 'Top Sürme (DRI)', str(skills['dribbling'])],
        ['Şut (SHO)', str(skills['shooting']), 'Defans (DEF)', str(skills['defending'])],
        ['Pas (PAS)', str(skills['passing']), 'Fizik (PHY)', str(skills['physic'])]
    ]
    t = Table(table_data, colWidths=[130, 100, 130, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f1c2e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9fa')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
    ]))
    story.append(t)
    doc.build(story)
    buffer.seek(0)
    return buffer

st.markdown("""<div class="hero-container"><p class="hero-title">💛💙 Fener Analytics Scout Pro Panel</p><p class="hero-subtitle">Yapay Zekâ, XAI ve Moneyball Bütçe Güdümlü Küresel Futbolcu Analiz Platformu</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛠️ Sistem Yapılandırması")
    app_mode = st.radio("Çalışma Analiz Modu", ["Tekli Oyuncu Scout Raporu", "Kafa Kafaya Karşılaştırma"])
    st.markdown("---")

if app_mode == "Tekli Oyuncu Scout Raporu":
    with st.sidebar:
        st.markdown("### 🟡🔵 Oyuncu Parametreleri")
        with st.form("single_scout_form"):
            s_name = st.text_input("Oyuncu Adı", value="Yeni Transfer")
            s_pos = st.selectbox("Hedef Mevki", options=available_positions, format_func=lambda x: f"{x} - {POSITION_NAMES.get(x, x)}")
            st.markdown("---")
            s_age = st.slider("Yaş", 16, 45, 24)
            s_h = st.slider("Boy (cm)", 150, 210, 182)
            s_w = st.slider("Kilo (kg)", 50, 110, 77)
            s_val = st.number_input("Piyasa Değeri (€)", 0, 500_000_000, 8_500_000, step=500_000)
            s_wage = st.number_input("Haftalık Maaş (€)", 0, 5_000_000, 35_000, step=5_000)
            st.markdown("---")
            s_pac = st.slider("⚡ Hız (Pace)", 0, 100, 75)
            s_sho = st.slider("🥅 Şut (Shooting)", 0, 100, 68)
            s_pas = st.slider("🎯 Pas (Passing)", 0, 100, 72)
            s_dri = st.slider("⚽ Top Sürme (Dribbling)", 0, 100, 74)
            s_def = st.slider("🛡️ Defans (Defending)", 0, 100, 45)
            s_phy = st.slider("💪 Fizik (Physic)", 0, 100, 70)
            submit_single = st.form_submit_button("🔍 Detaylı Analiz Raporu Üret")

    if submit_single or "single_active_data" in st.session_state:
        if submit_single:
            input_dict = {"age": s_age, "height_cm": s_h, "weight_kg": s_w, "value_eur": s_val, "wage_eur": s_wage, "pace": s_pac, "shooting": s_sho, "passing": s_pas, "dribbling": s_dri, "defending": s_def, "physic": s_phy}
            for col in feature_cols:
                if col not in input_dict: input_dict[col] = 0
            if s_pos in position_columns: input_dict[position_columns[s_pos]] = 1
            pred_df = pd.DataFrame([input_dict]).reindex(columns=feature_cols, fill_value=0)
            score = float(model.predict(pred_df)[0])
            
            rounded_ovr = int(round(score))
            peers = raw_dataset[(raw_dataset['overall'] >= rounded_ovr - 1) & (raw_dataset['overall'] <= rounded_ovr + 1)]
            m_wage = float(peers['wage_eur'].mean()) if len(peers) > 0 else s_wage * 0.95
            m_value = float(peers['value_eur'].mean()) if len(peers) > 0 else s_val * 0.95
            efficiency = ((m_wage - s_wage) / m_wage) * 100 if m_wage > 0 else 0
            m_desc = f"📈 FİNANSAL VERİMLİLİK: %{efficiency:.1f} AVANTAJ. Oyuncunun bütçe yapısı piyasa ortalamasından daha tasarrufludur." if efficiency >= 0 else f"⚠️ BÜTÇE ANOMALİSİ: %{abs(efficiency):.1f} AŞIRI ÖDEME. Oyuncunun maaş talebi ürettiği teknik kalitenin çok üzerindedir!"
            sim_df = find_similar_players(input_dict, s_pos, top_n=3)
            
            st.session_state["single_active_data"] = {
                "name": s_name, "pos": s_pos, "score": score, "metrics": [s_val, s_wage, s_h, s_w],
                "skills": {"pace": s_pac, "shooting": s_sho, "passing": s_pas, "dribbling": s_dri, "defending": s_def, "physic": s_phy, "name": s_name},
                "pred_df": pred_df, "sim_df": sim_df, "m_desc": m_desc, "m_data": [m_wage, m_value]
            }

        data = st.session_state["single_active_data"]
        c_score, c_meta = st.columns([1, 2])
        with c_score:
            st.markdown(f"""<div class="prediction-card"><div class="prediction-label">{data['name']} | {data['pos']}</div><div class="prediction-number">{data['score']:.1f}</div><div class="prediction-comment">{get_player_level(data['score'])}</div></div>""", unsafe_allow_html=True)
        with c_meta:
            m1, m2, m3 = st.columns(3)
            with m1: st.markdown(f'<div class="info-box"><div class="info-box-title">Piyasa Değeri</div><div class="info-box-value">{format_currency(data["metrics"][0])}</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="info-box"><div class="info-box-title">Haftalık Maaş</div><div class="info-box-value">{format_currency(data["metrics"][1])}</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="info-box"><div class="info-box-title">Fiziksel Profil</div><div class="info-box-value" style="font-size:1.15rem; padding-top:4px;">{data["metrics"][2]} cm / {data["metrics"][3]} kg</div></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["🧠 XAI Karar Mekanizması", "🕸️ Yetenek Profil Grafiği", "💸 Moneyball Finans Raporu", "📊 Benzer Oyuncu Algoritması"])
        
        with tab1:
            st.subheader("🔍 Yapay Zekâ Neden Bu Puanı Verdi?")
            try:
                s_values = explainer(data["pred_df"])
                s_values.feature_names = [FEATURE_TRANSLATIONS.get(n, n) for n in s_values.feature_names]
                importance_df = pd.DataFrame({"Analiz Modülü": s_values.feature_names, "Girdi Değeri": np.asarray(s_values.data).flatten(), "Skora Katkısı (Puan)": np.asarray(s_values.values).flatten()})
                importance_df["Mutlak"] = importance_df["Skora Katkısı (Puan)"].abs()
                importance_df = importance_df.sort_values(by="Mutlak", ascending=False).drop(columns=["Mutlak"])
                st.dataframe(importance_df.style.format({"Girdi Değeri": "{:,.0f}", "Skora Katkısı (Puan)": "{:+.2f}"}), use_container_width=True, hide_index=True)
            except:
                st.warning("XAI Tablosu oluşturulurken teknik bir sınıra takılındı.")
        with tab2:
            st.plotly_chart(draw_plotly_radar(data["skills"]), use_container_width=True)
        with tab4:
            st.subheader("📋 Algoritmik Olarak En Benzer Oyuncu Profilleri")
            if not data["sim_df"].empty:
                for idx, row in data["sim_df"].iterrows():
                    name_val = row.get("short_name", row.get("player_name", row.get("generated_name", "Bilinmiyor")))
                    st.markdown(f"""<div class="similar-card"><strong style='color:#ffed00; font-size:1.1rem;'>{name_val}</strong> — <b>Genel Puan:</b> {row['overall']} | <b>Yaş:</b> {row.get('age', 'Bilinmiyor')} | <b>Tahmini Değer:</b> {format_currency(row['value_eur']) if 'value_eur' in row else 'Bilinmiyor'} <span style='color:#00a3ff; float:right;'><b>Benzerlik Oranı:</b> %{row['Similarity_Score']:.1f}</span></div>""", unsafe_allow_html=True)
        with tab3:
            st.subheader("💰 Moneyball Akıllı Bütçe Analizi")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label="Model Öngörüsü Adil Piyasa Maaşı", value=f"{format_currency(data['m_data'][0])} /Hafta")
            col_m2.metric(label="Model Öngörüsü Gerçek Bonservis Değeri", value=format_currency(data['m_data'][1]))
            st.info(data["m_desc"])
            
        st.markdown("---")
        pdf_buffer = export_scout_pdf(data["name"], data["pos"], data["score"], data["metrics"][0], data["metrics"][1], data["metrics"][2], data["metrics"][3], data["skills"], data["m_desc"])
        st.download_button(label="📄 RESMİ SCOUT RAPORUNU İNDİR (PDF)", data=pdf_buffer, file_name=f"ScoutIntel_Report_{data['name']}.pdf", mime="application/pdf", use_container_width=True)
else:
    st.subheader("⚔️ İki Oyuncu Kafa Kafaya Karşılaştırma Analizi")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        st.markdown("<h4 style='color:#ffed00;'>1. Oyuncu Bilgileri</h4>", unsafe_allow_html=True)
        p1_name = st.text_input("Adı", value="Oyuncu A", key="p1n")
        p1_pos = st.selectbox("Mevki", options=available_positions, key="p1p", format_func=lambda x: f"{x} - {POSITION_NAMES.get(x, x)}")
        p1_age = st.slider("Yaş ", 16, 45, 23, key="p1a")
        p1_val = st.number_input("Piyasa Değeri (€) ", 0, 500_000_000, 12_000_000, key="p1v")
        p1_pac = st.slider("Hız (Pace) ", 0, 100, 80, key="p1pac")
        p1_sho = st.slider("Şut (Shooting) ", 0, 100, 75, key="p1sho")
        p1_pas = st.slider("Pas (Passing) ", 0, 100, 70, key="p1pas")
        p1_dri = st.slider("Top Sürme (Dribbling) ", 0, 100, 78, key="p1dri")
        p1_def = st.slider("Defans (Defending) ", 0, 100, 40, key="p1def")
        p1_phy = st.slider("Fizik (Physic) ", 0, 100, 65, key="p1phy")
    with col_input2:
        st.markdown("<h4 style='color:#00a3ff;'>2. Oyuncu Bilgileri</h4>", unsafe_allow_html=True)
        p2_name = st.text_input("Adı ", value="Oyuncu B", key="p2n")
        p2_pos = st.selectbox("Mevki ", options=available_positions, key="p2p", format_func=lambda x: f"{x} - {POSITION_NAMES.get(x, x)}")
        p2_age = st.slider("Yaş  ", 16, 45, 27, key="p2a")
        p2_val = st.number_input("Piyasa Değeri (€)  ", 0, 500_000_000, 15_000_000, key="p2v")
        p2_pac = st.slider("Hız (Pace)  ", 0, 100, 72, key="p2pac")
        p2_sho = st.slider("Şut (Shooting)  ", 0, 100, 82, key="p2sho")
        p2_pas = st.slider("Pas (Passing)  ", 0, 100, 75, key="p2pas")
        p2_dri = st.slider("Top Sürme (Dribbling)  ", 0, 100, 74, key="p2dri")
        p2_def = st.slider("Defans (Defending)  ", 0, 100, 55, key="p2def")
        p2_phy = st.slider("Fizik (Physic)  ", 0, 100, 78, key="p2phy")

    st.markdown("---")
    trigger_compare = st.button("⚔️ Düelloyu Başlat ve Performansları Kıyasla")
    if trigger_compare or "compare_active_data" in st.session_state:
        if trigger_compare:
            d1 = {"age": p1_age, "height_cm": 180, "weight_kg": 75, "value_eur": p1_val, "wage_eur": 20_000, "pace": p1_pac, "shooting": p1_sho, "passing": p1_pas, "dribbling": p1_dri, "defending": p1_def, "physic": p1_phy}
            for col in feature_cols:
                if col not in d1: d1[col] = 0
            if p1_pos in position_columns: d1[position_columns[p1_pos]] = 1
            score1 = float(model.predict(pd.DataFrame([d1]).reindex(columns=feature_cols, fill_value=0))[0])
            
            d2 = {"age": p2_age, "height_cm": 180, "weight_kg": 75, "value_eur": p2_val, "wage_eur": 20_000, "pace": p2_pac, "shooting": p2_sho, "passing": p2_pas, "dribbling": p2_dri, "defending": p2_def, "physic": p2_phy}
            for col in feature_cols:
                if col not in d2: d2[col] = 0
            if p2_pos in position_columns: d2[position_columns[p2_pos]] = 1
            score2 = float(model.predict(pd.DataFrame([d2]).reindex(columns=feature_cols, fill_value=0))[0])
            st.session_state["compare_active_data"] = {
                "p1_name": p1_name, "p2_name": p2_name, "score1": score1, "score2": score2,
                "skills1": {"pace": p1_pac, "shooting": p1_sho, "passing": p1_pas, "dribbling": p1_dri, "defending": p1_def, "physic": p1_phy, "name": p1_name},
                "skills2": {"pace": p2_pac, "shooting": p2_sho, "passing": p2_pas, "dribbling": p2_dri, "defending": p2_def, "physic": p2_phy, "name": p2_name},
                "meta1": [p1_age, p1_val], "meta2": [p2_age, p2_val]
            }
        c_data = st.session_state["compare_active_data"]
        sc1, sc2 = st.columns(2)
        with sc1: st.markdown(f"""<div class="prediction-card" style="background: linear-gradient(135deg, #ffed00 0%, #caa400 100%);"><div class="prediction-label">{c_data['p1_name']}</div><div class="prediction-number">{c_data['score1']:.1f}</div><div class="prediction-comment">Piyasa Değeri: {format_currency(c_data['meta1'][1])}</div></div>""", unsafe_allow_html=True)
        with sc2: st.markdown(f"""<div class="prediction-card" style="background: linear-gradient(135deg, #00a3ff 0%, #0066cc 100%); color:#ffffff !important;"><div class="prediction-label" style="color:#ffffff;">{c_data['p2_name']}</div><div class="prediction-number" style="color:#ffffff;">{c_data['score2']:.1f}</div><div class="prediction-comment" style="color:#ffffff;">Piyasa Değeri: {format_currency(c_data['meta2'][1])}</div></div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.plotly_chart(draw_plotly_radar(c_data["skills1"], comparison_mode=True, pair_data_dict=c_data["skills2"]), use_container_width=True)