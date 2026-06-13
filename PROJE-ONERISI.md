# Proje Önerisi

**Seçilen Görev Numarası:** Seçenek 3 - Açıklanabilir Makine Öğrenmesi Karar Destek Ürünü
**Ürünün Adı:** Futbolcu Performans Tahmin ve Açıklanabilir Yapay Zekâ (XAI) Sistemi

### 1. Çözülecek Problem
Modern futbolda scoutlar, teknik direktörler ve kulüp yöneticileri oyuncu transfer ederken veya maç stratejisi belirlerken ham istatistiklere (gol, asist, pas yüzdesi vb.) odaklanmaktadır. Ancak klasik makine öğrenmesi modelleri bir oyuncunun gelecekteki performansını tahmin etse bile, bu tahminin arka planındaki nedenleri (hangi istatistiğin tahmini ne kadar etkilediğini) açıklayamayan bir "kara kutu" (black-box) olarak kalmaktadır. Bu proje, futbolcu verileri üzerinden performans tahmini yaparken SHAP yöntemiyle kararların nedenlerini şeffaf ve açıklanabilir hale getirerek teknik ekiplere güvenilir bir karar destek mekanizması sunmayı amaçlar.

### 2. Hedef Kullanıcı
- Futbol Scoutları (Gözlemciler)
- Teknik Direktörler ve Spor Analistleri
- Kulüp Sportif Direktörleri

### 3. Kullanılacak Veri veya Bilgi Kaynakları
Kaggle üzerinden alınacak güncel ve doğrulanmış futbolcu istatistik veri setleri (FIFA veri setleri veya Avrupa Top 5 Lig oyuncu istatistikleri) kullanılacaktır.

### 4. Kullanılması Planlanan Teknolojiler
- **Dil:** Python
- **Veri Önişleme:** Pandas, NumPy
- **Modelleme:** Scikit-learn (Random Forest), XGBoost
- **Açıklanabilir Yapay Zekâ:** SHAP (SHapley Additive exPlanations)
- **Arayüz ve Dağıtım:** Streamlit

### 5. Beklenen Ürün Çıktısı
Kullanıcının bir futbolcunun teknik özelliklerini (yaş, pas yüzdesi, şut, top çalma vb.) arayüz üzerinden girerek oyuncunun potansiyel performans/reyting skorunu görebileceği, aynı zamanda SHAP grafikler yardımıyla bu tahminin "neden" üretildiğini (hangi özelliklerin skoru artırıp azalttığını) canlı olarak inceleyebileceği web tabanlı bir karar destek uygulaması.

### 6. Ürünün Diğer Çalışmalardan Ayrılan Yönü
Sadece ham bir tahmin skoru üretmekle kalmayıp, Açıklanabilir Yapay Zekâ (XAI) entegrasyonu sayesinde her futbolcu özelinde modelin karar mekanizmasını görsel grafiklerle (Force Plot / Summary Plot) tamamen şeffaf hale getirmesi ve kullanıcılara "tahminin gerekçesini" sunmasıdır.
