"""
YouTube Video Araştırmacısı
Maestra-Style SaaS UI — Clean, Corporate, Modern
Geliştirilmiş RAG motoru, prompt mühendisliği ve chat geçmişi desteği.
"""

import os
import warnings
import logging

# Telemetri ve gereksiz kütüphane uyarılarını kapat
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)

# ChromaDB içindeki bozuk telemetri fonksiyonunu tamamen etkisiz hale getir (Monkey Patch)
try:
    import chromadb.telemetry.posthog as _tp
    _tp.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

try:
    import chromadb.telemetry.product.posthog as _tpp
    _tpp.Posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

import time
import streamlit as st
from langchain_community.llms import Ollama
from fetch_transcript import get_clean_transcript, get_timestamped_transcript, format_timestamp
from rag_engine import create_vector_db, search_in_db

# ─────────────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI YouTube Video Özetleyici",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────
# MAESTRA-STYLE CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── GLOBAL RESET ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── SAYFA ARKA PLAN: Soft Aurora (#f8fafc tabanlı) ── */
    .stApp {
        background: linear-gradient(160deg,
            #f8fafc 0%, #eff6ff 25%, #f8fafc 50%, #faf5ff 75%, #f8fafc 100%
        ) !important;
        background-size: 300% 300% !important;
        animation: softAurora 20s ease infinite !important;
        min-height: 100vh !important;
    }
    @keyframes softAurora {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    header[data-testid="stHeader"], #MainMenu, footer { display: none !important; }

    /* ── LAYOUT ── */
    .main .block-container {
        max-width: 100% !important;
        padding: 2rem 4rem 3rem !important;
    }

    /* ── TÜM METİNLER KOYU ── */
    h1, h2, h3, h4, h5, h6, p, span, label, div,
    [class*="stMarkdown"] p, .stCaption p {
        color: #1e293b !important;
    }

    /* ── HERO ── */
    .hero-section {
        text-align: center;
        padding: 5rem 2rem 2rem;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: #eff6ff;
        border: 1.5px solid #bfdbfe;
        border-radius: 9999px;
        padding: 0.45rem 1.2rem;
        font-size: 0.85rem; font-weight: 600;
        color: #2563eb !important;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: clamp(2.5rem, 5vw, 3.8rem);
        font-weight: 900;
        color: #0f172a !important;
        letter-spacing: -0.035em;
        line-height: 1.15;
        margin-bottom: 1rem;
    }
    .hero-title .blue { color: #2563eb !important; }
    .hero-desc {
        font-size: 1.15rem;
        color: #64748b !important;
        font-weight: 500;
        max-width: 600px;
        margin: 0 auto 2.5rem;
        line-height: 1.65;
    }

    /* ── PILL SEARCH BAR ── */
    .pill-search-wrapper {
        max-width: 640px;
        margin: 0 auto 3rem;
        position: relative;
    }
    .stTextInput label { display: none !important; }
    .pill-search-wrapper .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 14px 0 0 14px !important;
        color: #0f172a !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        padding: 0.85rem 1.2rem !important;
        height: 3.2rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04) !important;
        transition: all 0.2s ease !important;
        border-right: none !important;
    }
    .pill-search-wrapper .stTextInput > div > div > input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.1), 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    .pill-search-wrapper .stTextInput > div > div > input::placeholder {
        color: #94a3b8 !important; font-weight: 400 !important;
    }

    /* ── PRIMARY BUTON (Mavi) ── */
    .stButton > button[kind="primary"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        padding: 0.85rem 1.8rem !important;
        height: 3.2rem !important;
        background: #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(37,99,235,0.3) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"] * { color: #ffffff !important; }

    /* Pill-shape: buton sağ yuvarlak */
    .pill-search-wrapper .stButton > button[kind="primary"] {
        border-radius: 0 14px 14px 0 !important;
        height: 3.2rem !important;
        border-left: none !important;
    }

    /* ── SECONDARY BUTON ── */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind]) {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        background: transparent !important;
        color: #475569 !important;
        border: 1.5px solid #e2e8f0 !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind]):hover {
        background: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
        color: #1e293b !important;
    }
    .stButton > button[kind="secondary"] *,
    .stButton > button:not([kind]) * { color: #475569 !important; }

    /* ── VIDEO KARTLARI (Küçük, zarif) ── */
    .vcard {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        overflow: hidden;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
        position: relative;
    }
    .vcard:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }
    .vcard img {
        width: 100%;
        height: 150px;
        object-fit: cover;
    }
    .vcard-body { padding: 0.9rem 1rem 0.7rem; }
    .vcard-title {
        font-size: 0.88rem; font-weight: 700;
        color: #0f172a !important; line-height: 1.35;
        display: -webkit-box; -webkit-line-clamp: 2;
        -webkit-box-orient: vertical; overflow: hidden;
    }
    .vcard-channel {
        font-size: 0.78rem; font-weight: 500;
        color: #64748b !important; margin-top: 0.25rem;
    }
    /* Play icon overlay */
    .vcard-play {
        position: absolute; bottom: 0.7rem; right: 0.8rem;
        width: 36px; height: 36px;
        background: #2563eb;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3);
        transition: all 0.2s ease;
    }
    .vcard:hover .vcard-play { transform: scale(1.1); background: #1d4ed8; }
    .vcard-play svg { fill: #fff; width: 14px; height: 14px; }

    /* ── ÖZELLİK KARTLARI ── */
    .features-row {
        display: flex; gap: 1.5rem; justify-content: center;
        margin-top: 3.5rem; padding: 0 3rem; flex-wrap: wrap;
    }
    .feat-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 1.5rem;
        text-align: center; flex: 1; min-width: 200px; max-width: 280px;
        transition: all 0.2s ease;
    }
    .feat-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); transform: translateY(-2px); }
    .feat-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
    .feat-title { font-size: 0.95rem; font-weight: 700; color: #0f172a !important; margin-bottom: 0.3rem; }
    .feat-desc { font-size: 0.82rem; color: #64748b !important; font-weight: 400; line-height: 1.5; }

    /* ── WORKSPACE — Ghost Back Button ── */
    .ghost-back {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.88rem; font-weight: 600;
        color: #64748b !important; cursor: pointer;
        padding: 0.4rem 0; border: none; background: transparent;
        transition: color 0.2s ease;
    }
    .ghost-back:hover { color: #2563eb !important; }

    /* ── TABS (st.tabs modernize) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: #f1f5f9 !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid #e2e8f0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 0.55rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #64748b !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1e293b !important;
        background: rgba(255,255,255,0.6) !important;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Transcript Chunks ── */
    .t-chunk {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: all 0.15s ease;
    }
    .t-chunk:hover { border-color: #cbd5e1; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
    .t-time {
        font-size: 0.8rem; font-weight: 700;
        color: #2563eb !important;
        margin-bottom: 0.3rem;
        font-variant-numeric: tabular-nums;
    }
    .t-text { font-size: 0.9rem; color: #334155 !important; line-height: 1.65; font-weight: 400; }

    /* ── Scrollable container ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        background: #f8fafc !important;
    }

    /* ── Chat ── */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 0.9rem 1.2rem !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stChatMessage"] p {
        color: #1e293b !important; font-weight: 500 !important; font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
        font-size: 0.95rem !important; font-weight: 500 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
    }

    /* ── Video Info Box (workspace) ── */
    .vid-info {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem 1.2rem; margin-top: 1rem;
    }
    .vid-info-title { font-weight: 700; font-size: 0.95rem; color: #0f172a !important; }
    .vid-info-ch { font-size: 0.82rem; font-weight: 500; color: #64748b !important; margin-top: 0.15rem; }

    /* ── Summary Box ── */
    .sum-box {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 1.8rem;
    }
    .sum-label { font-weight: 700; font-size: 1rem; color: #2563eb !important; margin-bottom: 0.6rem; }
    .sum-text { color: #334155 !important; line-height: 1.8; font-weight: 400; font-size: 0.95rem; }

    /* ── Section Title ── */
    .section-title {
        text-align: center;
        font-size: 0.9rem;
        font-weight: 600;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1.2rem;
    }

    /* ── Workspace Top Bar ── */
    .ws-bar {
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: 1rem; margin-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .ws-bar-title { font-size: 1.1rem; font-weight: 700; color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# DEMO VERİLER
# ─────────────────────────────────────────────────────
DEMO_VIDEOS = {
    "demo1": {
        "title": "Büyük Dil Modelleri & RAG Mimarisi",
        "channel": "Andrej Karpathy (Türkçe Özet)",
        "url": "https://www.youtube.com/watch?v=zjkBMFhNj_g",
        "thumb": "https://img.youtube.com/vi/zjkBMFhNj_g/hqdefault.jpg",
        "duration_sec": 3587,  # 59:47
        "text": (
            "Büyük Dil Modellerine (LLM) giriş sunumuna hoş geldiniz. Bu videoda Andrej Karpathy LLM'lerin çalışma prensiplerini anlatmaktadır. "
            "1. Büyük Dil Modelleri Nedir? Temel Yapı: Bir LLM temel olarak parametreler dosyası (ağırlıklar) ve bu parametreleri çalıştıran C veya Python kodundan oluşur. "
            "Karpathy LLM'leri internetin kayıplı sıkıştırılmış hali (lossy compression) olarak tanımlar. Model metindeki bir sonraki kelimeyi tahmin etmeye çalışırken dünya hakkındaki bilgileri ağırlıklarına sıkıştırır. "
            "2. Eğitim Aşamaları (Pre-training ve Fine-tuning): Ön Eğitim (Pre-training) internetten toplanan devasa veri setlerinin binlerce GPU ile haftalarca işlendiği maliyetli aşamadır. Sonunda temel model (base model) ortaya çıkar. "
            "İnce Ayar (Fine-tuning): Soru-cevap formatındaki kaliteli verilerle modelin asistan haline getirildiği aşamadır. Ayrıca karşılaştırmalı verilerle RLHF (Insan Geri Bildirimli Pekiştirmeli Öğrenme) uygulanır. "
            "3. Gelecekteki Gelişmeler ve Yetenekler: Ölçeklenme Yasaları (Scaling Laws) parametre ve veri sayısı arttıkça modelin başarım gücünün öngörülebilir şekilde arttığını gösterir. "
            "Araç Kullanımı: Modern LLM'ler hesap makinesi, web tarayıcısı, Python kod yorumlayıcısı gibi harici araçları kullanır. "
            "Sistem 1 ve Sistem 2 Düşünce Yapısı: Şu anki modeller refleksif Sistem 1 yapısındadır. Hedef, modelin problem çözerken zaman harcadığı Sistem 2 seviyesine çıkmasıdır. "
            "4. Yeni İşletim Sistemi Olarak LLM'ler (LLM OS): Karpathy LLM'leri bir sohbet botu değil, yeni nesil işletim sisteminin çekirdeği (kernel) olarak konumlandırır. Bağlam penceresi (Context Window) RAM gibi, internet ve yerel dosyalar disk gibi çalışır. "
            "5. Yeni Güvenlik Zafiyetleri ve Saldırılar: Jailbreak (sınırları aşma) güvenlik filtrelerini aşmak, Prompt Injection (istem enjeksiyonu) gizli zararlı komutlarla manipüle etmek, Data Poisoning (veri zehirlenmesi) eğitim verilerine tetikleyiciler eklemektir. "
        ) * 10
    },
    "demo2": {
        "title": "İngilizce Öğrenme ve C2 Seviyesine Ulaşma",
        "channel": "POC English (Türkçe Özet)",
        "url": "https://www.youtube.com/watch?v=3bh6Gb0oBN0",
        "thumb": "https://img.youtube.com/vi/3bh6Gb0oBN0/hqdefault.jpg",
        "duration_sec": 1120,
        "timestamps": [
            {"start": 0, "duration": 180, "text": "Sıfırdan C2 seviyesine İngilizce öğrenme deneyimi ve akıcı konuşma stratejileri sunumu."},
            {"start": 180, "duration": 220, "text": "1. Dil Öğreniminde Psikolojik Engeller: Birçok kişi gramer kurallarına takıldığı için konuşmaktan korkar. Hata yapma korkusunu yenmek en önemli adımdır."},
            {"start": 400, "duration": 240, "text": "2. Maruz Kalma Yöntemi (Immersion): Dili ders gibi değil, hayatın bir parçası gibi yaşamak gerekir. İngilizce podcast dinlemek, dizi izlemek ve telefon dilini İngilizce yapmak kalıcılığı sağlar."},
            {"start": 640, "duration": 240, "text": "3. Bağlam İçi Kelime Öğrenimi: Tekil kelime listeleri ezberlemek yerine kelimeleri cümle ve bağlam içinde öğrenmek gerekir. Aralıklı tekrar (Spaced Repetition) yöntemi unutmayı engeller."},
            {"start": 880, "duration": 240, "text": "4. Akıcı Konuşma ve Output Pratikleri: Sadece dinlemek yetmez, kendi kendinize İngilizce konuşmak (Shadowing tekniği) ve sesinizi kaydetmek C2 seviyesine ulaşmanın anahtarıdır."}
        ],
        "text": (
            "Sıfırdan C2 seviyesine İngilizce öğrenme deneyimi ve akıcı konuşma rehberine hoş geldiniz. "
            "1. Dil Öğreniminde Psikolojik Engeller ve Zihniyet Dönüşümü: Birçok insan yıllarca İngilizce öğrenmeye çalışmasına rağmen akıcı konuşamaz. Bunun temel nedeni gramer kurallarına aşırı odaklanmak ve hata yapmaktan korkmaktır. "
            "Konuşmacı, dil öğreniminde mükemmeliyetçiliği bırakıp iletişime odaklanmanın C2 seviyesine ulaşmadaki kritik önemini vurgulamaktadır. "
            "2. Maruz Kalma Yöntemi (Passive & Active Immersion): Dili bir ders konusu olarak görmek yerine günlük hayatın doğal bir parçası haline getirmek gerekir. "
            "İngilizce podcast'ler dinlemek, alt yazısız videolar izlemek ve düşünce dilini kademeli olarak İngilizceye çevirmek dil edinimini hızlandırır. "
            "3. Bağlam İçi Kelime Öğrenimi ve Aralıklı Tekrar (Spaced Repetition): Kelime listelerini ezberlemek yerine kelimeleri içinde geçtikleri cümleler ve gerçek hikayelerle öğrenmek gerekir. "
            "Öğrenilen kelimelerin unutulmaması için Anki gibi kart uygulamaları ile aralıklı tekrar yöntemini uygulamak kalıcılığı garantiler. "
            "4. Konuşma Pratiği ve Shadowing Tekniği: Sadece dinleme (input) yapmak dili akıcı hale getirmez. Shadowing tekniği ile anadili İngilizce olan kişilerin taklit edilmesi, kendi sesini kaydedip dinleme ve günlük konuşma pratikleri ile C2 seviyesine ulaşılır. "
        ) * 10
    },
    "demo3": {
        "title": "Neural Networks & Deep Learning",
        "channel": "3Blue1Brown (İngilizce)",
        "url": "https://www.youtube.com/watch?v=aircAruvnKk",
        "thumb": "https://img.youtube.com/vi/aircAruvnKk/hqdefault.jpg",
        "duration_sec": 1119,
        "timestamps": [
            {"start": 0, "duration": 180, "text": "Introduction to Neural Networks and Deep Learning fundamentals by 3Blue1Brown."},
            {"start": 180, "duration": 220, "text": "1. What is a Neuron?: A neuron is a mathematical function holding a number between 0 and 1, representing activation levels."},
            {"start": 400, "duration": 240, "text": "2. Network Layers & Weights: Input layers receive raw pixels or data, passing them through weighted connections and biases to hidden layers."},
            {"start": 640, "duration": 240, "text": "3. Gradient Descent & Loss Function: The cost function measures how wrong the network is. Gradient descent adjusts weights to minimize error."},
            {"start": 880, "duration": 239, "text": "4. Backpropagation Algorithm: Backpropagation efficiently calculates partial derivatives to tune millions of parameters layer by layer."}
        ],
        "text": (
            "Welcome to the deep dive into Neural Networks and Deep Learning by 3Blue1Brown. "
            "1. What is a Neuron and Neural Network Structure: A neuron in deep learning is a mathematical node holding an activation value between 0 and 1. "
            "Networks consist of an input layer, multiple hidden layers, and an output layer that classifies complex visual inputs like handwritten digits. "
            "2. Weights, Biases, and Activation Functions: Connections between neurons have weights that amplify or suppress signals, while biases determine threshold activation. "
            "Non-linear activation functions like Sigmoid and ReLU enable neural networks to learn complex non-linear patterns. "
            "3. Cost Function and Optimization via Gradient Descent: The cost (loss) function quantifies the performance error of the network. "
            "Gradient descent calculates the direction of steepest decline in the high-dimensional cost landscape to continuously optimize weights. "
            "4. Backpropagation Algorithm: Backpropagation is the mathematical engine of deep learning. It uses the chain rule of calculus to compute gradient contributions from output layers back to input layers effectively. "
        ) * 10
    }
}


# ─────────────────────────────────────────────────────
# GELİŞTİRİLMİŞ RAG SİSTEMİ
# ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sen bir YouTube video analisti ve araştırmacısısın.
Görevin, video transkript parçalarına dayanarak kullanıcının sorularını doğru ve anlaşılır şekilde Türkçe yanıtlamaktır.

KURALLAR:
1. SADECE verilen bağlam metninde bulunan bilgilere dayanarak yanıt ver.
2. Bağlamda bulunmayan bilgiyi ASLA uydurma — "Bu konuda videoda bilgi bulunmamaktadır" de.
3. Yanıtını açık ve düzenli formatla (gerektiğinde madde işaretleri kullan).
4. DİL KURALI: SADECE KUSURSUZ VE DOĞAL TÜRKÇE KULLAN. Eğer kaynak metin İngilizce ise, İngilizce kelimeleri doğrudan kopyalayıp sonuna Türkçe ek getirme (Örnek YANLIŞ: "conceptü", "inspire etse de"). Mutlaka tüm cümleleri ve terimleri Türkçe'ye çevir. Gerekirse orijinal terimi parantez içinde verebilirsin.
5. Eğer önceki konuşma geçmişi varsa, bağlamı dikkate al ve takip sorularını anlamlı şekilde yanıtla."""

SUMMARY_PROMPT = """Sen bilgi yoğunluğu yüksek ve akademik düzeyde Türkçe içerik analizi yapan bir AI uzmansın.
Aşağıdaki video transkriptini inceleyerek 'Kapsamlı, Zaman Damgalı ve Yüksek Bilgi Yoğunluklu Tematik Özet' formatında özetle. Yüzeysel bir özet kesinlikle istemiyorum; videodaki teknik nüansları, verilen örnekleri ve arka plandaki temel mantığı atlamadan, bilgi yoğunluğu son derece yüksek bir çıktı oluşturmalısın. Çıktının tam olarak şu kurallara uymasını istiyorum:

1. Kapsamlı Giriş Paragrafı (En az 3-4 cümle): İçeriğin kime ait olduğunu, ana konusunu, genel amacını anlatan doyurucu bir giriş yaz.
2. Temel Konu Başlıkları: İçeriği 4 veya 5 ana temaya böl. Her birini numaralandırılmış kalın başlıklar halinde yaz (Örn: 1. Büyük Dil Modelleri Nedir?). Madde İmleri'nde başlıklar için birkaç cümle açıklama da ekle.
3. Yoğun Alt Başlıklar ve Madde İmleri: Her ana başlığın altına o konunun detaylarını madde imleri ile yaz. ÖNEMLİ: Her bir madde iminin karşısındaki açıklama en az 3 ila 5 detaylı cümleden oluşmalıdır. Geçilen hiçbir önemli örneği veya kavramı atlama.
4. Analitik Sonuç (En az 3 cümle): Tüm içeriğin ana mesajını toparlayan, geleceğe dair öngörüleri veya çıkarımları içeren ve 'Özetle,' kelimesi ile başlayan kapsamlı bir kapanış paragrafı yaz.
5. Ton ve Dil: Objektif, akademik düzeyde bilgilendirici ve okuması kolay bir dil kullan. Cümleleri basit tutma, bağlaçlarla zenginleştirilmiş uzun ve doyurucu cümleler kur.

ÖRNEK ÇIKTI (BU UZUNLUĞU, BİLGİ YOĞUNLUĞUNU VE FORMATI BİREBİR TAKLİT ET):

Bu metin, yapay zeka alanının önde gelen isimlerinden Andrej Karpathy tarafından hazırlanan ve Büyük Dil Modellerinin (LLM) çalışma prensiplerini derinlemesine inceleyen bir sunumun kapsamlı özetidir. Yaklaşık bir saat süren bu eğitim videosunda, modellerin eğitim aşamalarından başlayarak gelecekteki potansiyel kullanım senaryolarına ve güvenlik zafiyetlerine kadar geniş bir yelpazede teknik bilgiler sunulmaktadır (URL: https://www.youtube.com/watch?v=zjkBMFhNj_g). Karpathy'nin temel amacı, karmaşık görünen bu yapay zeka sistemlerini sıradan bir yazılım paradigması üzerinden açıklayarak, teknoloji dünyasında gerçekleşmekte olan "yeni işletim sistemi" devrimini izleyicilere net bir şekilde aktarmaktır.

**1. Büyük Dil Modellerinin (LLM) Temel Anatomisi ve Doğası**
Bu bölümde modellerin temel fiziksel yapısı ve çalışma mantığı ele alınmaktadır.
* **Dosya Yapısı ve Sıkıştırma Paradigması:** Bir büyük dil modelinin özünde, sadece iki temel dosyadan oluşan oldukça minimalist bir yapı yatmaktadır; milyarlarca parametreyi barındıran devasa bir ağırlık dosyası ve bu parametreleri çalıştıran nispeten küçük hacimli bir kod dosyası. Karpathy, Llama-2 70B modeli üzerinden verdiği örnekte, modelin internetteki devasa veriyi bir tür "kayıplı sıkıştırma" yöntemiyle bu parametrelerin içine hapsettiğini teknik bir dille ifade eder. Modelin temel görevi "bir sonraki kelimeyi tahmin etmek" gibi basit bir işlemmiş gibi görünse de, bu işlemi yüksek doğrulukla yapabilmek için internet üzerindeki tüm bağlamsal ve dünyevi bilgiyi kendi sinir ağı mimarisinde sentezlemek zorundadır.
* **Tahmin ve Halüsinasyon Mekanizması:** Modellerin metin üretimi aslında daha önce öğrendiği verilerin istatistiksel bir rüyasını görme işlemine (dreaming internet documents) benzetilmektedir. Örneğin, bir ürün sayfası oluşturması istendiğinde model daha önce hiç var olmamış bir ISBN numarası veya ürün kodu üretebilir; çünkü model veriyi birebir ezberlemek yerine, o formatın yapısal bir kopyasını öğrenmiştir. Bu durum, modellerin hem son derece yaratıcı metinler üretmesini sağlarken hem de "halüsinasyon" adı verilen, ikna edici ancak tamamen yanlış bilgiler sunma riskini doğuran temel istatistiksel bir özelliktir.

(Diğer 2., 3., 4. başlıklar da aynı uzunluk ve derinlikte devam edecektir...)

Özetle, Andrej Karpathy'nin bu ufuk açıcı sunumu, büyük dil modellerinin yalnızca istatistiksel bir kelime bulmaca aracı olmaktan çıkarıp onları geleceğin teknoloji ekosistemini yönetecek merkezi işletim sistemleri olarak konumlandırmaktadır. Trilyonlarca verinin sıkıştırılmasıyla başlayan bu süreç, modellerin harici araçları kullanabilme ve çok boyutlu düşünme (Sistem 2) kapasitesi kazanmasıyla bilişsel bir devrime dönüşmektedir. Bununla birlikte, bu eşi benzeri görülmemiş yeteneklerin veri zehirlenmesi ve istem enjeksiyonu gibi yepyeni siber güvenlik sorunlarını da beraberinde getirdiği, geleceğin teknoloji mimarisinde donanım veya yazılım güvenliğinden ziyade "bağlam ve komut güvenliğinin" en büyük sınavımız olacağı açıkça ortaya konmaktadır.

YUKARIDAKİ ÖRNEK GİBİ UZUN, DERİN VE BİLGİ YOĞUNLUKLU BİR ÖZET ÇIKAR. 

TRANSKRİPT:
{text}"""


def _build_chat_context(messages: list, max_history: int = 4) -> str:
    """Son N mesajı konuşma geçmişi olarak formatlar."""
    if not messages:
        return ""
    recent = messages[-max_history:]
    lines = []
    for msg in recent:
        role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
        lines.append(f"{role}: {msg['content'][:300]}")
    return "\n".join(lines)


def generate_real_rag_response(query: str, chat_history: list = None) -> tuple:
    """
    Geliştirilmiş RAG yanıtı — chat geçmişi + skor tabanlı kaynaklar.
    Dönüş: (yanıt_metni, kaynak_listesi)
    """
    # Skor tabanlı arama
    search_results = search_in_db(query, k=4)

    # Bağlam metni oluştur (skor bilgisiyle)
    context_parts = []
    for i, r in enumerate(search_results, 1):
        time_info = ""
        if "time_label" in r.get("metadata", {}):
            time_info = f" [⏱ {r['metadata']['time_label']}]"
        context_parts.append(f"[Kaynak {i}{time_info}]\n{r['content']}")
    context_text = "\n\n".join(context_parts)

    # Chat geçmişi
    history_text = ""
    if chat_history:
        history_str = _build_chat_context(chat_history)
        if history_str:
            history_text = f"\n\nÖNCEKİ KONUŞMA:\n{history_str}\n"

    prompt = f"""{SYSTEM_PROMPT}
{history_text}
BAĞLAM (Video Transkript Parçaları):
{context_text}

KULLANICININ SORUSU: {query}

YANIT:"""

    llm = Ollama(model="qwen2.5:3b")
    response = llm.invoke(prompt)
    return response, search_results


def generate_summary(text: str) -> str:
    """Derinlemesine, uzun ve detaylı video analiz raporu üretir."""
    llm = Ollama(model="qwen2.5:3b")
    # Transkriptin tamamını kapsamak için 30.000 karaktere kadar oku
    prompt = SUMMARY_PROMPT.format(text=text[:30000])
    return llm.invoke(prompt)


# ─────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────
for key, default in [("messages", []), ("is_processed", False),
                      ("video_data", {}), ("active_tab", "yaziya_dokme"),
                      ("cached_summary", None), ("timestamps", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================
# EKRAN 1 — LANDING PAGE
# =====================================================
if not st.session_state.is_processed:

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">✨ AI Destekli Video Analizi</div>
        <h1 class="hero-title">
            YouTube Videolarını<br/>
            <span class="blue">Anında Özetle</span>
        </h1>
        <p class="hero-desc">
            Herhangi bir YouTube videosunun transkriptini çıkarın, yapay zeka ile özetleyin
            ve video içeriği hakkında akıllı sorular sorun.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pill Search Bar — max 640px ortalı
    _spacer1, pill_col, _spacer2 = st.columns([2, 3, 2])
    with pill_col:
        st.markdown('<div class="pill-search-wrapper">', unsafe_allow_html=True)
        in_col, btn_col = st.columns([4, 1.2])
        with in_col:
            yt_url = st.text_input("url", placeholder="YouTube video bağlantısını yapıştırın...", key="landing_url")
        with btn_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            summarize_btn = st.button("Özetle →", type="primary", key="btn_sum", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Örnek Videolar
    st.markdown('<div class="section-title">Örnek videolarla deneyin</div>', unsafe_allow_html=True)

    _s1, cards_col, _s2 = st.columns([1, 5, 1])
    with cards_col:
        c1, c2, c3 = st.columns(3, gap="large")
        for col, key in zip([c1, c2, c3], ["demo1", "demo2", "demo3"]):
            d = DEMO_VIDEOS[key]
            with col:
                play_svg = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>'
                st.markdown(f"""
                <div class="vcard">
                    <img src="{d['thumb']}" />
                    <div class="vcard-body">
                        <div class="vcard-title">{d['title']}</div>
                        <div class="vcard-channel">{d['channel']}</div>
                    </div>
                    <div class="vcard-play">{play_svg}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"▶ Özetle", key=f"btn_{key}", type="primary", use_container_width=True):
                    with st.spinner("İndeksleniyor..."):
                        create_vector_db(d['text'], video_title=d['title'])
                        st.session_state.is_processed = True
                        st.session_state.video_data = d
                        st.session_state.messages = []
                        st.session_state.cached_summary = None
                        st.session_state.timestamps = None
                        st.rerun()

    # Özellik Kartları
    st.markdown("""
    <div class="features-row">
        <div class="feat-card">
            <div class="feat-icon">📝</div>
            <div class="feat-title">Akıllı Transkript</div>
            <div class="feat-desc">Zaman damgalı, düzenli ve okunabilir transkript çıkarımı</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🧠</div>
            <div class="feat-title">AI Özet</div>
            <div class="feat-desc">Qwen 2.5 ile kapsamlı ve doğru video özetleme</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">💬</div>
            <div class="feat-title">Video Chatbot</div>
            <div class="feat-desc">RAG tabanlı akıllı soru-cevap sistemi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # URL İşleme
    if summarize_btn:
        url_str = st.session_state.get("landing_url", "").strip()
        if not url_str or not url_str.startswith("http"):
            st.error("❌ Lütfen geçerli bir YouTube URL'si yapıştırın.")
        else:
            # Zaman damgalı transkript dene
            with st.spinner("Altyazılar çekiliyor..."):
                ts_result = get_timestamped_transcript(url_str)

            if isinstance(ts_result, str):  # Hata mesajı
                st.error(ts_result)
                st.info("💡 YouTube bot engeli nedeniyle yukarıdaki örnek videolarla deneyin.")
            else:
                transcript = " ".join(item["text"] for item in ts_result)
                with st.spinner("Vektör veritabanı oluşturuluyor..."):
                    create_vector_db(transcript, video_title="YouTube Videosu",
                                     timestamps=ts_result)
                st.session_state.is_processed = True
                st.session_state.video_data = {
                    "title": "YouTube Videosu", "channel": "YouTube",
                    "url": url_str, "thumb": "", "text": transcript
                }
                st.session_state.timestamps = ts_result
                st.session_state.messages = []
                st.session_state.cached_summary = None
                st.rerun()


# =====================================================
# EKRAN 2 — WORKSPACE (İki Kolonlu, Tabs)
# =====================================================
else:
    vdata = st.session_state.video_data

    # Ghost Geri Butonu
    if st.button("← Başka bir videoyu özetle", key="btn_back"):
        st.session_state.is_processed = False
        st.session_state.messages = []
        st.session_state.video_data = {}
        st.session_state.cached_summary = None
        st.session_state.timestamps = None
        st.rerun()

    # Video başlık bar
    st.markdown(f"""
    <div class="ws-bar">
        <div class="ws-bar-title">🎥 {vdata.get('title', 'Video')}</div>
    </div>
    """, unsafe_allow_html=True)

    # İki Sütun: Sol %42 | Sağ %58
    col_left, col_right = st.columns([42, 58], gap="large")

    # ── SOL: Video + Bilgi ──
    with col_left:
        try:
            st.video(vdata.get("url", ""))
        except:
            if vdata.get("thumb"):
                st.image(vdata["thumb"], use_container_width=True)

        # Video Bilgi
        st.markdown(f"""
        <div class="vid-info">
            <div class="vid-info-title">{vdata.get('title','')}</div>
            <div class="vid-info-ch">📺 {vdata.get('channel','')}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SAĞ: Modern Tabs ──
    with col_right:
        tab_transcript, tab_summary, tab_chat = st.tabs(["📄 Transkript", "✏️ AI Özeti", "💬 AI Chatbot"])

        # TAB 1: Transkript (gerçek zaman damgaları varsa kullanır)
        with tab_transcript:
            ts_data = st.session_state.timestamps

            with st.container(height=520):
                if ts_data and isinstance(ts_data, list):
                    group_size = 5
                    for gi in range(0, len(ts_data), group_size):
                        group = ts_data[gi:gi + group_size]
                        start_time = format_timestamp(group[0]["start"])
                        end_sec = group[-1]["start"] + group[-1].get("duration", 0)
                        end_time = format_timestamp(end_sec)
                        group_text = " ".join(item["text"] for item in group)
                        st.markdown(f"""<div class="t-chunk">
                            <div class="t-time">⏱ {start_time} — {end_time}</div>
                            <div class="t-text">{group_text}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    text_full = vdata.get("text", "")
                    import re
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_full) if s.strip()]
                    
                    chunks = []
                    current_chunk = []
                    word_count = 0
                    for s in sentences:
                        current_chunk.append(s)
                        word_count += len(s.split())
                        if word_count >= 40:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = []
                            word_count = 0
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    
                    total_duration = vdata.get("duration_sec", len(chunks) * 45)
                    chunk_duration = total_duration / max(1, len(chunks))
                    
                    sec = 0.0
                    for chunk_str in chunks:
                        start_time = format_timestamp(sec)
                        sec += chunk_duration
                        end_time = format_timestamp(sec)
                        st.markdown(f"""<div class="t-chunk">
                            <div class="t-time">⏱ {start_time} — {end_time}</div>
                            <div class="t-text">{chunk_str}</div>
                        </div>""", unsafe_allow_html=True)

        # TAB 2: AI Özeti (önbellekli)
        with tab_summary:
            if st.session_state.cached_summary:
                st.markdown(f"""<div class="sum-box">
                    <div class="sum-label">📌 Video Özeti</div>
                    <div class="sum-text">{st.session_state.cached_summary}</div>
                </div>""", unsafe_allow_html=True)
                col_cap, col_ref = st.columns([3, 1])
                with col_cap:
                    st.caption("💾 Önbellekten yüklendi")
                with col_ref:
                    if st.button("🔄 Yeniden Özetle", key="btn_refresh_sum"):
                        st.session_state.cached_summary = None
                        st.rerun()
            else:
                with st.spinner("🧠 Qwen 2.5 yapılandırılmış özet oluşturuyor..."):
                    try:
                        resp = generate_summary(vdata.get('text', ''))
                        st.session_state.cached_summary = resp
                        st.markdown(f"""<div class="sum-box">
                            <div class="sum-label">📌 Video Özeti</div>
                            <div class="sum-text">{resp}</div>
                        </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"⚠️ Hata: {e}")

        # TAB 3: Chatbot (chat geçmişi + skor tabanlı kaynaklar)
        with tab_chat:
            for msg in st.session_state.messages:
                avatar = "👤" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander("📚 Kaynak Parçaları (skor ve zaman bilgisiyle)"):
                            for i, src in enumerate(msg["sources"], 1):
                                meta = src.get("metadata", {})
                                score = src.get("score", 0)
                                time_lbl = meta.get("time_label", "")
                                score_pct = f"{score*100:.0f}%" if score else ""
                                header = f"**{i}.** "
                                if time_lbl:
                                    header += f"⏱ {time_lbl} "
                                if score_pct:
                                    header += f"| 🎯 {score_pct} ilgili"
                                st.markdown(header)
                                st.markdown(f"> {src.get('content', '')[:350]}...")

            if user_input := st.chat_input("Bu video hakkında soru sorun..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_input)
                with st.chat_message("assistant", avatar="🤖"):
                    ph = st.empty()
                    with st.spinner("🧠 Qwen 2.5 düşünüyor..."):
                        try:
                            resp, srcs = generate_real_rag_response(
                                user_input,
                                chat_history=st.session_state.messages[:-1]
                            )
                            ph.markdown(resp)
                            with st.expander("📚 Kaynak Parçaları (skor ve zaman bilgisiyle)"):
                                for i, src in enumerate(srcs, 1):
                                    meta = src.get("metadata", {})
                                    score = src.get("score", 0)
                                    time_lbl = meta.get("time_label", "")
                                    score_pct = f"{score*100:.0f}%" if score else ""
                                    header = f"**{i}.** "
                                    if time_lbl:
                                        header += f"⏱ {time_lbl} "
                                    if score_pct:
                                        header += f"| 🎯 {score_pct} ilgili"
                                    st.markdown(header)
                                    st.markdown(f"> {src.get('content', '')[:350]}...")
                            st.session_state.messages.append(
                                {"role": "assistant", "content": resp, "sources": srcs}
                            )
                        except Exception as e:
                            ph.error(f"⚠️ {e}")
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"⚠️ {e}", "sources": []}
                            )
