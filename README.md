# 🎥 YouTube Video Özetleyici ve RAG Tabanlı Araştırma Asistanı

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-5A67D8)](https://www.trychroma.com/)
[![pytest](https://img.shields.io/badge/pytest-30%20passed-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-Unspecified-lightgrey)](#lisans)

Herhangi bir YouTube videosunun transkriptini çıkaran, çok adımlı prompt mühendisliğiyle akademik derinlikte bir özet üreten ve videoya dair soruları **RAG (Retrieval-Augmented Generation)** mimarisiyle, kaynak alıntılı olarak yanıtlayan, tamamen yerel çalışan bir Streamlit uygulaması.

Tüm çıkarım (inference) [Ollama](https://ollama.com/) üzerinden, kullanıcının kendi makinesinde yapılır — hiçbir transkript, soru veya yanıt üçüncü taraf bir API'ye gönderilmez.

---

## İçindekiler

- [Mimari](#mimari)
- [Özellikler](#özellikler)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Yapılandırma Referansı](#yapılandırma-referansı)
- [Testler](#testler)
- [Veri Akışı — Teknik Detay](#veri-akışı--teknik-detay)
- [Mühendislik Notları ve Bilinen Kısıtlar](#mühendislik-notları-ve-bilinen-kısıtlar)
- [Sorun Giderme](#sorun-giderme)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

---

## Mimari

```mermaid
flowchart LR
    U["Kullanıcı\n(YouTube URL)"] --> FT["fetch_transcript.py"]
    FT -->|"1. caption API"| YT[("youtube-transcript-api")]
    FT -->|"2. fallback: ASR"| ASR["yt-dlp + faster-whisper"]
    FT --> TXT["Zaman damgalı\ntranskript"]
    TXT --> RAG["rag_engine.py"]
    RAG -->|"chunk + embed"| EMB[("nomic-embed-text\nvia Ollama")]
    EMB --> VDB[("ChromaDB\n(cosine, video-başına koleksiyon)")]
    TXT --> SUM["app.py: generate_summary()"]
    SUM --> LLM[("qwen2.5:3b\nvia Ollama")]
    U -->|"soru"| QRY["app.py: generate_real_rag_response()"]
    QRY --> VDB
    VDB -->|"top-k, skor + dedup"| QRY
    QRY --> LLM
    LLM --> UI["Streamlit UI"]
    SUM --> UI
```

**Kritik tasarım kararları:**
- **Her video kendi ChromaDB koleksiyonunda** (`yt_<sha1-16>`), URL'den deterministik olarak türetilir — aynı video yeniden gönderilirse yeniden embed edilmez, önbellekten kullanılır.
- **Cosine mesafe uzayı** (`hnsw:space: cosine`) açıkça ayarlanır; relevance skoru `1 - distance` olarak hesaplanır (keyfi bir sabite bölme yerine).
- **İki aşamalı transkript kaynağı:** önce YouTube'un caption API'si, başarısız olursa (altyazı kapalı/bot engeli) `yt-dlp` ile ses indirilip yerel `faster-whisper` ile ASR fallback'i devreye girer.
- **`num_ctx` açıkça ayarlanır** (Ollama'da varsayılan 2048 token'dır) — aksi halde uzun transkript + prompt kombinasyonu sessizce kırpılır.

---

## Özellikler

| Özellik | Açıklama |
|---|---|
| **Zaman damgalı transkript** | Caption API → ASR fallback zinciri; transkript sekmesinde içerik ~60 saniyelik zaman pencerelerine göre (sabit öğe sayısına göre değil) gruplanır. |
| **Çok dilli altyazı desteği** | `tr, en, es, de, fr, ja, ko` öncelik sırasıyla denenir; hiçbiri yoksa çevrilebilir herhangi bir altyazı Türkçe'ye çevrilir. |
| **Akademik derinlikte özet** | Few-shot prompting ile 4-5 temaya ayrılmış, zaman damgalı analiz raporu; `num_ctx`/`num_predict` açıkça ayarlanmış, karakter sınırını aşan transkriptlerde kullanıcı uyarılır. |
| **RAG tabanlı chatbot** | Cosine benzerlik + relevance threshold (0.3) + ilk-80-karakter dedup; kaynaklar tıklanınca açılan (`<details>/<summary>`) alıntı çipleri olarak gösterilir. |
| **Çoklu video + kalıcı oturum** | Her video kendi koleksiyonunda; sohbet geçmişi ve özet video bazında önbellekte tutulur (`video_sessions`), aynı video tekrar açıldığında geri yüklenir. |
| **Hata toleransı** | Bot engeli, altyazısız video, Ollama bağlantı hatası, zaman aşımı (300sn) için ayrı, anlaşılır hata mesajları. |
| **Güvenli render** | LLM/kullanıcı çıktısı `html.escape` → `markdown` zinciriyle işlenir (XSS'e karşı güvenli), gerçek `<ul>/<strong>/<h2>` render edilir. |
| **Koyu tema arayüz** | Geist + Instrument Serif tipografi, segmented-control sekmeler, CSS-only hareketli arka plan (prefers-reduced-motion uyumlu). |

---

## Proje Yapısı

```text
.
├── app.py                   # Streamlit UI, prompt şablonları, oturum durumu, CSS
├── fetch_transcript.py      # YouTube transkript çıkarma (caption API + ASR fallback)
├── rag_engine.py            # Chunking, embedding, ChromaDB CRUD, arama/skor/dedup
├── requirements.txt         # Üretim bağımlılıkları
├── requirements-dev.txt     # + pytest
├── tests/
│   ├── conftest.py              # requires_ollama marker, isolated_chroma fixture
│   ├── test_fetch_transcript.py # URL ayrıştırma, format_timestamp, proxy (ağ gerektirmez)
│   ├── test_rag_engine.py       # koleksiyon adı, zaman damgası eşleştirme (ağ gerektirmez)
│   └── test_rag_engine_integration.py  # gerçek Ollama embedding ile create/search/dedup
├── pytest.ini
├── chroma_db/                # Kalıcı vektör verisi (git'e dahil değil)
├── .streamlit/config.toml    # Streamlit tema ayarları
└── .gitignore
```

### Modül Sorumlulukları

**`app.py`** (tek dosyalık Streamlit uygulaması) şu sorumlulukları taşır:
- Sayfa durumu yönetimi (`st.session_state`): `is_processed`, `video_data`, `collection_name`, `messages`, `video_sessions` (video başına kalıcı sohbet/özet önbelleği).
- `generate_synthetic_timestamps()` — gerçek zaman damgası olmayan içerikler için cümle sınırına duyarlı, süreye oranlanmış yaklaşık zaman damgaları üretir.
- `render_markdown_html()` — LLM çıktısını güvenli HTML'e çevirir (`_ensure_blank_line_before_lists` ile Markdown'ın liste-öncesi-boş-satır gereksinimini telafi eder).
- Prompt şablonları: `SYSTEM_PROMPT` (chatbot), `SUMMARY_PROMPT` (few-shot, akademik özet formatı).
- `generate_summary()`, `generate_real_rag_response()` — `langchain_ollama.OllamaLLM` üzerinden çıkarım, açık `num_ctx`/`num_predict`/`client_kwargs={"timeout": ...}`.

**`fetch_transcript.py`**:
- `_extract_video_id()` — `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/` formatlarını ayrıştırır.
- `_fetch_raw_transcript()` — öncelikli dil listesi → auto-generated altyazı → çevrilebilir herhangi bir altyazı zinciri.
- `_fetch_transcript_via_asr()` — `yt-dlp` ile `bestaudio` indirir, `faster-whisper` (`small`, `int8`, CPU) ile transkribe eder.
- `get_timestamped_transcript()` — yukarıdaki zinciri `try/except` ile sarar, her aşamada anlaşılır Türkçe hata mesajı döner.
- `format_timestamp()` — `MM:SS`, 1 saat ve üzeri için `H:MM:SS`.

**`rag_engine.py`**:
- `_make_collection_name(source_key)` — `sha1(source_key)[:16]` → `yt_<hash>`, deterministik ve tekrar işlenmeyi önler.
- `create_vector_db()` — `RecursiveCharacterTextSplitter` (500/100, cümle-duyarlı ayırıcılar) ile chunking; `collection_metadata={"hnsw:space": "cosine", "created_at": ...}`; `MAX_COLLECTIONS` üstü en eski koleksiyonları temizler (`_cleanup_old_collections`, `_purge_orphaned_segment_dirs`).
- `search_in_db()` — `k*4` aday çeker, `relevance = 1 - cosine_distance` hesaplar, `RELEVANCE_THRESHOLD` altını eler, ilk-80-karaktere göre dedup yapar, `k`'ya kırpar.

---

## Kurulum

### Gereksinimler
- Python 3.9+
- [Ollama](https://ollama.com/download) (kurulu ve çalışır durumda)
- `ffmpeg` — yalnızca ASR fallback'i (Whisper) tetiklenirse gerekir

### Adımlar

```bash
# 1. Ollama modellerini indirin
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 2. Depoyu klonlayın ve sanal ortam kurun
git clone https://github.com/ctnasu/YouTube-Video-ozetleyici-ve-RAG-Tabanli-Arastirma-Asistan-.git
cd "YouTube Video ozetleyici ve RAG Tabanli Arastirma Asistan"
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

# 3. Bağımlılıkları kurun
pip install -r requirements.txt
```

**Daha güçlü bir makineniz varsa** (≥16GB RAM önerilir): `app.py` içindeki `LLM_MODEL = "qwen2.5:3b"` satırını `"qwen2.5:7b"` yapın. Bu geliştirme makinesinde (8GB RAM) 7B modeli uzun oturumlarda Ollama'nın sessizce yanıt vermeyi bırakmasına yol açtığı için 3B varsayılan yapıldı — bkz. [Mühendislik Notları](#mühendislik-notları-ve-bilinen-kısıtlar).

---

## Çalıştırma

```bash
# Ollama servisinin çalıştığından emin olun
curl -s http://localhost:11434/api/tags   # boş dönerse: ollama serve

streamlit run app.py
# http://localhost:8501
```

### Kullanım Akışı
1. Giriş sayfasına bir YouTube URL'si yapıştırın **veya** sağlanan 3 örnek videodan birini seçin.
2. Transkript çekilir (ya da örnek videolar için sentetik zaman damgaları üretilir) ve ChromaDB'ye vektörleştirilir.
3. **Transkript** sekmesinde ~60sn'lik zaman damgalı bloklar halinde içeriği inceleyin.
4. **Akıllı Özet** sekmesinde "Özeti Oluştur"a basarak 1-4 dakikada kapsamlı bir rapor alın.
5. **AI Chatbot** sekmesinde soru sorun; her yanıtın altında tıklanınca açılan, relevance yüzdeli kaynak alıntıları bulunur.
6. "← Geri Dön" ile başka bir videoya geçin — aynı videoyu tekrar seçerseniz sohbet/özet kaldığı yerden devam eder.

---

## Yapılandırma Referansı

### LLM (`app.py`)

| Sabit | Varsayılan | Açıklama |
|---|---|---|
| `LLM_MODEL` | `"qwen2.5:3b"` | Ollama model adı. 7B+ için RAM uyarısına bakın. |
| `SUMMARY_NUM_CTX` | `8192` | Özet üretimi context penceresi (token). `16384`'te bu makinede ölçülen hız kaybı: ~5x. |
| `SUMMARY_NUM_PREDICT` | `2048` | Özet için maksimum üretilecek token. |
| `SUMMARY_TEXT_CHAR_CAP` | `12000` | Prompt'a dahil edilecek maksimum transkript karakteri; aşılırsa kullanıcı uyarılır. |
| `CHAT_NUM_CTX` | `8192` | Chatbot context penceresi. |
| `CHAT_NUM_PREDICT` | `768` | Chatbot yanıtı için maksimum token. |
| `LLM_REQUEST_TIMEOUT_SEC` | `300` | Ollama isteği için üst sınır (bkz. aşağıdaki not). |
| `TRANSCRIPT_WINDOW_SEC` | `60` | Transkript sekmesinde hedeflenen blok süresi. |

### RAG (`rag_engine.py`)

| Sabit | Varsayılan | Açıklama |
|---|---|---|
| `CHROMA_PATH` | `"chroma_db"` | Kalıcı ChromaDB dizini. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | `RecursiveCharacterTextSplitter` parametreleri (karakter). |
| `RELEVANCE_THRESHOLD` | `0.3` | Bu skorun altındaki arama sonuçları elenir (fiilen uygulanır, bkz. `search_in_db`). |
| `MAX_COLLECTIONS` | `15` | Bu sayıyı aşan en eski koleksiyonlar otomatik silinir. |

### Ortam Değişkenleri

| Değişken | Açıklama |
|---|---|
| `YT_PROXY_URL` | `http://kullanici:parola@proxy-host:port` — YouTube bot engeline takılan kullanıcılar için (hem caption API hem yt-dlp isteklerine uygulanır). |

```bash
export YT_PROXY_URL="http://kullanici:parola@proxy.example.com:8080"
streamlit run app.py
```

---

## Testler

```bash
pip install -r requirements-dev.txt
pytest -q                                           # tümü
pytest tests/test_fetch_transcript.py tests/test_rag_engine.py   # sadece ağ gerektirmeyenler
pytest tests/test_rag_engine_integration.py          # gerçek Ollama embedding
```

| Dosya | Kapsam | Ollama gerekli mi? |
|---|---|---|
| `test_fetch_transcript.py` | URL ayrıştırma, `format_timestamp`, proxy config, hata mesajları | Hayır |
| `test_rag_engine.py` | `_make_collection_name`, `_find_chunk_timestamp`, hata toleransı | Hayır |
| `test_rag_engine_integration.py` | `create_vector_db`/`search_in_db` uçtan uca (gerçek embedding, izole `chroma_db`) | Evet — çalışmıyorsa `conftest.requires_ollama` ile otomatik atlanır |

Son çalıştırma: **30/30 geçti** (4 entegrasyon testi dahil, yerel Ollama ile).

---

## Veri Akışı — Teknik Detay

1. **URL → video ID**: `_extract_video_id()`, dört URL formatını destekler.
2. **Transkript**: `YouTubeTranscriptApi.get_transcript(video_id, languages=SUPPORTED_LANGUAGES)` → başarısızsa `list_transcripts()` üzerinden auto-generated/çevrilebilir altyazı denenir → o da başarısızsa `yt-dlp` ile ses indirilip `faster-whisper` ile ASR yapılır.
3. **Chunking**: Tam transkript metni `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n","\n",". ","! ","? ","; ",", "," "])` ile bölünür.
4. **Zaman damgası eşleştirme**: Her chunk'ın ilk 60 karakteri, orijinal zaman damgalı segmentlerde kümülatif aranarak `start_time`/`time_label` metadata'sına bağlanır.
5. **Embedding + depolama**: `OllamaEmbeddings(model="nomic-embed-text")` ile vektörleştirilip `Chroma.from_texts(..., collection_metadata={"hnsw:space": "cosine"})` ile ChromaDB'ye yazılır. Koleksiyon adı zaten varsa (`client.get_collection`) bu adım tamamen atlanır.
6. **Arama**: Kullanıcı sorgusu embed edilir, `similarity_search_with_score(query, k=k*4)` ile fazladan aday çekilir; `relevance = 1 - distance` hesaplanıp `RELEVANCE_THRESHOLD` altı elenir, ilk 80 karaktere göre dedup yapılır, `k`'ya kırpılır.
7. **Üretim**: Seçilen `k=4` kaynak + son 4 mesajlık sohbet geçmişi, `SYSTEM_PROMPT` ile birleştirilip `OllamaLLM.invoke()` çağrılır.
8. **Render**: Yanıt `html.escape()` → `_ensure_blank_line_before_lists()` → `markdown.markdown(extensions=["nl2br"])` zincirinden geçip tek bir `unsafe_allow_html=True` çağrısıyla basılır.

---

## Mühendislik Notları ve Bilinen Kısıtlar

Aşağıdaki maddeler, kod incelemesiyle fark edilmeyecek, yalnızca çalışma zamanında (ölçüm veya canlı test ile) ortaya çıkan kararlardır — şeffaflık için belgelenmiştir.

| # | Konu | Kök Neden | Çözüm/Durum |
|---|---|---|---|
| 1 | Sessizce kırpılan context | Ollama `num_ctx` verilmezse 2048 token'da kalır | `num_ctx`/`num_predict` her çağrıda açıkça ayarlanıyor |
| 2 | `num_ctx=16384`'te %80 hız kaybı | Bu makinede (Apple Silicon) KV cache hızlı bellek havuzuna sığmıyor (ölçüldü: ~16→~3 tok/s) | `num_ctx=8192`'de sabitlendi |
| 3 | **8GB RAM'de Ollama'nın süresiz takılması** | `qwen2.5:7b` (~5GB) uzun oturumda sistem belleğini zorluyor; Ollama hatasız şekilde süresiz yanıt vermeyi bırakabiliyor (doğrudan `curl` ve izole script ile doğrulandı) | Varsayılan model `qwen2.5:3b`'ye düşürüldü + tüm çağrılara 300sn timeout eklendi. **Bu, projenin en büyük açık ödünleşimidir** — küçük model özet kalitesinde sınırlıdır. |
| 4 | Anlamsız relevance skoru | `distance/800` gibi keyfi bir sabite dayanıyordu | Koleksiyon cosine uzayında kuruluyor, `relevance = 1 - distance` |
| 5 | Streamlit: widget kendi tetiklediği koşuda `st.empty()` ile kaldırılamıyor | Platform davranışı (doğrulandı: hem `st.form_submit_button` hem düz `st.button` için) | Widget silinmek yerine aynı koşuda enjekte edilen CSS ile gizleniyor |
| 6 | Streamlit: tıklamadan hemen sonraki `st.rerun()` aktif sekmeyi sıfırlıyor | Platform davranışı (doğrulandı) | `st.rerun()` yalnızca uzun süren LLM çağrısı tamamlandıktan SONRA çağrılıyor; ara durumlar (typing göstergesi, yükleniyor kartı) aynı script koşusu içinde, `st.empty()`/`.container()` ile render ediliyor |
| 7 | `<div>`'i bir `st.markdown()` ile açıp başka bir çağrıda kapatmak DOM'da iç içe geçmiyor | Her `st.markdown()` kendi izole bloğunu oluşturuyor | Sarmalayıcı yerine hedef eleman doğrudan ya da `:has()` marker'ı üzerinden CSS'le stilleniyor |
| 8 | Demo videoların "transkripti" gerçek değil | Bu geliştirme ortamında YouTube bot koruması canlı erişimi engelliyor (hem caption API hem ASR fallback başarısız — doğrulandı) | Demo içerik, konuyu temsilen yazılmış metin olarak açıkça etiketlendi (kart rozeti + sekme içi uyarı); gerçek URL'ler bu sorundan etkilenmez |

**Bilinen sınırlamalar:**
- Tek kullanıcılı/yerel kullanım için tasarlandı — paylaşılan `st.session_state` ve tek ChromaDB dizini; çoklu eşzamanlı kullanıcı için ek izolasyon gerekir.
- Özet üretimi CPU-bound bir yerel modelle 1-4 dakika sürebilir.
- YouTube'un bot koruma sistemine karşı kırılgan (IP/istek sıklığına bağlı); `YT_PROXY_URL` kısmi bir çözümdür.

---

## Sorun Giderme

| Belirti | Kontrol | Çözüm |
|---|---|---|
| Model bulunamadı hatası | `ollama list` | `ollama pull qwen2.5:3b && ollama pull nomic-embed-text` |
| `Connection refused` | `curl http://localhost:11434/api/tags` | `ollama serve` |
| "Ollama ... saniye içinde yanıt vermedi" | `ps aux \| grep ollama`, sistem bellek kullanımı | Ollama'yı yeniden başlatın; RAM yetersizse `LLM_MODEL`'i küçültün |
| Modül bulunamadı | `which python`, `pip show streamlit` | Sanal ortamın aktif olduğundan emin olun, `pip install -r requirements.txt` |
| `ffmpeg` bulunamadı (ASR fallback) | — | macOS: `brew install ffmpeg` · Debian/Ubuntu: `sudo apt install ffmpeg` |
| Altyazı hiç bulunamıyor | Video kilitli/altyazısız mı, farklı ağ/VPN deneyin | `YT_PROXY_URL` tanımlayın |
| Eski/bozuk vektör verisi | — | `rm -rf chroma_db` (geri alınamaz, tüm önbellek silinir) |

---

## Katkıda Bulunma

```bash
git checkout -b feature/yeni-ozellik
# değişiklik + pytest -q
git commit -m "açık ve anlaşılır bir mesaj"
```
Pull request açmadan önce `pytest -q`'nun geçtiğinden emin olun. Hata raporu ve öneriler için GitHub Issues kullanılabilir.

---

## Lisans

Bu depoda bir `LICENSE` dosyası bulunmuyor — kullanım/dağıtım koşulları için proje sahibiyle iletişime geçin.

---

## Güvenlik ve Gizlilik

- Tüm çıkarım (LLM + embedding) yerelde, Ollama üzerinden çalışır; transkript/soru/yanıt verisi üçüncü bir sunucuya gönderilmez.
- LLM çıktısı render edilmeden önce `html.escape()`'den geçer (XSS'e karşı).
- ASR fallback'i tetiklenirse indirilen ses dosyası geçici bir dizinde (`tempfile.TemporaryDirectory`) tutulur ve işlem bitince otomatik silinir.
- `.env`, API anahtarı veya parola gibi gizli bilgileri depoya eklemeyin; `YT_PROXY_URL` gibi hassas değerleri ortam değişkeni olarak geçirin.
