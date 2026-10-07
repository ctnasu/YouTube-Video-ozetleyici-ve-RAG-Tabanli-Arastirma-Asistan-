# 🎥 YouTube Video Özetleyici ve RAG Tabanlı Araştırma Asistanı

YouTube videolarının transkriptini çıkaran, derinlemesine akademik özetler üreten ve video içeriği hakkındaki sorularınızı kaynak alıntılarıyla yanıtlayan, **tamamen yerel (local-first)** bir **RAG (Retrieval-Augmented Generation)** uygulaması.

Tüm dil modeli işlemleri [Ollama](https://ollama.com/) üzerinden, kendi makinenizde çalışır. Hiçbir veri dışarı çıkmaz, bulut API maliyeti gerektirmez.

> Bu README, projenin ilk halinden bugüne geçirdiği süreci, karşılaşılan gerçek sorunları ve bunların nasıl çözüldüğünü de belgeliyor — "her şey ilk seferde mükemmel çalıştı" demiyoruz; aşağıda **dürüst bir değerlendirme** de var.

---

## ✨ Özellikler

- **📝 Akıllı Transkript Çıkarımı** — YouTube altyazılarını zaman damgalarıyla çeker; desteklenen dillerde altyazı yoksa mevcut altyazıyı Türkçe'ye çevirir, hiç altyazı yoksa yt-dlp + yerel Whisper ile sesten transkript çıkarır (ASR fallback). Transkript sekmesinde içerik, sabit sayıda değil ~60 saniyelik zaman pencerelerine göre gruplanır.
- **🧠 Kapsamlı AI Özeti** — Videoyu 4-5 ana temaya ayıran, zaman damgalı, akademik derinlikte bir analiz raporu üretir (few-shot prompting).
- **💬 RAG Tabanlı Video Chatbot** — Transkripti vektörleştirip ChromaDB'de saklar; sorularınızı sadece videonun ilgili kısımlarına dayanarak, tıklanınca açılan kaynak alıntılarıyla (relevance % dahil) yanıtlar.
- **🗂️ Çoklu Video Desteği** — Her video kendi koleksiyonunda saklanır; sohbet geçmişi ve özet video bazında kalıcıdır (aynı videoyu tekrar açtığınızda kaldığınız yerden devam eder).
- **🎨 Modern, koyu temalı arayüz** — Geist ve Instrument Serif tipografisi, segmented-control sekmeler, göz yormayan hareketli arka plan, tıklanabilir kaynak çipleri, adım adım ilerleme göstergeleri.
- **🛡️ Güvenli render** — LLM/kullanıcı çıktısı gerçek Markdown'a (kalın, liste, başlık) çevrilir; ham HTML önce kaçışlanır (XSS'e karşı güvenli).
- **🧪 Test edilmiş** — `tests/` altında pytest ile birim ve entegrasyon testleri.

---

## 🛠️ Mimari ve Teknoloji Yığını

### Arayüz
- **[Streamlit](https://streamlit.io/)** — `app.py` tek dosyalık UI; tasarım dili büyük ölçüde özel CSS ile (Streamlit'in varsayılan görünümü neredeyse tamamen değiştirilmiş durumda).

### Dil Modelleri (Ollama, tamamen yerel)
- **LLM:** `qwen2.5:3b` — `num_ctx=8192`, özetleme ve chatbot için ayrı `num_predict` değerleriyle. *(Neden 7B değil 3B: aşağıdaki "Geliştirme Süreci" bölümündeki #13'e bakın — bu, projenin en önemli, hâlâ tam çözülmemiş dengesi.)*
- **Embedding:** `nomic-embed-text`
- **[langchain-ollama](https://pypi.org/project/langchain-ollama/)** — resmi/güncel LangChain-Ollama entegrasyonu.

### RAG Motoru
- **[ChromaDB](https://www.trychroma.com/)** — cosine mesafeli vektör veritabanı; her video kendi koleksiyonunda.
- **[LangChain](https://www.langchain.com/)** `RecursiveCharacterTextSplitter` — cümle sınırına duyarlı, örtüşmeli chunking.

### Veri Çekme
- **[youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/)** — zaman damgalı altyazı çekimi (`YT_PROXY_URL` ile proxy desteği).
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp) + [faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — altyazısız videolarda son çare: ses indirip yerel Whisper (`small`) ile transkript çıkarır. `ffmpeg` gerektirir.

### Render ve Test
- **[markdown](https://python-markdown.github.io/)** — güvenli HTML render (`render_markdown_html`).
- **[pytest](https://pytest.org/)** — birim + entegrasyon testleri.

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.9+
- Ollama (yüklü ve çalışır durumda)
- ffmpeg (Whisper ASR fallback'i için: `brew install ffmpeg` / `apt install ffmpeg`)

```bash
# 1. Ollama modellerini indirin
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 2. Python ortamını kurun
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Çalıştırın
streamlit run app.py        # http://localhost:8501
```

**Daha güçlü bir makineniz varsa** (≥16GB RAM önerilir): `app.py` içindeki `LLM_MODEL` sabitini `"qwen2.5:7b"` yapın — özet kalitesi belirgin şekilde artar (bkz. aşağıdaki #8 ve #13).

### Testler
```bash
pip install -r requirements-dev.txt
pytest
```
`tests/` iki tür test içerir: ağ/Ollama gerektirmeyen **birim testleri** (her zaman çalışır) ve gerçek Ollama embedding çağrısı yapan, Ollama kapalıyken otomatik atlanan **entegrasyon testleri**.

---

## 🏗️ Geliştirme Süreci: Karşılaşılan Sorunlar ve Çözümler

Proje tek seferde bu hâline gelmedi. Aşağıda, gerçekten karşılaşılan (çoğu tarayıcıda canlı test edilerek doğrulanmış) sorunlar ve çözümleri kronolojik olarak listeleniyor.

### Fonksiyonel ve Performans Sorunları

1. **Yüzeysel özet:** Küçük modeller metni çok kısa özetliyordu → **Few-shot prompting** (örnek format gömme) ile çözüldü.
2. **Chatbot aynı kaynağı tekrar ediyordu** → `search_in_db`'ye ilk 80 karaktere göre deduplication eklendi.
3. **Chunk optimizasyonu** → 500 karakterlik, 100 karakter örtüşmeli, cümle sınırına duyarlı parçalama.
4. **Sessizce kırpılan context (kök neden):** Ollama `num_ctx` verilmezse varsayılan olarak 2048 token'da kalıyor — uzun transkript + few-shot prompt kombinasyonunda transkriptin büyük kısmı modele hiç ulaşmıyordu. `num_ctx`/`num_predict` artık açıkça ayarlanıyor.
5. **Anlamsız skor hesabı:** İlgililik skoru keyfi bir sabite (`distance/800`) dayanıyordu → koleksiyon cosine mesafesiyle kuruldu, `relevance = 1 - distance`, gerçek bir `RELEVANCE_THRESHOLD` uygulanıyor.
6. **Tek video sınırlaması:** Yeni video, eskisinin ChromaDB koleksiyonunu siliyordu → her video kendi koleksiyonunda, tekrar açılan video yeniden embed edilmiyor.
7. **Altyazısız videolar:** `youtube-transcript-api` başarısız olursa yt-dlp + yerel `faster-whisper` ile sesten transkript (ASR fallback) devreye giriyor.
8. **`num_ctx` ile hız/kalite dengesi:** Kaliteyi artırmak için `num_ctx=16384`'e çıkınca, bu geliştirme makinesinde (Apple Silicon) üretim hızının **5 kat düştüğü** ölçüldü (~16 tok/s → ~3 tok/s). `num_ctx=8192`'de bu düşüş yok; hem orijinal 2048 sınırından çok daha geniş hem pratik hızda.
9. **Deprecated `langchain_community.llms.Ollama`** → `langchain-ollama`'ya geçildi (ve uyumluluk için tüm LangChain paketleri 0.3.x'e yükseltildi).
10. **Bozuk Markdown render:** AI çıktısı ekranda ham `**`/`*` karakterleri olarak görünüyordu (metin doğrudan HTML `<div>`'e basılıyordu) → `render_markdown_html()` ile gerçek, güvenli (kaçışlanmış) HTML'e çevrildi. Ayrıca LLM'in madde imi öncesi boş satır bırakmaması nedeniyle listelerin düz metin kalması ayrı bir hataydı (`_ensure_blank_line_before_lists` ile çözüldü).
11. **Hayali URL üretimi:** Özet prompt'undaki few-shot örnekte sabit bir `(URL: ...)` referansı vardı; model her özette var olmayan bir video linki uyduruyordu (hallucination) → örnekten kaldırıldı.
12. **Test suite eksikliği** → pytest ile birim + entegrasyon testleri eklendi.
13. **8GB RAM'de sessizce takılan Ollama istekleri (önemli, kalıcı bir ödünleşim):** `qwen2.5:7b` (~5GB), bu geliştirme makinesinde (toplam 8GB RAM) uzun kullanımda Ollama'nın **hiç hata vermeden süresiz olarak** yanıt vermeyi bırakmasına yol açıyordu. Doğrudan `curl` ile ve izole bir Python betiğiyle doğrulandı. İki parçalı çözüm: **(a)** varsayılan model `qwen2.5:3b`'ye (~2GB) düşürüldü — çok daha güvenilir ama gözle görülür biçimde daha sade/tutarsız kalite; **(b)** tüm LLM çağrılarına 300 saniyelik bir zaman aşımı eklendi, böylece gerçekten takılırsa kullanıcı sonsuza kadar değil, anlaşılır bir hata mesajıyla karşılaşıyor. **Bu, projenin en büyük açık ödünleşimi** — detay için aşağıdaki "Mevcut Durum" bölümüne bakın.
14. **"Temizle" butonu aktif sekmeyi sıfırlıyordu:** Butona dinamik bir `disabled=...` parametresi verilmişti; tıklanıp `st.rerun()` çağrıldığında Streamlit aktif sekmeyi ilk sekmeye (Transkript) sıfırlıyordu. Kök neden `disabled` parametresiydi (aynı deseni kullanan `disabled` içermeyen diğer butonlarda sorun yoktu); kaldırılınca düzeldi. Bu, projede birkaç kez daha karşılaşılan bir **Streamlit platform kısıtının** ilk örneğiydi (bkz. #24).

### Arayüz (UI/UX) Yeniden Tasarımı ve Bulunan Hatalar

Proje başlangıçta açık renkli, jenerik bir "SaaS şablonu" görünümündeydi. Kullanıcı isteği üzerine koyu temalı, Linear/Vercel tarzı bir tasarıma geçildi; bu süreçte **gerçek tarayıcı testleriyle (Playwright)** bulunan somut hatalar:

15. **Font gerçekten uygulanmıyordu:** Başlıklar yeni fontu alıyordu ama gövde metninin çoğu hâlâ Streamlit'in varsayılanı "Source Sans Pro" ile render oluyordu — global seçicinin kapsamı yetersizdi. Renk için zaten kullanılan geniş seçiciye `font-family` de eklenince düzeldi. (Font birkaç kez değişti: Plus Jakarta Sans → Space Grotesk → son olarak **Geist** + vurgu kelimesi için **Instrument Serif** italik.)
16. **"Hayalet widget" DOM hatası (tekrarlayan bir desen):** Bir `<div>`'i bir `st.markdown()` çağrısıyla açıp aradan `st.columns`/`st.form` gibi native bileşenler geçtikten SONRA başka bir çağrıyla kapatmak, DOM'da **gerçekten iç içe geçmiyor** — her `st.markdown` çağrısı kendi izole bloğunu oluşturuyor. Bu, arama çubuğunda ve chat input'ta aynı hatayı birkaç kez yarattı (boş, stilsiz bir "çubuk" asıl bileşenin üstünde ayrı duruyordu). Çözüm: sarmalayıcı div yerine hedef eleman doğrudan, ya da `:has()` ile referans alınan bir marker üzerinden CSS'le stilleniyor.
17. **Aşırı geniş kapsamlı CSS seçici:** `stVerticalBlockBorderWrapper` seçicisi sadece transkript kutusunu değil, Streamlit'in oluşturduğu HER sütun/panel sarmalayıcısını hedefliyordu — sayfanın tamamı "iç içe kart" gibi görünüyordu. `[height]` özniteliğiyle (sadece o kutuda kullanılan) kesin olarak hedeflenerek çözüldü.
18. **Dar içerik alanı + hizasız arama kutusu:** `max-width` çok dar olduğu için geniş ekranlarda (büyük monitör/Safari) içerik ortada sıkışıp kenarlar boş kalıyordu; arama inputu ile buton da dikey hizasızdı. `max-width` artırıldı, `st.columns(..., vertical_alignment="bottom")` ile hizalama düzeltildi.
19. **Sabit yükseklikli transkript kutusu:** 520px sabitti, uzun ekranlarda gereksiz küçük kalıyordu → `min(72vh, 900px)` ile viewport'a duyarlı hâle getirildi.
20. **Çoklu video desteğiyle gelen yeni bir regresyon:** "Geri Dön" ile çıkılıp aynı video tekrar açıldığında sohbet geçmişi ve özet siliniyordu → `video_sessions` sözlüğüyle koleksiyon bazında kalıcı hâle getirildi.
21. **Demo videoların kaynak alıntılarında zaman damgası hiç çıkmıyordu** (gerçek videoların aksine) → `generate_synthetic_timestamps()` ile demo videolar da aynı koddan geçirilip tutarlı zaman etiketleri kazandı.
22. **Chatbot'un "yanıt oluşturuyor" göstergesi** sohbet akışının dışında, input'un altında kopuk bir Streamlit spinner'ıydı → sohbet balonu görünümünde, zıplayan noktalı bir "typing" göstergesine çevrildi.
23. **"Kaynaklar" üç kez revize edildi:** Önce kalabalık bir dikey liste → yatay çipler (ama sadece hover'da ipucu gösteriyordu, tıklamaya tepkisiz — kullanıcı "tıklayınca bir şey çıkmıyor" diye bildirdi) → `<details>/<summary>` ile gerçekten tıklanınca açılan, JS'siz çipler. Ardından açılan çip komşularının üstünde boşluk bırakıyordu (flex-wrap'in doğası) — `[open] { flex-basis:100% }` ile düzeltildi.
24. **Özet oluştururken buton, yükleniyor kartının altında asılı kalıyordu.** Bunun düzeltilmesi, gerçek bir **Streamlit platform kısıtıyla** uğraşmayı gerektirdi: `st.empty()`/`.container()` ile bir widget'ı KENDİSİNİ tetikleyen script koşusunda silmek güvenilir çalışmıyor; tıklamadan hemen sonra ek bir `st.rerun()` çağırmak ise aktif sekmeyi ilk sekmeye sıfırlıyor (hem form hem düz buton için ayrı ayrı doğrulandı — #14'teki sorunla aynı kök neden). Üç farklı yaklaşım denendi; nihai çözüm: widget'ı SİLMEK yerine, aynı koşuda enjekte edilen kapsamlı bir CSS kuralıyla **görsel olarak gizlemek**.
25. **Transkript blokları çok kabaydı:** Sabit sayıda (5) öğe gruplanıyordu — gerçek altyazılarda makuldü ama sentetik zaman damgalarında (öğe başına ~30-45sn) 3-4 dakikalık, kalabalık bloklar oluşturuyordu. Sabit bir **zaman penceresine** (~60sn, "look-ahead" mantığıyla) göre gruplamaya çevrildi; artık her blok tutarlı biçimde ~40-60 saniye.
26. **Demo videoların "transkripti" gerçek değildi:** Bu geliştirme ortamında YouTube'un bot koruma sistemi canlı altyazı/ASR erişimini engellediği doğrulandı (hem caption API hem Whisper fallback başarısız). Bu yüzden 3 örnek video için gösterilen metin, konuyu temsilen **yazılmış bir özetti** — ama "Transkript" etiketi altında sunulduğu için yanıltıcıydı. Kullanıcı bunu doğru şekilde fark edip sorguladı. Çözüm: demo kartlarına "Örnek metin" rozeti ve Transkript sekmesine açık bir uyarı eklendi; gerçek URL ile eklenen videolar bu sorundan hiç etkilenmiyor (o yol LLM'den hiç geçmiyor, doğrudan gerçek YouTube altyazısını çekiyor).

---

## 📊 Mevcut Durum: Dürüst Bir Değerlendirme

**İyi çalışan, sağlam yönler:**
- Mimari gerçek bir RAG sistemi — anlamlı skorlama, deduplication, kaynak alıntılama, çoklu video desteği, kalıcı oturumlar.
- Hata yönetimi düşünülmüş: bot engeli, ASR fallback, zaman aşımı, anlaşılır hata mesajları.
- Arayüz onlarca kez gerçek tarayıcı testiyle (Playwright) doğrulandı; bulunan her hata kök nedenine kadar izlenip düzeltildi — "görünüşte çalışıyor" değil, gerçekten test edilmiş.
- Otomatik test suite (pytest) mevcut.
- Tamamen yerel/gizlilik odaklı, bulut maliyeti yok.

**Bilinen sınırlamalar (gizlenmiyor):**
- **Özet kalitesi donanıma bağlı ve şu anki varsayılan ayarda sınırlı:** 8GB RAM'lik bu geliştirme makinesinde `qwen2.5:3b` kullanmak zorunda kalındı (bkz. #13); bu küçük model bazen few-shot örnekteki **içeriği** (konuyla alakasız ifadeleri) taklit edebiliyor — bu **aktif, tam çözülmemiş bir kalite sorunudur**. ≥16GB RAM'li bir makinede `LLM_MODEL = "qwen2.5:7b"` yapmak bunu büyük ölçüde iyileştirir.
- **Demo videoların transkripti gerçek değil** — sadece bu ortamda YouTube erişimi engellendiği için geçici bir temsili içerik (açıkça etiketlenmiş durumda). Gerçek bir URL girildiğinde bu sorun yok.
- **Özet üretimi yavaş** (video uzunluğuna göre 1-4 dakika) — tamamen yerel/CPU-bound bir model olduğu için; bulut API'lerine göre daha yavaş ama ücretsiz ve gizli.
- **Tek kullanıcılı/yerel kullanım için tasarlandı** — Streamlit `session_state` ve paylaşılan bir ChromaDB kullanıyor; çoklu eşzamanlı kullanıcı için (üretim/sunucu ortamı) ek çalışma gerekir.
- YouTube'un bot koruma sistemine karşı kırılgan (IP/istek sıklığına bağlı); `YT_PROXY_URL` ile kısmen aşılabilir.

**Özetle:** Kişisel/yerel kullanım için mimarisi sağlam, arayüzü modern ve gerçekten test edilmiş bir araç. En büyük açık nokta, donanım kısıtı yüzünden küçük bir modele düşülmesinin özet kalitesine etkisi — bu, kod kalitesinden değil, kullanılabilir RAM'den kaynaklanan, bilinçli ve belgelenmiş bir ödünleşim.
