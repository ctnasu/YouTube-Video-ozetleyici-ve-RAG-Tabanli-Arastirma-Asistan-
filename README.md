# 🎥 YouTube Video Özetleyici ve RAG Tabanlı Araştırma Asistanı

Bu proje, YouTube videolarını yapay zeka kullanarak analiz eden, derinlemesine akademik özetler çıkaran ve video içeriğine dair sorularınızı yanıtlayan akıllı bir **RAG (Retrieval-Augmented Generation)** uygulamasıdır.

Tüm dil modeli işlemleri **tamamen yerel (local)** olarak [Ollama](https://ollama.com/) üzerinden çalıştırılır. Gizlilik odaklıdır ve bulut API maliyeti gerektirmez.

---

## ✨ Temel Özellikler

- **📝 Akıllı Transkript Çıkarımı:** YouTube videolarının altyazılarını (zaman damgaları ile birlikte) otomatik olarak çeker.
- **🧠 Kapsamlı AI Özeti (Few-Shot Prompting):** Videoları sadece yüzeysel olarak özetlemez; özel olarak tasarlanmış katı "Few-Shot" istem şablonları sayesinde, teknik detayları ve örnekleri atlamadan akademik derinlikte (Yoğunluklu Tematik Özet) analiz raporları üretir.
- **💬 RAG Tabanlı Video Chatbot:** Videonun transkriptini kelime parçalarına (chunk) böler ve vektörleştirir. Siz videoyla ilgili bir soru sorduğunuzda, sadece videodaki ilgili kısımları bularak size referanslı (zaman damgalı) cevaplar verir.
- **🛡️ Tekrarsız Kaynak Filtreleme:** Chatbot'un aynı veya çok benzer kaynakları (duplicate chunks) tekrar etmesini önleyen özel bir deduplication (tekilleştirme) algoritmasına sahiptir.
- **🎨 Modern Arayüz:** Streamlit kullanılarak geliştirilmiş, akıcı ve şık (Glassmorphism esintili) bir kullanıcı deneyimi sunar.

---

## 🛠️ Kullanılan Teknolojiler ve Mimari

### 1. Arayüz ve Uygulama Çatısı
- **[Streamlit](https://streamlit.io/):** Projenin hızlı ve etkileşimli web arayüzünü oluşturmak için kullanıldı.

### 2. Dil Modelleri ve Vektörleştirme (Local AI)
- **[Ollama](https://ollama.com/):** Yerel makinede açık kaynaklı modelleri çalıştırmak için.
- **LLM Modeli:** `qwen2.5:3b` (Özetleme ve Chatbot yanıtları için). 
- **Gömme (Embedding) Modeli:** `nomic-embed-text` (Metinleri vektörlere dönüştürmek için).

### 3. RAG (Retrieval-Augmented Generation) Motoru
- **[ChromaDB](https://www.trychroma.com/):** Vektör veritabanı olarak kullanıldı. Transkript parçaları burada saklanır ve semantik (anlamsal) arama ile sorgulanır.
- **[LangChain](https://www.langchain.com/):** Metinleri anlamlı parçalara (chunk) bölmek için `RecursiveCharacterTextSplitter` kullanıldı (Cümle sınırlarına saygılı ve örtüşmeli - overlap - bölme).

### 4. Veri Çekme
- **[youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/):** YouTube videolarından zaman damgalı transkriptleri çekmek için entegre edildi.

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.9+
- Ollama (Bilgisayarınızda yüklü ve çalışıyor olmalıdır)

### 1. Ollama Modellerini İndirin
Terminali açın ve projenin ihtiyaç duyduğu dil modellerini yerel makinenize indirin:
```bash
ollama run qwen2.5:3b
ollama pull nomic-embed-text
```

### 2. Python Kütüphanelerini Kurun
Proje dizininde sanal bir ortam (venv) oluşturun ve gerekli kütüphaneleri yükleyin:
```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
pip install streamlit chromadb langchain-text-splitters langchain-community youtube-transcript-api
```

### 3. Uygulamayı Başlatın
Aşağıdaki komutla Streamlit sunucusunu başlatın:
```bash
streamlit run app.py
```
Uygulama tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

---

## 🏗️ Projenin Gelişim Sürecinde Yapılan Önemli Geliştirmeler

Proje geliştirilirken karşılaşılan zorluklar ve uygulanan çözümler:

1. **Yüzeysel Özet Sorunu (Çözüldü):** Küçük dil modelleri (3B vb.) metinleri çok kısa ve yetersiz özetliyordu. Çözüm olarak **"Few-Shot Prompting"** uygulandı. Sistemin içine Andrej Karpathy'nin devasa uzunluktaki mükemmel bir AI özeti "örnek format" olarak gömüldü. Model artık bu örneği taklit ederek son derece yoğun ve akademik özetler çıkarıyor.
2. **"Alt Başlık/Kavram" Meta Etiketlerinin Çıktıya Sızması (Çözüldü):** İlk başlarda AI, şablondaki köşeli parantezli açıklama kelimelerini birebir metne basıyordu. Prompt güncellenerek görsel bir Markdown iskeleti çizildi ve kalıp sözcüklerin yazılması yasaklandı.
3. **Chatbot'un Aynı Kaynakları Tekrar Etmesi (Çözüldü):** RAG motoru bazen aynı cümlenin farklı chunk'lardaki versiyonlarını buluyordu. `rag_engine.py` içerisindeki `search_in_db` fonksiyonuna özel bir filtreleme mantığı (ilk 80 karaktere göre deduplication) eklenerek kaynakların her zaman %100 eşsiz (unique) olması sağlandı.
4. **Chunk (Parçalama) Optimizasyonu:** Vektör aramalarının daha isabetli olabilmesi için metinler 500 karakterlik küçük bloklara (100 karakter örtüşme ile) bölündü. Bu sayede AI, kullanıcının sorusuna doğrudan ilgili 2-3 cümleyi bularak nokta atışı cevaplar verebilir hale geldi.
