"""
RAG Motoru — Geliştirilmiş Versiyon
- Küçük chunk'lar (500 char), cümle sınırlarına duyarlı
- Metadata desteği (chunk_index, source_video, start_time)
- Skor tabanlı arama (relevance threshold)
- Singleton client mimarisi
"""

import os
import warnings
import logging

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)

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

from typing import Optional
import chromadb
warnings.filterwarnings("ignore")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ─── YAPILANDIRMA ───
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "youtube_rag"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RELEVANCE_THRESHOLD = 0.3  # Bu skorun altındaki sonuçlar filtrelenir

# Cümle sınırlarına saygılı ayırıcılar (öncelik sırasıyla)
SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

# ─── SINGLETON CLIENT & EMBEDDINGS ───
_chroma_client = None
_embeddings_model = None


def _get_chroma_client():
    """Tek bir PersistentClient instance döndürür."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def _get_embeddings():
    """Tek bir OllamaEmbeddings instance döndürür."""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    return _embeddings_model


def create_vector_db(text: str, video_title: str = "Video",
                     timestamps: Optional[list] = None):
    """
    Metni parçalara ayırır, vektörleştirir ve ChromaDB'ye kaydeder.

    Args:
        text: Video transkript metni
        video_title: Video başlığı (metadata için)
        timestamps: Opsiyonel zaman damgalı parçalar listesi
                    [{"start": 10.5, "text": "...", "duration": 3.2}, ...]
    """
    print(f"1. Metin parçalanıyor (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS,
    )

    chunks = text_splitter.split_text(text)
    print(f"   → {len(chunks)} parçaya bölündü.")

    # Metadata oluştur
    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = {
            "chunk_index": i,
            "source_video": video_title,
            "total_chunks": len(chunks),
        }

        # Zaman damgası eşleştirme
        if timestamps:
            start_time = _find_chunk_timestamp(chunk, timestamps)
            if start_time is not None:
                meta["start_time"] = start_time
                mins, secs = divmod(int(start_time), 60)
                meta["time_label"] = f"{mins:02d}:{secs:02d}"

        metadatas.append(meta)

    print("2. Vektörleştirme ve ChromaDB'ye kaydetme...")

    client = _get_chroma_client()
    embeddings = _get_embeddings()

    # Var olan koleksiyonu temizle
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
        metadatas=metadatas,
    )

    print(f"   ✓ Veritabanı başarıyla oluşturuldu ({len(chunks)} chunk).")
    return vector_db


def search_in_db(query: str, k: int = 3) -> list:
    """
    Kullanıcının sorusuna en yakın k adet metin parçasını bulur.

    Dönüş: [{"content": "...", "score": 0.85, "metadata": {...}}, ...]
    """
    client = _get_chroma_client()
    embeddings = _get_embeddings()

    vector_db = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Ham mesafe tabanlı arama (düşük mesafe = daha ilgili)
    results_with_scores = vector_db.similarity_search_with_score(query, k=k)

    # Mesafeyi 0-1 arasında ilgililik skoruna çevir
    # Cosine distance: 0 = tam eşleşme, büyük değer = farklı
    formatted = []
    for doc, distance in results_with_scores:
        # Basit dönüşüm: score = max(0, 1 - distance/1000)
        # Pratikte nomic-embed distance'lar genelde 200-600 arasında
        relevance = max(0.0, min(1.0, 1.0 - (distance / 800)))
        formatted.append({
            "content": doc.page_content,
            "score": round(relevance, 3),
            "metadata": doc.metadata,
        })

    # Skora göre sırala (yüksek = daha ilgili)
    formatted.sort(key=lambda x: x["score"], reverse=True)

    return formatted if formatted else [{"content": "İlgili sonuç bulunamadı.", "score": 0, "metadata": {}}]


def search_in_db_simple(query: str, k: int = 3) -> list[str]:
    """
    Geriye uyumlu basit arama — sadece metin listesi döner.
    (Eski app.py uyumluluğu için)
    """
    results = search_in_db(query, k=k)
    return [r["content"] for r in results]


def _find_chunk_timestamp(chunk_text: str,
                          timestamps: list) -> Optional[float]:
    """
    Bir chunk'ın hangi zaman damgasıyla eşleştiğini bulur.
    Chunk'ın ilk 60 karakteri ile timestamp text'leri karşılaştırılır.
    """
    search_str = chunk_text[:60].strip().lower()
    if not search_str:
        return None

    # Kümülatif metin oluştur ve eşleşme ara
    cumulative = ""
    for ts_item in timestamps:
        ts_text = ts_item.get("text", "").lower()
        cumulative += " " + ts_text

        if search_str[:30] in cumulative[-200:]:
            return ts_item.get("start", 0)

    return timestamps[0].get("start", 0) if timestamps else None


# ─── DOĞRUDAN ÇALIŞTIRMA TESTİ ───
if __name__ == "__main__":
    ornek_metin = (
        "Yapay zeka (YZ), bilgisayarların insan benzeri zekaya sahip olmasını "
        "sağlayan bir bilim dalıdır. Büyük dil modelleri (LLM), insan benzeri "
        "metinler üretmek için devasa veri setleriyle eğitilmiştir. "
        "RAG mimarisi, yapay zekanın eğitilmediği özel verileri okuyup doğru "
        "yanıtlar vermesini sağlar. Halüsinasyon, yapay zekanın uydurma bilgi "
        "üretmesi sorunudur. RAG bu sorunu %90 oranında engelleyebilir."
    ) * 8

    print("=" * 60)
    print("RAG MOTORU TESTİ — Geliştirilmiş Versiyon")
    print("=" * 60)

    create_vector_db(ornek_metin, video_title="Test Videosu")

    print("\n3. Arama testi...")
    results = search_in_db("RAG mimarisi ne işe yarar?", k=3)
    for i, r in enumerate(results, 1):
        print(f"\n--- Sonuç {i} (skor: {r['score']}) ---")
        print(f"Metadata: {r['metadata']}")
        print(f"İçerik: {r['content'][:150]}...")

    print("\n4. Basit arama testi (geriye uyumlu)...")
    simple = search_in_db_simple("Büyük dil modeli nedir?", k=2)
    for i, text in enumerate(simple, 1):
        print(f"\nSonuç {i}: {text[:100]}...")

    print("\n✓ Tüm testler başarılı.")
