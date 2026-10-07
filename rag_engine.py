"""
RAG Motoru — Geliştirilmiş Versiyon
- Küçük chunk'lar (500 char), cümle sınırlarına duyarlı
- Metadata desteği (chunk_index, source_video, start_time)
- Skor tabanlı arama (relevance threshold)
- Singleton client mimarisi
"""

import os
import time
import shutil
import hashlib
import sqlite3
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
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ─── YAPILANDIRMA ───
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RELEVANCE_THRESHOLD = 0.3  # Bu skorun altındaki sonuçlar filtrelenir
MAX_COLLECTIONS = 15       # chroma_db'nin sınırsız büyümesini önlemek için basit bir üst sınır

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


def _make_collection_name(source_key: str) -> str:
    """source_key'den (video URL'si veya başlığı) sabit/deterministik bir koleksiyon adı üretir."""
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]
    return f"yt_{digest}"


def _purge_orphaned_segment_dirs():
    """
    Bu chromadb sürümünde (0.5.5) delete_collection() segment klasörlerini
    diskten silmeyebiliyor (bilinen bir davranış) — sqlite'daki geçerli segment
    id'leriyle eşleşmeyen artık klasörleri fiziksel olarak temizler.
    """
    db_file = os.path.join(CHROMA_PATH, "chroma.sqlite3")
    if not os.path.exists(db_file):
        return
    try:
        con = sqlite3.connect(db_file)
        valid_ids = {row[0] for row in con.execute("SELECT id FROM segments")}
        con.close()
    except Exception:
        return
    try:
        entries = os.listdir(CHROMA_PATH)
    except Exception:
        return
    for name in entries:
        path = os.path.join(CHROMA_PATH, name)
        if os.path.isdir(path) and name not in valid_ids:
            shutil.rmtree(path, ignore_errors=True)


def _cleanup_old_collections(client, keep: int = MAX_COLLECTIONS):
    """MAX_COLLECTIONS'ı aşan en eski koleksiyonları siler (chroma_db'nin sınırsız birikmesini önler)."""
    try:
        collections = client.list_collections()
    except Exception:
        return
    if len(collections) > keep:
        collections.sort(key=lambda c: (c.metadata or {}).get("created_at", 0))
        for col in collections[: len(collections) - keep]:
            try:
                client.delete_collection(col.name)
            except Exception:
                pass
    _purge_orphaned_segment_dirs()


def create_vector_db(text: str, video_title: str = "Video",
                     timestamps: Optional[list] = None,
                     source_key: Optional[str] = None) -> str:
    """
    Metni parçalara ayırır, vektörleştirir ve ChromaDB'ye kaydeder.
    Her video kendi koleksiyonunda saklanır (diğer videoların verisi silinmez).
    Aynı source_key ile daha önce işlenmiş bir video varsa yeniden embed edilmez.

    Args:
        text: Video transkript metni
        video_title: Video başlığı (metadata için)
        timestamps: Opsiyonel zaman damgalı parçalar listesi
                    [{"start": 10.5, "text": "...", "duration": 3.2}, ...]
        source_key: Videoyu tekilleştiren anahtar (URL önerilir). Verilmezse
                    video_title kullanılır.

    Dönüş: collection_name — search_in_db()'ye geçirilmesi gereken koleksiyon adı.
    """
    client = _get_chroma_client()
    collection_name = _make_collection_name(source_key or video_title)

    # Video zaten indekslenmişse yeniden embed etme
    try:
        client.get_collection(collection_name)
        print(f"   ↺ '{video_title}' zaten indeksli, önbellekten kullanılıyor.")
        return collection_name
    except Exception:
        pass

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

    embeddings = _get_embeddings()

    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name,
        metadatas=metadatas,
        # Varsayılan (L2) yerine cosine mesafesi: skor hesaplamasını
        # keyfi bir sabite (distance/800) değil, gerçek bir ilgi ölçüsüne dayandırır.
        collection_metadata={"hnsw:space": "cosine", "created_at": time.time()},
    )

    _cleanup_old_collections(client)

    print(f"   ✓ Veritabanı başarıyla oluşturuldu ({len(chunks)} chunk).")
    return collection_name


def search_in_db(query: str, collection_name: str, k: int = 3) -> list:
    """
    Kullanıcının sorusuna en yakın k adet metin parçasını bulur.
    Alakasız (threshold altı) ve neredeyse birebir aynı (duplicate) sonuçlar elenir.

    Args:
        collection_name: create_vector_db()'nin döndürdüğü koleksiyon adı.

    Dönüş: [{"content": "...", "score": 0.85, "metadata": {...}}, ...]
    """
    client = _get_chroma_client()
    embeddings = _get_embeddings()

    vector_db = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # Dedup/threshold sonrası k'nın altına düşmemek için fazladan aday çek
    fetch_k = min(k * 4, 20)
    results_with_scores = vector_db.similarity_search_with_score(query, k=fetch_k)

    # Koleksiyon cosine mesafesiyle oluşturulduğu için (create_vector_db bkz.)
    # cosine_distance = 1 - cosine_similarity ⇒ relevance = 1 - distance
    formatted = []
    for doc, distance in results_with_scores:
        relevance = max(0.0, min(1.0, 1.0 - distance))
        formatted.append({
            "content": doc.page_content,
            "score": round(relevance, 3),
            "metadata": doc.metadata,
        })

    # Skora göre sırala (yüksek = daha ilgili)
    formatted.sort(key=lambda x: x["score"], reverse=True)

    # Alakasız sonuçları ele
    formatted = [r for r in formatted if r["score"] >= RELEVANCE_THRESHOLD]

    # Neredeyse aynı chunk'ları tekilleştir (ilk 80 karaktere göre)
    seen_prefixes = set()
    deduped = []
    for r in formatted:
        prefix = r["content"][:80].strip().lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        deduped.append(r)

    deduped = deduped[:k]

    return deduped if deduped else [{"content": "İlgili sonuç bulunamadı.", "score": 0, "metadata": {}}]


def search_in_db_simple(query: str, collection_name: str, k: int = 3) -> list[str]:
    """
    Geriye uyumlu basit arama — sadece metin listesi döner.
    (Eski app.py uyumluluğu için)
    """
    results = search_in_db(query, collection_name, k=k)
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

    col_name = create_vector_db(ornek_metin, video_title="Test Videosu", source_key="test-videosu")

    print("\n3. Arama testi...")
    results = search_in_db("RAG mimarisi ne işe yarar?", col_name, k=3)
    for i, r in enumerate(results, 1):
        print(f"\n--- Sonuç {i} (skor: {r['score']}) ---")
        print(f"Metadata: {r['metadata']}")
        print(f"İçerik: {r['content'][:150]}...")

    print("\n4. Basit arama testi (geriye uyumlu)...")
    simple = search_in_db_simple("Büyük dil modeli nedir?", col_name, k=2)
    for i, text in enumerate(simple, 1):
        print(f"\nSonuç {i}: {text[:100]}...")

    print("\n✓ Tüm testler başarılı.")
