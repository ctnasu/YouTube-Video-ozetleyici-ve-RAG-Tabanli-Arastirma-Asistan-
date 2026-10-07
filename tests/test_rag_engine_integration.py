"""
Bu dosyadaki testler gerçek Ollama embedding çağrıları yapar (nomic-embed-text).
Yerel Ollama sunucusu çalışmıyorsa otomatik olarak atlanır (bkz. conftest.requires_ollama).
"""
import rag_engine as rag
from conftest import requires_ollama


SAMPLE_TEXT = (
    "Yapay zeka, bilgisayarların insan benzeri zekaya sahip olmasını sağlayan bir bilim dalıdır. "
    "Büyük dil modelleri, insan benzeri metinler üretmek için devasa veri setleriyle eğitilir. "
    "RAG mimarisi, yapay zekanın eğitilmediği özel verileri okuyup doğru yanıtlar vermesini sağlar. "
    "Halüsinasyon, yapay zekanın uydurma bilgi üretmesi sorunudur. "
) * 6


@requires_ollama
class TestCreateAndSearch:
    def test_create_vector_db_returns_collection_name(self, isolated_chroma):
        col_name = rag.create_vector_db(SAMPLE_TEXT, video_title="Test", source_key="integration-test-1")
        assert col_name.startswith("yt_")

    def test_search_returns_relevant_results_with_valid_scores(self, isolated_chroma):
        col_name = rag.create_vector_db(SAMPLE_TEXT, video_title="Test", source_key="integration-test-2")
        results = rag.search_in_db("RAG mimarisi ne işe yarar?", col_name, k=3)

        assert len(results) > 0
        for r in results:
            assert 0.0 <= r["score"] <= 1.0
            assert "content" in r and r["content"]

    def test_search_results_are_deduplicated(self, isolated_chroma):
        # SAMPLE_TEXT tekrarlı olduğu için chunk'lar arasında yüksek benzerlik var;
        # dedup mantığı aynı ilk-80-karakter önekine sahip sonuçları elemeli.
        col_name = rag.create_vector_db(SAMPLE_TEXT, video_title="Test", source_key="integration-test-3")
        results = rag.search_in_db("Halüsinasyon nedir?", col_name, k=10)

        prefixes = [r["content"][:80].strip().lower() for r in results]
        assert len(prefixes) == len(set(prefixes))

    def test_reprocessing_same_source_key_uses_cache(self, isolated_chroma, capsys):
        source_key = "integration-test-cache"
        rag.create_vector_db(SAMPLE_TEXT, video_title="Test", source_key=source_key)
        capsys.readouterr()  # ilk çağrının çıktısını temizle

        rag.create_vector_db(SAMPLE_TEXT, video_title="Test", source_key=source_key)
        captured = capsys.readouterr()
        assert "önbellekten" in captured.out
