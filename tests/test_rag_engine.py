import rag_engine as rag


class TestMakeCollectionName:
    def test_deterministic_for_same_input(self):
        a = rag._make_collection_name("https://www.youtube.com/watch?v=aircAruvnKk")
        b = rag._make_collection_name("https://www.youtube.com/watch?v=aircAruvnKk")
        assert a == b

    def test_different_inputs_produce_different_names(self):
        a = rag._make_collection_name("video-a")
        b = rag._make_collection_name("video-b")
        assert a != b

    def test_name_is_chroma_safe(self):
        name = rag._make_collection_name("https://www.youtube.com/watch?v=aircAruvnKk")
        assert name.startswith("yt_")
        # Chroma koleksiyon adları alfanümerik + alt çizgi/tire ile sınırlıdır.
        assert all(c.isalnum() or c in "_-" for c in name)


class TestFindChunkTimestamp:
    def _timestamps(self):
        return [
            {"start": 0.0, "text": "Merhaba millet bugün yapay zeka konuşacağız"},
            {"start": 5.0, "text": "büyük dil modelleri hakkında konuşacağız"},
            {"start": 12.0, "text": "ilk olarak RAG mimarisini ele alalım"},
        ]

    def test_matches_chunk_at_start(self):
        result = rag._find_chunk_timestamp("Merhaba millet bugün yapay zeka konuşacağız", self._timestamps())
        assert result == 0.0

    def test_empty_chunk_text_returns_none(self):
        assert rag._find_chunk_timestamp("   ", self._timestamps()) is None

    def test_empty_timestamps_returns_none(self):
        assert rag._find_chunk_timestamp("herhangi bir metin", []) is None

    def test_no_match_falls_back_to_first_timestamp(self):
        result = rag._find_chunk_timestamp("alakasız tamamen farklı bir cümle burada", self._timestamps())
        assert result == self._timestamps()[0]["start"]


class TestCleanupOldCollectionsNoOp:
    def test_handles_client_errors_gracefully(self, tmp_path, monkeypatch):
        # Gerçek proje chroma_db/ klasörünü etkilememesi için izole bir yol kullan.
        monkeypatch.setattr(rag, "CHROMA_PATH", str(tmp_path))

        class BrokenClient:
            def list_collections(self):
                raise RuntimeError("boom")

        # Hata fırlatmamalı, sessizce dönmeli.
        rag._cleanup_old_collections(BrokenClient(), keep=5)
