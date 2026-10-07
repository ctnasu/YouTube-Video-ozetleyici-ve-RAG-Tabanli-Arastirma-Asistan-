import urllib.request

import pytest


def _ollama_available() -> bool:
    """Yerel Ollama sunucusu ayaktaysa True döner (embedding gerektiren testler için)."""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(), reason="Bu test için yerel Ollama sunucusu çalışıyor olmalı"
)


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """Testlerin gerçek proje chroma_db/ klasörünü kirletmemesi için izole, geçici bir yol kullanır."""
    import rag_engine

    monkeypatch.setattr(rag_engine, "CHROMA_PATH", str(tmp_path))
    monkeypatch.setattr(rag_engine, "_chroma_client", None)
    yield
    monkeypatch.setattr(rag_engine, "_chroma_client", None)
