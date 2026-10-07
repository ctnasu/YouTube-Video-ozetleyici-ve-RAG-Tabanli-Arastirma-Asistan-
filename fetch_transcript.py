"""
YouTube Transkript Çekici
- get_clean_transcript(): düz metin (mevcut, geriye uyumlu)
- get_timestamped_transcript(): zaman damgalı parçalar döndürür
"""

import os
from typing import Union, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


# Desteklenen diller — öncelik sırasıyla denenir
SUPPORTED_LANGUAGES = ['tr', 'en', 'es', 'de', 'fr', 'ja', 'ko']

# ASR (Whisper) fallback yapılandırması
WHISPER_MODEL_SIZE = "small"
_whisper_model = None


def _get_whisper_model():
    """faster-whisper modelini tek seferlik (singleton) yükler."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def _get_proxies() -> Optional[dict]:
    """
    YT_PROXY_URL env değişkeni ayarlıysa istekleri bu proxy üzerinden yönlendirir.
    YouTube'un IP bazlı bot engeline takılan kullanıcılar için (ör. Webshare gibi bir
    residential proxy) isteğe bağlı bir kaçış yolu. Örnek:
        export YT_PROXY_URL="http://kullanici:sifre@proxy-host:port"
    """
    proxy_url = os.environ.get("YT_PROXY_URL")
    if proxy_url:
        return {"http": proxy_url, "https": proxy_url}
    return None


def _extract_video_id(video_url: str):
    """YouTube URL'sinden video ID'sini çıkarır (watch, youtu.be, shorts, embed formatları)."""
    if "v=" in video_url:
        return video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1].split("?")[0]
    elif "shorts/" in video_url:
        return video_url.split("shorts/")[1].split("?")[0].split("&")[0]
    elif "embed/" in video_url:
        return video_url.split("embed/")[1].split("?")[0].split("&")[0]
    return None


def _fetch_raw_transcript(video_id: str) -> list[dict]:
    """
    YouTube API'den ham transcript verisini çeker.
    Öncelik: SUPPORTED_LANGUAGES > auto-generated altyazı > çevrilebilir herhangi bir altyazı (Türkçe'ye çevrilir).
    Dönüş: [{"text": "...", "start": 10.5, "duration": 3.2}, ...]
    """
    proxies = _get_proxies()
    try:
        return YouTubeTranscriptApi.get_transcript(
            video_id, languages=SUPPORTED_LANGUAGES, proxies=proxies
        )
    except (TranscriptsDisabled, NoTranscriptFound):
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, proxies=proxies)

        # Auto-generated altyazı dene
        for t in transcript_list:
            if t.is_generated:
                return t.fetch()

        # Desteklenen dillerde/otomatik altyazı yoksa: mevcut herhangi bir
        # altyazıyı bul ve Türkçe'ye çevirmeyi dene (ör. sadece Portekizce altyazısı olan video)
        for t in transcript_list:
            if t.is_translatable:
                return t.translate("tr").fetch()

        # Çevrilemeyen ama var olan ilk altyazıyı olduğu gibi döndür
        for t in transcript_list:
            return t.fetch()

        raise NoTranscriptFound(video_id, SUPPORTED_LANGUAGES, transcript_list)


def _fetch_transcript_via_asr(video_id: str) -> list[dict]:
    """
    Son çare yöntemi: Altyazı hiç yoksa/erişilemiyorsa, yt-dlp ile videonun sesini
    indirir ve faster-whisper (yerel/offline) ile videonun kendi dilinde transkript
    çıkarır. Çeviri yapılmaz — Türkçe'ye çeviri, LLM özet/chat aşamasında yapılır.

    Dönüş: [{"start": 10.5, "duration": 3.2, "text": "..."}, ...]
    """
    import tempfile
    import glob
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmp_dir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmp_dir, "audio.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        proxy_url = os.environ.get("YT_PROXY_URL")
        if proxy_url:
            ydl_opts["proxy"] = proxy_url
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        audio_files = glob.glob(os.path.join(tmp_dir, "audio.*"))
        if not audio_files:
            raise RuntimeError("Ses dosyası indirilemedi.")
        audio_path = audio_files[0]

        model = _get_whisper_model()
        segments, _info = model.transcribe(audio_path, beam_size=5)

        result = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                result.append({
                    "start": seg.start,
                    "duration": max(0.0, seg.end - seg.start),
                    "text": text,
                })
        return result


def _clean_transcript_items(raw: list[dict]) -> list[dict]:
    """Ham transkript parçalarını (altyazı veya ASR kaynaklı) tekdüze formata temizler."""
    result = []
    for item in raw:
        cleaned_text = item['text'].replace("\n", " ").strip()
        if cleaned_text:
            result.append({
                "start": item.get("start", 0),
                "duration": item.get("duration", 0),
                "text": cleaned_text
            })
    return result


def get_timestamped_transcript(video_url: str) -> Union[list, str]:
    """
    Zaman damgalı transkript döndürür. Önce altyazıları dener; hiçbiri
    çalışmazsa (kapalı/bulunamadı/bot engeli) son çare olarak sesten
    Whisper ile transkript çıkarmayı dener.

    Dönüş: [{"start": 10.5, "duration": 3.2, "text": "..."}, ...]
    Hata durumunda string döner.
    """
    video_id = _extract_video_id(video_url)
    if not video_id:
        return "Hata: Geçersiz YouTube URL'si."

    try:
        raw = _fetch_raw_transcript(video_id)
        return _clean_transcript_items(raw)
    except TranscriptsDisabled:
        caption_error = "Bu videoda altyazılar kapalı veya bulunmuyor."
    except NoTranscriptFound:
        caption_error = "Bu videoda desteklenen dilde altyazı bulunamadı."
    except Exception as e:
        error_msg = str(e).lower()
        if "no element found" in error_msg or "xml" in error_msg:
            caption_error = (
                "YouTube'un bot koruma sistemine takıldınız (sık istek atılması "
                "veya IP kısıtlaması nedeniyle YouTube altyazıyı vermeyi reddetti)."
            )
        else:
            caption_error = f"Altyazı çekilemedi: {e}"

    # Altyazı yolu başarısız oldu — son çare: sesten transkript çıkar (ASR)
    try:
        raw = _fetch_transcript_via_asr(video_id)
        return _clean_transcript_items(raw)
    except Exception as asr_error:
        return (
            f"Hata: {caption_error}\n"
            f"Sesten transkript çıkarma (yedek yöntem) da başarısız oldu: {asr_error}\n"
            "Farklı bir ağ/VPN deneyin veya YT_PROXY_URL ortam değişkeniyle bir proxy tanımlayın."
        )


def get_clean_transcript(video_url: str) -> str:
    """
    YouTube URL'sinden düz metin transkript döndürür (geriye uyumlu).
    """
    result = get_timestamped_transcript(video_url)

    if isinstance(result, str):
        return result  # Hata mesajı

    # Zaman damgalı parçalardan düz metin oluştur
    clean_text = " ".join(item["text"] for item in result)
    # Çift boşlukları temizle
    while "  " in clean_text:
        clean_text = clean_text.replace("  ", " ")

    return clean_text


def format_timestamp(seconds: float) -> str:
    """Saniyeyi MM:SS (1 saatten kısa) ya da H:MM:SS (1 saat ve üzeri) formatına çevirir."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"
