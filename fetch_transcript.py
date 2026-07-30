"""
YouTube Transkript Çekici
- get_clean_transcript(): düz metin (mevcut, geriye uyumlu)
- get_timestamped_transcript(): zaman damgalı parçalar döndürür
"""

from typing import Union
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


# Desteklenen diller — öncelik sırasıyla denenir
SUPPORTED_LANGUAGES = ['tr', 'en', 'es', 'de', 'fr', 'ja', 'ko']


def _extract_video_id(video_url: str):
    """YouTube URL'sinden video ID'sini çıkarır."""
    if "v=" in video_url:
        return video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1].split("?")[0]
    return None


def _fetch_raw_transcript(video_id: str) -> list[dict]:
    """
    YouTube API'den ham transcript verisini çeker.
    Dönüş: [{"text": "...", "start": 10.5, "duration": 3.2}, ...]
    """
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=SUPPORTED_LANGUAGES)
    except (TranscriptsDisabled, NoTranscriptFound):
        # Manuel/otomatik altyazı yoksa, auto-generated dene
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for t in transcript_list:
            if t.is_generated:
                return t.fetch()
        raise


def get_timestamped_transcript(video_url: str) -> Union[list, str]:
    """
    Zaman damgalı transkript döndürür.
    Dönüş: [{"start": 10.5, "duration": 3.2, "text": "..."}, ...]
    Hata durumunda string döner.
    """
    try:
        video_id = _extract_video_id(video_url)
        if not video_id:
            return "Hata: Geçersiz YouTube URL'si."

        raw = _fetch_raw_transcript(video_id)

        # Temizle ve döndür
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

    except TranscriptsDisabled:
        return "Hata: Bu videoda altyazılar kapalı veya bulunmuyor."
    except NoTranscriptFound:
        return "Hata: Bu videoda desteklenen dilde altyazı bulunamadı."
    except Exception as e:
        error_msg = str(e).lower()
        if "no element found" in error_msg or "xml" in error_msg:
            return (
                "Hata: YouTube'un bot koruma sistemine takıldınız (Sık istek atılması "
                "veya IP kısıtlaması nedeniyle YouTube altyazıyı vermeyi reddetti). "
                "Lütfen biraz bekleyip tekrar deneyin veya farklı bir ağ/VPN kullanarak test edin."
            )
        return f"Hata oluştu: {str(e)}"


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
    """Saniyeyi MM:SS formatına çevirir."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"
