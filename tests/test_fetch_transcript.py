import fetch_transcript as ft


class TestExtractVideoId:
    def test_watch_url(self):
        assert ft._extract_video_id("https://www.youtube.com/watch?v=aircAruvnKk") == "aircAruvnKk"

    def test_watch_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=aircAruvnKk&t=42s&list=PL123"
        assert ft._extract_video_id(url) == "aircAruvnKk"

    def test_youtu_be_short_url(self):
        assert ft._extract_video_id("https://youtu.be/aircAruvnKk") == "aircAruvnKk"

    def test_youtu_be_with_query(self):
        assert ft._extract_video_id("https://youtu.be/aircAruvnKk?t=5") == "aircAruvnKk"

    def test_shorts_url(self):
        assert ft._extract_video_id("https://www.youtube.com/shorts/aircAruvnKk") == "aircAruvnKk"

    def test_embed_url(self):
        assert ft._extract_video_id("https://www.youtube.com/embed/aircAruvnKk") == "aircAruvnKk"

    def test_invalid_url_returns_none(self):
        assert ft._extract_video_id("https://example.com/not-a-youtube-link") is None


class TestFormatTimestamp:
    def test_zero(self):
        assert ft.format_timestamp(0) == "00:00"

    def test_under_a_minute(self):
        assert ft.format_timestamp(45) == "00:45"

    def test_over_a_minute(self):
        assert ft.format_timestamp(65) == "01:05"

    def test_over_an_hour_worth_of_seconds(self):
        # 1 saat ve üzeri süreler H:MM:SS formatına geçer.
        assert ft.format_timestamp(3661) == "1:01:01"

    def test_fractional_seconds_are_truncated(self):
        assert ft.format_timestamp(65.9) == "01:05"


class TestCleanTranscriptItems:
    def test_strips_newlines_and_whitespace(self):
        raw = [{"start": 0, "duration": 2, "text": "Merhaba\ndünya  "}]
        result = ft._clean_transcript_items(raw)
        assert result == [{"start": 0, "duration": 2, "text": "Merhaba dünya"}]

    def test_drops_empty_text_items(self):
        raw = [
            {"start": 0, "duration": 1, "text": "  "},
            {"start": 1, "duration": 1, "text": "gerçek metin"},
        ]
        result = ft._clean_transcript_items(raw)
        assert len(result) == 1
        assert result[0]["text"] == "gerçek metin"

    def test_missing_start_duration_default_to_zero(self):
        result = ft._clean_transcript_items([{"text": "merhaba"}])
        assert result[0]["start"] == 0
        assert result[0]["duration"] == 0


class TestGetProxies:
    def test_returns_none_without_env_var(self, monkeypatch):
        monkeypatch.delenv("YT_PROXY_URL", raising=False)
        assert ft._get_proxies() is None

    def test_returns_proxy_dict_with_env_var(self, monkeypatch):
        monkeypatch.setenv("YT_PROXY_URL", "http://user:pass@proxy.example.com:8080")
        proxies = ft._get_proxies()
        assert proxies == {
            "http": "http://user:pass@proxy.example.com:8080",
            "https": "http://user:pass@proxy.example.com:8080",
        }


class TestGetTimestampedTranscriptInvalidUrl:
    def test_invalid_url_returns_error_string(self):
        result = ft.get_timestamped_transcript("https://example.com/not-a-video")
        assert isinstance(result, str)
        assert "Hata" in result
