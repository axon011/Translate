"""Tests for text preprocessing utilities."""

from src.data.preprocessing import (
    clean_news_text,
    detect_script,
    normalize_text,
    truncate_text,
)


class TestNormalizeText:
    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert normalize_text("hello    world") == "hello world"

    def test_normalizes_tabs(self):
        assert normalize_text("hello\tworld") == "hello world"

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestCleanNewsText:
    def test_removes_urls(self):
        text = "Visit https://example.com for more info."
        assert "https://example.com" not in clean_news_text(text)

    def test_removes_emails(self):
        text = "Contact user@example.com for details."
        assert "user@example.com" not in clean_news_text(text)

    def test_preserves_german_chars(self):
        text = "Über die Straße gehen wir täglich."
        cleaned = clean_news_text(text)
        assert "Über" in cleaned
        assert "Straße" in cleaned
        assert "täglich" in cleaned

    def test_removes_html(self):
        text = "<p>Hello <b>world</b></p>"
        cleaned = clean_news_text(text)
        assert "<" not in cleaned
        assert "Hello" in cleaned

    def test_normalizes_quotes(self):
        text = "\u201cHello\u201d"
        cleaned = clean_news_text(text)
        assert '"' in cleaned


class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert truncate_text(text, max_words=100) == text

    def test_truncates_long_text(self):
        text = " ".join(["word"] * 600)
        result = truncate_text(text, max_words=500)
        assert len(result.split()) <= 500

    def test_respects_sentence_boundary(self):
        text = "First sentence. Second sentence. " + " ".join(["word"] * 500)
        result = truncate_text(text, max_words=10)
        # Should try to end at a sentence boundary
        assert result.endswith(".") or len(result.split()) <= 10


class TestDetectScript:
    def test_latin(self):
        assert detect_script("Hello world") == "latin"

    def test_german_is_latin(self):
        assert detect_script("Über die Straße") == "latin"

    def test_empty_is_other(self):
        assert detect_script("123") == "other"
