"""Tests for language detection module."""

import pytest

from src.langdetect_util import detect_language


class TestDetectLanguage:
    def test_detects_english(self):
        text = "The president announced new economic policies during the summit."
        assert detect_language(text) == "en"

    def test_detects_german(self):
        text = "Der Bundeskanzler hat neue Wirtschaftspolitik angekündigt."
        assert detect_language(text) == "de"

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="empty"):
            detect_language("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            detect_language("   ")

    def test_unsupported_language_raises(self):
        # French text should raise ValueError
        with pytest.raises(ValueError, match="Unsupported"):
            detect_language("Le président a annoncé de nouvelles politiques économiques.")
