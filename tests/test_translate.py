"""Tests for translation module (new MarianMT implementation with load/unload)."""

import pytest

from src.models.translator import Translator


@pytest.fixture(scope="module")
def translator():
    """Load translator once for all tests (model loading is expensive)."""
    t = Translator(device="cpu")
    t.load()
    yield t
    t.unload()


class TestTranslator:
    def test_simple_translation(self, translator):
        german = "Hallo, wie geht es Ihnen?"
        english = translator.translate(german)
        assert isinstance(english, str)
        assert len(english) > 0
        # Should contain some English words
        english_lower = english.lower()
        assert any(word in english_lower for word in ["hello", "how", "you", "are"])

    def test_news_translation(self, translator):
        german = "Der Bundeskanzler hat neue Wirtschaftspolitik angekündigt."
        english = translator.translate(german)
        assert isinstance(english, str)
        assert len(english) > 0

    def test_batch_translation(self, translator):
        texts = [
            "Hallo Welt",
            "Die Wirtschaft wächst.",
        ]
        results = translator.translate_batch(texts, batch_size=2)
        assert len(results) == 2
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_empty_batch(self, translator):
        results = translator.translate_batch([])
        assert results == []

    def test_load_unload(self):
        """Test explicit load/unload cycle."""
        t = Translator(device="cpu")
        assert not t.is_loaded

        t.load()
        assert t.is_loaded

        t.unload()
        assert not t.is_loaded
