"""Tests for NER module (new cross-lingual implementation)."""

import pytest

from src.models.ner import NERExtractor, ENTITY_TYPES


@pytest.fixture(scope="module")
def ner():
    extractor = NERExtractor(device="cpu")
    extractor.load()
    yield extractor
    extractor.unload()


class TestNERExtractor:
    def test_extracts_person_english(self, ner):
        text = "Angela Merkel visited Berlin yesterday."
        entities = ner.extract(text)
        labels = {e.label for e in entities}
        texts = {e.text for e in entities}
        assert "PER" in labels
        assert any("Merkel" in t for t in texts)

    def test_extracts_person_german(self, ner):
        """Cross-lingual: should extract German entities directly."""
        text = "Angela Merkel besuchte gestern Berlin."
        entities = ner.extract(text)
        labels = {e.label for e in entities}
        assert "PER" in labels

    def test_extracts_location(self, ner):
        text = "The conference was held in Berlin, Germany."
        entities = ner.extract(text)
        labels = {e.label for e in entities}
        assert "LOC" in labels

    def test_extracts_organization(self, ner):
        text = "Microsoft announced a partnership with the United Nations."
        entities = ner.extract(text)
        labels = {e.label for e in entities}
        assert "ORG" in labels

    def test_empty_text_returns_empty(self, ner):
        assert ner.extract("") == []
        assert ner.extract("   ") == []

    def test_entity_has_valid_fields(self, ner):
        text = "Elon Musk founded SpaceX in Los Angeles."
        entities = ner.extract(text)
        assert len(entities) > 0
        for e in entities:
            assert 0.0 <= e.score <= 1.0
            assert e.label in ENTITY_TYPES
            assert e.start >= 0
            assert e.end > e.start
            assert len(e.text) > 0

    def test_batch_extraction(self, ner):
        texts = [
            "Angela Merkel is from Germany.",
            "Elon Musk runs Tesla.",
        ]
        results = ner.extract_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_load_unload(self):
        """Test explicit load/unload cycle."""
        extractor = NERExtractor(device="cpu")
        assert not extractor.is_loaded

        extractor.load()
        assert extractor.is_loaded

        extractor.unload()
        assert not extractor.is_loaded
