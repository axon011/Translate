"""Tests for event classification module (new 4-class implementation)."""

import pytest

from src.models.classifier import EventClassifier, LABEL2ID, ID2LABEL


@pytest.fixture(scope="module")
def classifier():
    clf = EventClassifier(device="cpu")
    clf.load()
    yield clf
    clf.unload()


class TestEventClassifier:
    def test_returns_valid_label(self, classifier):
        text = "The stock market crashed after the Federal Reserve raised interest rates."
        result = classifier.classify(text)
        assert result.label in LABEL2ID

    def test_returns_score_in_range(self, classifier):
        text = "The football team won the championship."
        result = classifier.classify(text)
        assert 0.0 <= result.score <= 1.0

    def test_returns_all_scores(self, classifier):
        text = "The prime minister held a press conference."
        result = classifier.classify(text)
        assert set(result.all_scores.keys()) == set(LABEL2ID.keys())
        assert all(0.0 <= v <= 1.0 for v in result.all_scores.values())

    def test_four_classes_only(self, classifier):
        """Ensure we have exactly 4 classes, not 5."""
        assert len(LABEL2ID) == 4
        assert "Political" in LABEL2ID
        assert "Economic" in LABEL2ID
        assert "Sports" in LABEL2ID
        assert "Technology" in LABEL2ID
        # Old "crime" class should NOT exist
        assert "crime" not in LABEL2ID
        assert "Crime" not in LABEL2ID

    def test_batch_classification(self, classifier):
        texts = [
            "The prime minister held a press conference.",
            "Tech stocks surged after the earnings report.",
        ]
        results = classifier.classify_batch(texts)
        assert len(results) == 2
        assert all(r.label in LABEL2ID for r in results)

    def test_load_unload(self):
        """Test explicit load/unload cycle."""
        clf = EventClassifier(device="cpu")
        assert not clf.is_loaded

        clf.load()
        assert clf.is_loaded

        clf.unload()
        assert not clf.is_loaded

    def test_german_text(self, classifier):
        """Multilingual model should handle German text."""
        text = "Der Bundeskanzler hat neue Maßnahmen angekündigt."
        result = classifier.classify(text)
        assert result.label in LABEL2ID
        assert result.score > 0.0
