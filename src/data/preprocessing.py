"""Text preprocessing and normalization utilities.

Handles cleaning, normalization, and text preparation for the NLP pipeline.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize unicode characters and whitespace.

    Args:
        text: Raw input text.

    Returns:
        Normalized text with consistent unicode and spacing.
    """
    # Unicode normalization (NFC for composed characters)
    text = unicodedata.normalize("NFC", text)

    # Normalize whitespace (tabs, multiple spaces, etc.)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_news_text(text: str) -> str:
    """Clean news article text for NLP processing.

    Removes:
    - URLs
    - Email addresses
    - Excessive punctuation
    - Leading/trailing whitespace

    Preserves:
    - German special characters (umlauts, eszett)
    - Sentence structure
    - Named entities (capitalization)

    Args:
        text: Raw news article text.

    Returns:
        Cleaned text suitable for NER and classification.
    """
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove HTML tags if any
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u201e", '"').replace("\u201f", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Remove excessive punctuation (3+ of the same)
    text = re.sub(r"([!?.]){3,}", r"\1\1", text)

    # Normalize whitespace
    text = normalize_text(text)

    return text


def truncate_text(text: str, max_words: int = 500) -> str:
    """Truncate text to a maximum number of words.

    Preserves sentence boundaries where possible.

    Args:
        text: Input text.
        max_words: Maximum number of words.

    Returns:
        Truncated text.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])

    # Try to end at a sentence boundary
    last_period = truncated.rfind(".")
    last_excl = truncated.rfind("!")
    last_quest = truncated.rfind("?")
    last_boundary = max(last_period, last_excl, last_quest)

    if last_boundary > len(truncated) * 0.7:
        truncated = truncated[: last_boundary + 1]

    return truncated


def detect_script(text: str) -> str:
    """Detect whether text is primarily Latin or other script.

    Args:
        text: Input text.

    Returns:
        "latin", "cyrillic", "cjk", or "other".
    """
    latin_count = sum(1 for c in text if unicodedata.category(c).startswith("L") and ord(c) < 0x250)
    total_alpha = sum(1 for c in text if unicodedata.category(c).startswith("L"))

    if total_alpha == 0:
        return "other"

    if latin_count / total_alpha > 0.8:
        return "latin"

    return "other"
