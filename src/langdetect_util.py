"""Language detection module using langdetect."""

from langdetect import DetectorFactory, detect

# Make detection deterministic
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en", "de"}


def detect_language(text: str) -> str:
    """Detect the language of the input text.

    Args:
        text: Input text string.

    Returns:
        ISO 639-1 language code ("en" or "de").

    Raises:
        ValueError: If the detected language is not supported.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty")

    lang = detect(text)

    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{lang}'. Supported: {SUPPORTED_LANGUAGES}")

    return lang
