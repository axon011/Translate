"""German-to-English translation using Helsinki-NLP MarianMT.

Used for the comparison pipeline (Translate-then-NER) to benchmark
against cross-lingual NER. Also useful for translating German news
before summarization (since DistilBART is English-only).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import MarianMTModel, MarianTokenizer

from src.utils.config import TranslatorConfig, get_config
from src.utils.logging import TimingContext, get_logger

logger = get_logger("translator")


@dataclass
class TranslationResult:
    """Result of translation."""

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str


class Translator:
    """MarianMT wrapper for DE->EN translation.

    Explicit load/unload for VRAM management on 4GB GPU.
    """

    def __init__(
        self,
        config: TranslatorConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or get_config().translator
        self.device = device or self.config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load model onto device."""
        if self._loaded:
            return

        logger.info(
            "Loading translator model",
            extra={"component": "translator", "model": self.config.model_id},
        )

        self._tokenizer = MarianTokenizer.from_pretrained(self.config.model_id)
        self._model = MarianMTModel.from_pretrained(self.config.model_id)
        self._model.to(self.device)
        self._model.eval()

        self._loaded = True

        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**2
            logger.info(
                f"Translator loaded, VRAM: {vram:.0f} MB",
                extra={"component": "translator", "vram_mb": round(vram, 1)},
            )

    def unload(self) -> None:
        """Unload model from GPU to free VRAM."""
        if not self._loaded:
            return

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Translator unloaded", extra={"component": "translator"})

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        """Translate a single German text to English.

        Args:
            text: German input text.

        Returns:
            Translated English string.
        """
        if not self._loaded:
            self.load()

        with TimingContext("translate_inference") as t:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            output_ids = self._model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
            )
            translated = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        logger.debug(
            f"Translated in {t.elapsed_ms:.1f}ms",
            extra={"component": "translator", "latency_ms": round(t.elapsed_ms, 1)},
        )

        return translated

    @torch.inference_mode()
    def translate_batch(self, texts: list[str], batch_size: int = 4) -> list[str]:
        """Translate a batch of German texts to English.

        Args:
            texts: List of German input texts.
            batch_size: Number of texts per batch (keep low for 4GB VRAM).

        Returns:
            List of translated English strings.
        """
        if not self._loaded:
            self.load()

        results: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            output_ids = self._model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_beams=self.config.num_beams,
            )
            decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)

        return results
