"""Content summarization using DistilBART.

Generates concise 1-3 sentence summaries of news articles.
Mirrors Voize's structured documentation output.

Uses sshleifer/distilbart-cnn-12-6 for abstractive summarization.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.config import SummarizerConfig, get_config
from src.utils.logging import TimingContext, get_logger

logger = get_logger("summarizer")


@dataclass
class SummaryResult:
    """Result of text summarization."""

    summary: str
    input_length: int
    output_length: int


class Summarizer:
    """DistilBART-based text summarizer.

    Key design choices:
    - distilbart-cnn-12-6: good quality with manageable VRAM (~0.8GB FP16)
    - Abstractive summarization (generates new sentences)
    - Configurable length via min/max tokens
    - Explicit load/unload for VRAM management
    """

    def __init__(
        self,
        config: SummarizerConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or get_config().summarizer
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
            "Loading summarizer model",
            extra={"component": "summarizer", "model": self.config.model_id},
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_id)

        # FP16 for GPU
        if self.device == "cuda":
            self._model = self._model.half()

        self._model.to(self.device)
        self._model.eval()

        self._loaded = True

        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**2
            logger.info(
                f"Summarizer loaded, VRAM: {vram:.0f} MB",
                extra={"component": "summarizer", "vram_mb": round(vram, 1)},
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

        logger.info("Summarizer unloaded", extra={"component": "summarizer"})

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.inference_mode()
    def summarize(self, text: str) -> SummaryResult:
        """Generate a summary of the input text.

        Args:
            text: Input text to summarize (typically a news article).

        Returns:
            SummaryResult with summary text and token counts.
        """
        if not text or not text.strip():
            return SummaryResult(summary="", input_length=0, output_length=0)

        if not self._loaded:
            self.load()

        with TimingContext("summarize_inference") as t:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,  # BART max input
            ).to(self.device)

            summary_ids = self._model.generate(
                **inputs,
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                length_penalty=self.config.length_penalty,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping,
            )

            summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        input_len = inputs["input_ids"].shape[1]
        output_len = summary_ids.shape[1]

        logger.debug(
            f"Summarized {input_len} -> {output_len} tokens in {t.elapsed_ms:.1f}ms",
            extra={
                "component": "summarizer",
                "latency_ms": round(t.elapsed_ms, 1),
            },
        )

        return SummaryResult(
            summary=summary.strip(),
            input_length=input_len,
            output_length=output_len,
        )

    @torch.inference_mode()
    def summarize_batch(self, texts: list[str]) -> list[SummaryResult]:
        """Summarize multiple texts."""
        return [self.summarize(t) for t in texts]
