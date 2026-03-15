"""Main pipeline orchestration: cross-lingual NER with sequential VRAM management.

Primary Pipeline:
    Audio → Whisper ASR → Language Detection → Cross-Lingual NER
    → Event Classification → Summarization → Structured JSON

Key design:
- Sequential model loading/unloading to fit in 4GB VRAM
- Cross-lingual NER (no translation needed for entity extraction)
- Translation only used for summarization (DistilBART is English-only)
"""

from __future__ import annotations

import gc
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.langdetect_util import detect_language
from src.models.classifier import ClassificationResult, EventClassifier
from src.models.ner import Entity, NERExtractor
from src.models.summarizer import Summarizer, SummaryResult
from src.models.translator import Translator
from src.utils.config import PipelineConfig, get_config
from src.utils.logging import TimingContext, get_logger

logger = get_logger("pipeline")


@dataclass
class PipelineResult:
    """Structured output from the full pipeline."""

    request_id: str
    original_text: str
    detected_language: str
    entities: list[Entity]
    classification: ClassificationResult
    summary: SummaryResult | None
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "request_id": self.request_id,
            "original_text": self.original_text,
            "detected_language": self.detected_language,
            "entities": [asdict(e) for e in self.entities],
            "classification": asdict(self.classification),
            "summary": asdict(self.summary) if self.summary else None,
            "timings": self.timings,
        }


def _clear_gpu_cache() -> None:
    """Free GPU memory between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class NewsPipeline:
    """End-to-end multilingual news NLP pipeline.

    Pipeline stages:
        1. Language Detection (langdetect) - CPU, instant
        2. Cross-Lingual NER (XLM-RoBERTa) - directly on German text
        3. Event Classification (DistilBERT multilingual) - directly on German text
        4. Translation DE→EN (MarianMT) - only if summarization needed
        5. Summarization (DistilBART) - English only

    VRAM Strategy:
        - sequential_mode=True: Load one model at a time (for 4GB GPUs)
        - sequential_mode=False: Keep NER + Classifier loaded (~2.2GB combined)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        classifier_model_path: str | None = None,
        sequential_mode: bool | None = None,
        enable_summary: bool = True,
        device: str | None = None,
    ) -> None:
        """Initialize pipeline components (lazy loading).

        Args:
            config: Pipeline configuration.
            classifier_model_path: Path to fine-tuned classifier.
            sequential_mode: If True, load/unload models sequentially.
                            Defaults to config.hardware.enable_sequential_loading.
            enable_summary: Whether to run summarization.
            device: Device override.
        """
        self.config = config or get_config()
        self.sequential_mode = (
            sequential_mode
            if sequential_mode is not None
            else self.config.hardware.enable_sequential_loading
        )
        self.enable_summary = enable_summary

        # Default to fine-tuned classifier if available
        if classifier_model_path is None:
            default_path = Path("models/event_classifier")
            if default_path.exists():
                classifier_model_path = str(default_path)

        # Initialize components (models loaded lazily)
        self.ner = NERExtractor(config=self.config.ner, device=device)
        self.classifier = EventClassifier(
            config=self.config.classifier,
            model_path=classifier_model_path,
            device=device,
        )
        self.summarizer = Summarizer(config=self.config.summarizer, device=device)
        self.translator = Translator(config=self.config.translator, device=device)

        logger.info(
            f"Pipeline initialized (sequential={self.sequential_mode}, "
            f"summary={self.enable_summary})",
            extra={"component": "pipeline"},
        )

    def _load_model(self, model: Any, name: str) -> None:
        """Load a model, clearing GPU cache first if in sequential mode."""
        if self.sequential_mode:
            _clear_gpu_cache()
        model.load()

    def _unload_model(self, model: Any, name: str) -> None:
        """Unload a model if in sequential mode."""
        if self.sequential_mode:
            model.unload()

    def run(self, text: str, include_summary: bool | None = None) -> PipelineResult:
        """Process a single text through the full pipeline.

        Args:
            text: Input news article text (German or English).
            include_summary: Override for whether to include summary.

        Returns:
            PipelineResult with entities, classification, summary, and timings.
        """
        request_id = str(uuid.uuid4())[:8]
        timings: dict[str, float] = {}
        do_summary = include_summary if include_summary is not None else self.enable_summary

        logger.info(
            f"[{request_id}] Processing text ({len(text)} chars)",
            extra={"component": "pipeline"},
        )

        # 1. Language detection (CPU, instant)
        with TimingContext("lang_detect", sync_cuda=False) as t:
            lang = detect_language(text)
        timings["lang_detect_ms"] = round(t.elapsed_ms, 2)

        # 2. Cross-lingual NER (works on both German and English)
        self._load_model(self.ner, "ner")
        with TimingContext("ner") as t:
            entities = self.ner.extract(text)
        timings["ner_ms"] = round(t.elapsed_ms, 2)
        self._unload_model(self.ner, "ner")

        # 3. Event classification (multilingual, works on German directly)
        self._load_model(self.classifier, "classifier")
        with TimingContext("classifier") as t:
            classification = self.classifier.classify(text)
        timings["classification_ms"] = round(t.elapsed_ms, 2)
        self._unload_model(self.classifier, "classifier")

        # 4. Summarization (optional, English-only model)
        summary = None
        if do_summary:
            # If German, translate first
            english_text = text
            if lang == "de":
                self._load_model(self.translator, "translator")
                with TimingContext("translation") as t:
                    english_text = self.translator.translate(text)
                timings["translation_ms"] = round(t.elapsed_ms, 2)
                self._unload_model(self.translator, "translator")

            self._load_model(self.summarizer, "summarizer")
            with TimingContext("summarize") as t:
                summary = self.summarizer.summarize(english_text)
            timings["summarization_ms"] = round(t.elapsed_ms, 2)
            self._unload_model(self.summarizer, "summarizer")

        timings["total_ms"] = round(sum(timings.values()), 2)

        result = PipelineResult(
            request_id=request_id,
            original_text=text,
            detected_language=lang,
            entities=entities,
            classification=classification,
            summary=summary,
            timings=timings,
        )

        logger.info(
            f"[{request_id}] Done: {len(entities)} entities, "
            f"{classification.label} ({classification.score:.2f}), "
            f"{timings['total_ms']:.0f}ms total",
            extra={
                "component": "pipeline",
                "latency_ms": timings["total_ms"],
                "items": len(entities),
            },
        )

        return result

    def run_batch(self, texts: list[str]) -> list[PipelineResult]:
        """Process multiple texts through the pipeline.

        In sequential mode, processes texts one at a time.
        In non-sequential mode, keeps models loaded across texts.

        Args:
            texts: List of input texts.

        Returns:
            List of PipelineResult objects.
        """
        if not self.sequential_mode:
            # Load all models once
            self.ner.load()
            self.classifier.load()

        results = [self.run(t) for t in texts]

        if not self.sequential_mode:
            self.ner.unload()
            self.classifier.unload()

        return results

    def run_with_audio(
        self,
        audio_path: str | Path,
        include_summary: bool | None = None,
    ) -> tuple[str, PipelineResult]:
        """Process an audio file through the full pipeline.

        Steps: ASR → Language Detection → NER → Classification → Summary

        Args:
            audio_path: Path to audio file.
            include_summary: Override for summary inclusion.

        Returns:
            Tuple of (transcribed_text, PipelineResult).
        """
        from src.models.asr import ASRModel

        asr = ASRModel(config=self.config.asr)
        self._load_model(asr, "asr")

        with TimingContext("asr") as t:
            transcription = asr.transcribe(audio_path)

        self._unload_model(asr, "asr")

        logger.info(
            f"ASR: transcribed {transcription.duration_s:.1f}s audio in {t.elapsed_ms:.0f}ms",
            extra={"component": "pipeline", "latency_ms": round(t.elapsed_ms, 1)},
        )

        # Run text pipeline on transcribed text
        result = self.run(transcription.text, include_summary=include_summary)
        result.timings["asr_ms"] = round(t.elapsed_ms, 2)
        result.timings["total_ms"] = round(sum(result.timings.values()), 2)

        return transcription.text, result
