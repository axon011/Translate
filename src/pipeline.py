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
        # NER + Classifier fit together (~1.6GB), so load both before unloading
        if not self.ner.is_loaded:
            self._load_model(self.ner, "ner")
        with TimingContext("ner") as t:
            entities = self.ner.extract(text)
        timings["ner_ms"] = round(t.elapsed_ms, 2)

        # 3. Event classification (multilingual, works on German directly)
        if not self.classifier.is_loaded:
            self._load_model(self.classifier, "classifier")
        with TimingContext("classifier") as t:
            classification = self.classifier.classify(text)
        timings["classification_ms"] = round(t.elapsed_ms, 2)

        # Now unload NER + Classifier only if we need VRAM for summarization
        if do_summary and self.sequential_mode:
            self._unload_model(self.ner, "ner")
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

    def run_batch(
        self, texts: list[str], include_summary: bool | None = None
    ) -> list[PipelineResult]:
        """Process multiple texts through the pipeline with stage-batched VRAM.

        Loads each model ONCE, processes ALL texts, then moves to the next model.
        For 3 articles this means 4 model loads instead of 12.

        Args:
            texts: List of input texts.
            include_summary: Override for whether to include summary.

        Returns:
            List of PipelineResult objects.
        """
        if not texts:
            return []

        do_summary = (
            include_summary if include_summary is not None else self.enable_summary
        )
        n = len(texts)
        request_ids = [str(uuid.uuid4())[:8] for _ in range(n)]
        all_timings: list[dict[str, float]] = [{} for _ in range(n)]

        for i, text in enumerate(texts):
            logger.info(
                f"[{request_ids[i]}] Processing text ({len(text)} chars)",
                extra={"component": "pipeline"},
            )

        # 1. Language detection (CPU, instant) - all texts
        langs = []
        for i, text in enumerate(texts):
            with TimingContext("lang_detect", sync_cuda=False) as t:
                langs.append(detect_language(text))
            all_timings[i]["lang_detect_ms"] = round(t.elapsed_ms, 2)

        # 2. NER - load once, process all
        _clear_gpu_cache()
        self.ner.load()
        all_entities = []
        for i, text in enumerate(texts):
            with TimingContext("ner") as t:
                all_entities.append(self.ner.extract(text))
            all_timings[i]["ner_ms"] = round(t.elapsed_ms, 2)

        # 3. Classification - NER+Classifier fit together (~1.6GB)
        self.classifier.load()
        all_classifications = []
        for i, text in enumerate(texts):
            with TimingContext("classifier") as t:
                all_classifications.append(self.classifier.classify(text))
            all_timings[i]["classification_ms"] = round(t.elapsed_ms, 2)

        # Unload both before loading translator/summarizer
        self.ner.unload()
        self.classifier.unload()
        _clear_gpu_cache()

        # 4. Translation (only German texts, only if summarizing)
        all_english_texts = list(texts)
        if do_summary:
            de_indices = [i for i, lang in enumerate(langs) if lang == "de"]
            if de_indices:
                self.translator.load()
                for i in de_indices:
                    with TimingContext("translation") as t:
                        all_english_texts[i] = self.translator.translate(texts[i])
                    all_timings[i]["translation_ms"] = round(t.elapsed_ms, 2)
                self.translator.unload()
                _clear_gpu_cache()

        # 5. Summarization - load once, summarize all
        all_summaries: list[SummaryResult | None] = [None] * n
        if do_summary:
            self.summarizer.load()
            for i in range(n):
                with TimingContext("summarize") as t:
                    all_summaries[i] = self.summarizer.summarize(all_english_texts[i])
                all_timings[i]["summarization_ms"] = round(t.elapsed_ms, 2)
            self.summarizer.unload()
            _clear_gpu_cache()

        # Build results
        results = []
        for i in range(n):
            all_timings[i]["total_ms"] = round(sum(all_timings[i].values()), 2)
            result = PipelineResult(
                request_id=request_ids[i],
                original_text=texts[i],
                detected_language=langs[i],
                entities=all_entities[i],
                classification=all_classifications[i],
                summary=all_summaries[i],
                timings=all_timings[i],
            )
            logger.info(
                f"[{request_ids[i]}] Done: {len(all_entities[i])} entities, "
                f"{all_classifications[i].label} ({all_classifications[i].score:.2f}), "
                f"{all_timings[i]['total_ms']:.0f}ms total",
                extra={
                    "component": "pipeline",
                    "latency_ms": all_timings[i]["total_ms"],
                    "items": len(all_entities[i]),
                },
            )
            results.append(result)

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
