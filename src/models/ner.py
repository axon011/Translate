"""Named Entity Recognition using XLM-RoBERTa (cross-lingual).

Uses xlm-roberta-large-finetuned-conll03-german for direct German NER
without requiring translation. Supports FP16 for 4GB VRAM constraint.

Entity types: PER (Person), ORG (Organization), LOC (Location), MISC (Miscellaneous)
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from src.utils.config import NERConfig, get_config
from src.utils.logging import TimingContext, get_logger

logger = get_logger("ner")

# Entity types extracted by the model
ENTITY_TYPES = {"PER", "ORG", "LOC", "MISC"}


@dataclass
class Entity:
    """A single named entity extracted from text."""

    text: str
    label: str
    score: float
    start: int
    end: int


class NERExtractor:
    """Cross-lingual NER using XLM-RoBERTa.

    Key design choices:
    - Uses xlm-roberta-large-finetuned-conll03-german for direct German NER
    - FP16 inference to fit within 4GB VRAM (~1.9GB model)
    - aggregation_strategy="simple" merges subword tokens into full entities
    - Explicit load/unload methods for sequential VRAM management
    """

    def __init__(
        self,
        config: NERConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or get_config().ner
        self.device = device or self.config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self._model = None
        self._tokenizer = None
        self._pipe = None
        self._loaded = False

    def load(self) -> None:
        """Load model onto device. Call before inference."""
        if self._loaded:
            return

        logger.info(
            "Loading NER model",
            extra={"component": "ner", "model": self.config.model_id},
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForTokenClassification.from_pretrained(self.config.model_id)

        # FP16 for GPU memory efficiency
        if self.device == "cuda" and self.config.precision == "fp16":
            self._model = self._model.half()

        self._model.to(self.device)
        self._model.eval()

        device_id = 0 if self.device == "cuda" else -1
        self._pipe = pipeline(
            "ner",
            model=self._model,
            tokenizer=self._tokenizer,
            device=device_id,
            aggregation_strategy=self.config.aggregation_strategy,
        )

        self._loaded = True

        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**2
            logger.info(
                f"NER model loaded, VRAM: {vram:.0f} MB",
                extra={"component": "ner", "vram_mb": round(vram, 1)},
            )

    def unload(self) -> None:
        """Unload model from GPU to free VRAM."""
        if not self._loaded:
            return

        del self._pipe
        del self._model
        del self._tokenizer
        self._pipe = None
        self._model = None
        self._tokenizer = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("NER model unloaded", extra={"component": "ner"})

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def extract(self, text: str) -> list[Entity]:
        """Extract named entities from text.

        Works on both German and English text (cross-lingual).

        Args:
            text: Input text (German or English).

        Returns:
            List of Entity objects with label, text, score, and character offsets.
        """
        if not text or not text.strip():
            return []

        if not self._loaded:
            self.load()

        with TimingContext("ner_inference") as t:
            raw_entities = self._pipe(text)

        entities: list[Entity] = []
        for ent in raw_entities:
            label = ent["entity_group"]
            if label not in ENTITY_TYPES:
                continue

            entities.append(
                Entity(
                    text=ent["word"].strip(),
                    label=label,
                    score=round(float(ent["score"]), 4),
                    start=ent["start"],
                    end=ent["end"],
                )
            )

        logger.debug(
            f"Extracted {len(entities)} entities in {t.elapsed_ms:.1f}ms",
            extra={
                "component": "ner",
                "latency_ms": round(t.elapsed_ms, 1),
                "items": len(entities),
            },
        )

        return entities

    def extract_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Extract entities from multiple texts using batched inference.

        Uses the HF pipeline's native batch support for fewer GPU kernel launches.

        Args:
            texts: List of input texts.

        Returns:
            List of entity lists, one per input text.
        """
        if not self._loaded:
            self.load()

        # Filter empty texts, track indices
        valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        results: list[list[Entity]] = [[] for _ in range(len(texts))]

        if not valid:
            return results

        valid_indices, valid_texts = zip(*valid)

        with TimingContext("ner_batch_inference") as t:
            batch_raw = self._pipe(list(valid_texts), batch_size=len(valid_texts))

        for idx, raw_entities in zip(valid_indices, batch_raw):
            entities = []
            for ent in raw_entities:
                label = ent["entity_group"]
                if label not in ENTITY_TYPES:
                    continue
                entities.append(
                    Entity(
                        text=ent["word"].strip(),
                        label=label,
                        score=round(float(ent["score"]), 4),
                        start=ent["start"],
                        end=ent["end"],
                    )
                )
            results[idx] = entities

        logger.debug(
            f"Batch NER: {len(valid_texts)} texts in {t.elapsed_ms:.1f}ms",
            extra={"component": "ner", "latency_ms": round(t.elapsed_ms, 1)},
        )

        return results
