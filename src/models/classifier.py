"""Event classification for news articles.

Fine-tunes distilbert-base-multilingual-cased on 10kGNAD for German news
classification into 4 categories: Political, Economic, Sports, Technology.

Key design choices:
- Multilingual DistilBERT handles German text directly (no translation needed)
- 4 classes instead of original 10kGNAD categories (merged for balance)
- Gradient accumulation for small batch sizes on 4GB VRAM
- FP16 training/inference
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.utils.config import ClassifierConfig, get_config
from src.utils.logging import TimingContext, get_logger

logger = get_logger("classifier")

# 4-class mapping (from 10kGNAD categories)
LABEL2ID = {
    "Political": 0,
    "Economic": 1,
    "Sports": 2,
    "Technology": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


@dataclass
class ClassificationResult:
    """Result of event classification."""

    label: str
    score: float
    all_scores: dict[str, float]


class NewsDataset(Dataset):
    """PyTorch dataset for news classification fine-tuning."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class EventClassifier:
    """Multilingual news event classifier using DistilBERT.

    Supports:
    - Inference on pre-trained or fine-tuned model
    - Fine-tuning on 10kGNAD with gradient accumulation
    - Explicit load/unload for VRAM management
    """

    def __init__(
        self,
        config: ClassifierConfig | None = None,
        model_path: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize classifier.

        Args:
            config: Classifier configuration.
            model_path: Path to a fine-tuned model. If None, loads base model.
            device: Device override ("cuda" or "cpu").
        """
        self.config = config or get_config().classifier
        self.device = device or self.config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model_path = model_path

        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load model onto device."""
        if self._loaded:
            return

        load_from = self.model_path or self.config.model_id
        logger.info(
            "Loading classifier model",
            extra={"component": "classifier", "model": load_from},
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id if self.model_path is None else load_from
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            load_from,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self._model.to(self.device)
        self._model.eval()

        self._loaded = True

        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**2
            logger.info(
                f"Classifier loaded, VRAM: {vram:.0f} MB",
                extra={"component": "classifier", "vram_mb": round(vram, 1)},
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

        logger.info("Classifier unloaded", extra={"component": "classifier"})

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @torch.inference_mode()
    def classify(self, text: str) -> ClassificationResult:
        """Classify a single text into an event category.

        Args:
            text: News text (German or English).

        Returns:
            ClassificationResult with label, confidence, and all scores.
        """
        if not self._loaded:
            self.load()

        with TimingContext("classify_inference") as t:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = probs.argmax(dim=-1).item()
            score = probs[0, pred_id].item()

            all_scores = {ID2LABEL[i]: round(probs[0, i].item(), 4) for i in range(NUM_LABELS)}

        logger.debug(
            f"Classified as {ID2LABEL[pred_id]} ({score:.3f}) in {t.elapsed_ms:.1f}ms",
            extra={"component": "classifier", "latency_ms": round(t.elapsed_ms, 1)},
        )

        # Low confidence means text doesn't fit any trained category
        label = ID2LABEL[pred_id] if score >= 0.75 else "Other"

        return ClassificationResult(
            label=label,
            score=round(score, 4),
            all_scores=all_scores,
        )

    @torch.inference_mode()
    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify multiple texts."""
        return [self.classify(t) for t in texts]

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
        epochs: int | None = None,
        batch_size: int = 4,
        lr: float | None = None,
        grad_accum_steps: int = 4,
        save_path: str = "models/event_classifier",
    ) -> dict:
        """Fine-tune the classifier on 10kGNAD.

        Uses small batch size + gradient accumulation for 4GB VRAM.

        Args:
            train_texts: Training texts.
            train_labels: Training label indices (0-3).
            val_texts: Optional validation texts.
            val_labels: Optional validation labels.
            epochs: Number of training epochs.
            batch_size: Micro batch size (keep 2-4 for RTX 3050).
            lr: Learning rate.
            grad_accum_steps: Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).
            save_path: Where to save the fine-tuned model.

        Returns:
            Dict with training metrics (train_loss, val_accuracy per epoch).
        """
        if not self._loaded:
            self.load()

        epochs = epochs or self.config.train.num_epochs
        lr = lr or self.config.train.learning_rate

        self._model.train()

        train_dataset = NewsDataset(
            train_texts, train_labels, self._tokenizer, self.config.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=lr,
            weight_decay=self.config.train.weight_decay,
        )
        total_steps = (len(train_loader) // grad_accum_steps) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(self.config.train.warmup_steps, total_steps // 10),
            num_training_steps=total_steps,
        )

        history = {"train_loss": [], "val_accuracy": []}

        logger.info(
            f"Starting training: {epochs} epochs, batch={batch_size}, "
            f"accum={grad_accum_steps}, lr={lr}",
            extra={"component": "classifier"},
        )

        for epoch in range(epochs):
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self._model(**batch)
                loss = outputs.loss / grad_accum_steps
                loss.backward()
                total_loss += loss.item() * grad_accum_steps

                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)

            # Validation
            val_acc = None
            if val_texts and val_labels:
                val_acc = self._evaluate(val_texts, val_labels)
                history["val_accuracy"].append(val_acc)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}"
                + (f" | Val Acc: {val_acc:.4f}" if val_acc is not None else ""),
                extra={"component": "classifier"},
            )

        # Save fine-tuned model
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(save_dir)
        self._tokenizer.save_pretrained(save_dir)

        self._model.eval()
        logger.info(
            f"Model saved to {save_path}",
            extra={"component": "classifier"},
        )

        return history

    @torch.inference_mode()
    def _evaluate(self, texts: list[str], labels: list[int]) -> float:
        """Calculate accuracy on a validation set."""
        self._model.eval()
        correct = 0
        for text, label in zip(texts, labels, strict=True):
            result = self.classify(text)
            if LABEL2ID[result.label] == label:
                correct += 1
        self._model.train()
        return correct / len(labels)
