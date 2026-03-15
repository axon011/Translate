"""Dataset loaders for training and evaluation.

Handles:
- 10kGNAD (German news classification) with category remapping
- GermEval 2014 (NER evaluation)
- FLEURS German (ASR evaluation)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger("dataset")

# 10kGNAD category mapping to 4 target classes
CATEGORY_MAPPING = {
    "Inland": "Political",
    "International": "Political",
    "Wirtschaft": "Economic",
    "Etat": "Economic",
    "Sport": "Sports",
    "Web": "Technology",
    "Wissenschaft": "Technology",
}

# Categories to drop (poorly mapped or unbalanced)
DROP_CATEGORIES = {"Kultur", "Panorama"}

LABEL2ID = {
    "Political": 0,
    "Economic": 1,
    "Sports": 2,
    "Technology": 3,
}


@dataclass
class ClassificationSplit:
    """A train/val/test split for classification."""

    texts: list[str]
    labels: list[int]
    label_names: list[str]

    def __len__(self) -> int:
        return len(self.texts)


def load_10kgnad(
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[ClassificationSplit, ClassificationSplit, ClassificationSplit]:
    """Load 10kGNAD dataset with category remapping to 4 classes.

    Remapping:
        Inland + International -> Political
        Wirtschaft + Etat -> Economic
        Sport -> Sports
        Web + Wissenschaft -> Technology
        Kultur, Panorama -> dropped

    Args:
        val_ratio: Fraction of training data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, val, test) ClassificationSplit objects.
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    config = get_config()
    mapping = config.datasets.classification.category_mapping

    logger.info("Loading 10kGNAD dataset", extra={"component": "dataset"})

    # Load from HuggingFace
    dataset = load_dataset(config.datasets.classification.hf_id)

    def process_split(split_data) -> tuple[list[str], list[int], list[str]]:
        """Process a dataset split, applying category mapping."""
        texts = []
        labels = []
        label_names = []

        # 10kGNAD has 'text' and 'label' columns
        # label is an integer index into the category names
        label_feature = split_data.features["label"]

        for item in split_data:
            original_label = label_feature.int2str(item["label"])

            if original_label in DROP_CATEGORIES:
                continue

            target_label = mapping.get(original_label)
            if target_label is None:
                continue

            texts.append(item["text"])
            labels.append(LABEL2ID[target_label])
            label_names.append(target_label)

        return texts, labels, label_names

    # Process train split
    train_texts, train_labels, train_names = process_split(dataset["train"])
    test_texts, test_labels, test_names = process_split(dataset["test"])

    logger.info(
        f"Loaded {len(train_texts)} train, {len(test_texts)} test samples "
        f"(after filtering and remapping)",
        extra={"component": "dataset"},
    )

    # Create validation split from training data
    if val_ratio > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=val_ratio,
            random_state=seed,
            stratify=train_labels,
        )
        val_names = [list(LABEL2ID.keys())[idx] for idx in val_labels]
        train_names = [list(LABEL2ID.keys())[idx] for idx in train_labels]

        logger.info(
            f"Split: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test",
            extra={"component": "dataset"},
        )
    else:
        val_texts, val_labels, val_names = [], [], []

    # Log class distribution
    from collections import Counter

    train_dist = Counter(train_names)
    logger.info(
        f"Train distribution: {dict(train_dist)}",
        extra={"component": "dataset"},
    )

    return (
        ClassificationSplit(train_texts, train_labels, train_names),
        ClassificationSplit(val_texts, val_labels, val_names),
        ClassificationSplit(test_texts, test_labels, test_names),
    )


def load_ner_eval(max_samples: int | None = None) -> tuple[list[dict], list[dict]]:
    """Load NER evaluation dataset (WikiANN German).

    Uses WikiANN German split which has PER, ORG, LOC tags matching
    our XLM-RoBERTa model's output format.

    Args:
        max_samples: Maximum samples per split (None = all).

    Returns:
        Tuple of (dev, test) where each is a list of dicts with
        'tokens' and 'ner_tags' keys. Tags are string BIO labels.
    """
    from datasets import load_dataset

    config = get_config()
    ner_config = config.datasets.ner_eval

    logger.info(
        f"Loading {ner_config.name} NER dataset",
        extra={"component": "dataset"},
    )

    dataset = load_dataset(ner_config.hf_id, ner_config.language)

    # WikiANN tag names: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    tag_names = dataset["test"].features["ner_tags"].feature.names

    def process_ner_split(split_data, limit: int | None = None) -> list[dict]:
        samples = []
        for i, item in enumerate(split_data):
            if limit and i >= limit:
                break
            # Convert integer tags to string BIO labels
            str_tags = [tag_names[t] for t in item["ner_tags"]]
            samples.append(
                {
                    "tokens": item["tokens"],
                    "ner_tags": str_tags,
                }
            )
        return samples

    dev = process_ner_split(dataset.get("validation", []), limit=max_samples)
    test = process_ner_split(dataset.get("test", []), limit=max_samples)

    logger.info(
        f"Loaded {ner_config.name}: {len(dev)} dev, {len(test)} test sentences",
        extra={"component": "dataset"},
    )

    return dev, test


def load_fleurs_german(max_samples: int | None = None) -> list[dict]:
    """Load FLEURS German test set for ASR evaluation.

    Args:
        max_samples: Maximum number of samples to load (None = all).

    Returns:
        List of dicts with 'audio' and 'transcription' keys.
    """
    from datasets import load_dataset

    config = get_config()
    logger.info("Loading FLEURS German dataset", extra={"component": "dataset"})

    dataset = load_dataset(
        config.datasets.asr_eval.hf_id,
        config.datasets.asr_eval.language,
        split=config.datasets.asr_eval.split,
    )

    samples = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        samples.append(
            {
                "audio": item["audio"],
                "transcription": item["transcription"],
                "id": item.get("id", i),
            }
        )

    logger.info(
        f"Loaded {len(samples)} FLEURS German test samples",
        extra={"component": "dataset"},
    )

    return samples
