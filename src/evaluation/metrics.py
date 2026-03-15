"""Evaluation metrics for all pipeline components.

Provides:
- NER: Span-level F1 (via seqeval)
- Classification: Accuracy, Macro F1, Balanced Accuracy, ROC-AUC, MCC
- ASR: Word Error Rate (WER), Character Error Rate (CER)
- Summarization: ROUGE-1, ROUGE-2, ROUGE-L
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.logging import get_logger

logger = get_logger("metrics")


@dataclass
class NERMetrics:
    """NER evaluation results."""

    overall_f1: float
    overall_precision: float
    overall_recall: float
    per_entity: dict[str, dict[str, float]]
    num_samples: int


@dataclass
class ClassificationMetrics:
    """Classification evaluation results."""

    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    roc_auc: float | None
    mcc: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    num_samples: int


@dataclass
class ASRMetrics:
    """ASR evaluation results."""

    wer: float
    cer: float
    num_samples: int
    total_duration_s: float


@dataclass
class SummarizationMetrics:
    """Summarization evaluation results."""

    rouge1: float
    rouge2: float
    rougeL: float
    num_samples: int


def compute_ner_metrics(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> NERMetrics:
    """Compute span-level NER metrics using seqeval.

    Args:
        true_labels: List of true label sequences (e.g., [["O", "B-PER", "I-PER", ...], ...]).
        pred_labels: List of predicted label sequences.

    Returns:
        NERMetrics with F1, precision, recall per entity type.
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    from seqeval.scheme import IOB2

    overall_f1 = f1_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
    overall_precision = precision_score(true_labels, pred_labels, mode="strict", scheme=IOB2)
    overall_recall = recall_score(true_labels, pred_labels, mode="strict", scheme=IOB2)

    # Per-entity breakdown
    report = classification_report(
        true_labels, pred_labels, mode="strict", scheme=IOB2, output_dict=True
    )

    per_entity = {}
    for entity_type, metrics in report.items():
        if isinstance(metrics, dict) and entity_type not in (
            "micro avg",
            "macro avg",
            "weighted avg",
        ):
            per_entity[entity_type] = {
                "f1": round(metrics["f1-score"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "support": int(metrics["support"]),
            }

    result = NERMetrics(
        overall_f1=round(overall_f1, 4),
        overall_precision=round(overall_precision, 4),
        overall_recall=round(overall_recall, 4),
        per_entity=per_entity,
        num_samples=len(true_labels),
    )

    logger.info(
        f"NER F1: {result.overall_f1:.4f} (P={result.overall_precision:.4f}, "
        f"R={result.overall_recall:.4f}) on {result.num_samples} samples",
        extra={"component": "metrics", "metric": "ner_f1"},
    )

    return result


def compute_classification_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    label_names: list[str] | None = None,
    pred_probs: list[list[float]] | None = None,
) -> ClassificationMetrics:
    """Compute classification accuracy, F1, balanced accuracy, ROC-AUC, and MCC.

    Args:
        true_labels: True label indices.
        pred_labels: Predicted label indices.
        label_names: Optional list of label names.
        pred_probs: Optional prediction probabilities per class (for ROC-AUC).

    Returns:
        ClassificationMetrics with accuracy, macro F1, balanced accuracy, ROC-AUC, MCC.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    bal_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    mcc = matthews_corrcoef(true_labels, pred_labels)

    # ROC-AUC requires prediction probabilities
    roc_auc = None
    if pred_probs is not None:
        try:
            roc_auc = round(
                roc_auc_score(true_labels, pred_probs, multi_class="ovr", average="macro"),
                4,
            )
        except ValueError:
            logger.warning(
                "Could not compute ROC-AUC (likely missing classes in split)",
                extra={"component": "metrics"},
            )

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=label_names,
        output_dict=True,
    )

    per_class = {}
    for label_name, metrics in report.items():
        if isinstance(metrics, dict) and label_name not in (
            "accuracy",
            "macro avg",
            "weighted avg",
        ):
            per_class[label_name] = {
                "f1": round(metrics["f1-score"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "support": int(metrics["support"]),
            }

    cm = confusion_matrix(true_labels, pred_labels).tolist()

    result = ClassificationMetrics(
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        balanced_accuracy=round(bal_accuracy, 4),
        roc_auc=roc_auc,
        mcc=round(mcc, 4),
        per_class=per_class,
        confusion_matrix=cm,
        num_samples=len(true_labels),
    )

    logger.info(
        f"Classification Accuracy: {result.accuracy:.4f}, "
        f"Macro F1: {result.macro_f1:.4f}, "
        f"Balanced Acc: {result.balanced_accuracy:.4f}, "
        f"MCC: {result.mcc:.4f} on {result.num_samples} samples",
        extra={"component": "metrics", "metric": "classification_f1"},
    )

    return result


def compute_wer(
    references: list[str],
    hypotheses: list[str],
) -> ASRMetrics:
    """Compute Word Error Rate and Character Error Rate for ASR evaluation.

    Args:
        references: Ground truth transcriptions.
        hypotheses: Model predicted transcriptions.

    Returns:
        ASRMetrics with WER and CER.
    """
    try:
        from jiwer import cer as compute_jiwer_cer
        from jiwer import wer as compute_jiwer_wer

        wer_score = compute_jiwer_wer(references, hypotheses)
        cer_score = compute_jiwer_cer(references, hypotheses)
    except ImportError:
        # Manual computation
        wer_score = _manual_wer(references, hypotheses)
        cer_score = _manual_cer(references, hypotheses)

    result = ASRMetrics(
        wer=round(wer_score, 4),
        cer=round(cer_score, 4),
        num_samples=len(references),
        total_duration_s=0.0,  # Set externally
    )

    logger.info(
        f"ASR WER: {result.wer:.4f}, CER: {result.cer:.4f} on {result.num_samples} samples",
        extra={"component": "metrics", "metric": "wer"},
    )

    return result


def compute_rouge(
    references: list[str],
    hypotheses: list[str],
) -> SummarizationMetrics:
    """Compute ROUGE scores for summarization evaluation.

    Args:
        references: Reference summaries.
        hypotheses: Generated summaries.

    Returns:
        SummarizationMetrics with ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, hyp in zip(references, hypotheses, strict=True):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    result = SummarizationMetrics(
        rouge1=round(sum(rouge1_scores) / len(rouge1_scores), 4),
        rouge2=round(sum(rouge2_scores) / len(rouge2_scores), 4),
        rougeL=round(sum(rougeL_scores) / len(rougeL_scores), 4),
        num_samples=len(references),
    )

    logger.info(
        f"ROUGE-1: {result.rouge1:.4f}, ROUGE-2: {result.rouge2:.4f}, "
        f"ROUGE-L: {result.rougeL:.4f} on {result.num_samples} samples",
        extra={"component": "metrics", "metric": "rouge"},
    )

    return result


def _manual_wer(references: list[str], hypotheses: list[str]) -> float:
    """Compute WER manually using edit distance."""
    total_words = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses, strict=True):
        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()
        total_words += len(ref_words)

        # Levenshtein distance at word level
        d = _word_edit_distance(ref_words, hyp_words)
        total_errors += d

    return total_errors / max(total_words, 1)


def _manual_cer(references: list[str], hypotheses: list[str]) -> float:
    """Compute CER manually using character-level edit distance."""
    total_chars = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses, strict=True):
        ref_chars = list(ref.lower())
        hyp_chars = list(hyp.lower())
        total_chars += len(ref_chars)
        d = _word_edit_distance(ref_chars, hyp_chars)
        total_errors += d

    return total_errors / max(total_chars, 1)


def _word_edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Compute word-level edit distance."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]
