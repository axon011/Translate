"""Compare Cross-Lingual NER vs Translate-then-NER approaches.

Demonstrates why direct cross-lingual NER (XLM-RoBERTa on German) is superior
to translating German->English first and then running NER.

Key findings:
- Translation corrupts entity boundaries (compound words, word order)
- Cross-lingual NER preserves original character offsets
- Direct NER has lower latency (1 model call vs 2)

Usage:
    python -m scripts.compare_ner_approaches
    python -m scripts.compare_ner_approaches --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data.dataset import load_ner_eval
from src.evaluation.metrics import compute_ner_metrics
from src.models.ner import NERExtractor
from src.models.translator import Translator
from src.utils.logging import get_logger

logger = get_logger("compare_ner")


def align_entities_to_tokens(
    tokens: list[str],
    entities: list,
    text: str,
) -> list[str]:
    """Map NER entity spans back to token-level BIO tags.

    Uses overlap-based alignment: a token is tagged if its character span
    overlaps with any entity span by more than 50% of the token length.

    Args:
        tokens: Original tokens.
        entities: Entity objects with .start, .end, .label attributes.
        text: The full text that was passed to NER.

    Returns:
        List of BIO tag strings, one per token.
    """
    pred_tags = ["O"] * len(tokens)

    # Build character offset map for each token in the reconstructed text
    token_spans = []
    pos = 0
    for token in tokens:
        # Find the token in the text starting from current position
        idx = text.find(token, pos)
        if idx == -1:
            # Fallback: assume space-separated
            token_spans.append((pos, pos + len(token)))
            pos = pos + len(token) + 1
        else:
            token_spans.append((idx, idx + len(token)))
            pos = idx + len(token)

    for entity in entities:
        first_token = True
        for i, (tok_start, tok_end) in enumerate(token_spans):
            tok_len = tok_end - tok_start
            if tok_len == 0:
                continue

            # Compute overlap
            overlap_start = max(tok_start, entity.start)
            overlap_end = min(tok_end, entity.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > tok_len * 0.5:
                prefix = "B" if first_token else "I"
                pred_tags[i] = f"{prefix}-{entity.label}"
                first_token = False

    return pred_tags


def evaluate_crosslingual_ner(
    ner: NERExtractor,
    test_data: list[dict],
) -> tuple[dict, float]:
    """Evaluate cross-lingual NER (direct German NER).

    Returns:
        Tuple of (metrics_dict, total_time_seconds).
    """
    true_labels = []
    pred_labels = []

    t0 = time.perf_counter()
    for sample in test_data:
        tokens = sample["tokens"]
        true_tags = sample["ner_tags"]
        text = " ".join(tokens)

        entities = ner.extract(text)
        pred_tags = align_entities_to_tokens(tokens, entities, text)

        true_labels.append(true_tags)
        pred_labels.append(pred_tags)

    total_time = time.perf_counter() - t0

    metrics = compute_ner_metrics(true_labels, pred_labels)
    return {
        "overall_f1": metrics.overall_f1,
        "overall_precision": metrics.overall_precision,
        "overall_recall": metrics.overall_recall,
        "per_entity": metrics.per_entity,
        "num_samples": metrics.num_samples,
    }, total_time


def evaluate_translate_then_ner(
    translator: Translator,
    ner: NERExtractor,
    test_data: list[dict],
) -> tuple[dict, float, list[dict]]:
    """Evaluate translate-then-NER approach.

    Translates German text to English, runs NER on English, then
    tries to map entities back to original tokens.

    Returns:
        Tuple of (metrics_dict, total_time_seconds, qualitative_examples).
    """
    true_labels = []
    pred_labels = []
    qualitative = []

    t0 = time.perf_counter()
    for i, sample in enumerate(test_data):
        tokens = sample["tokens"]
        true_tags = sample["ner_tags"]
        german_text = " ".join(tokens)

        # Step 1: Translate DE -> EN
        english_text = translator.translate(german_text)

        # Step 2: NER on English text
        entities = ner.extract(english_text)

        # Step 3: Map back to original German tokens
        # This is inherently lossy — translation changes word order and boundaries
        pred_tags = ["O"] * len(tokens)

        # Best-effort: for each English entity, find matching German tokens
        for entity in entities:
            entity_text_lower = entity.text.lower()
            # Try exact match in German tokens
            for j, token in enumerate(tokens):
                token_matches = (
                    token.lower() in entity_text_lower or entity_text_lower in token.lower()
                )
                if token_matches and pred_tags[j] == "O":
                    # Check if previous token was already tagged with same entity
                    if (
                        j > 0
                        and pred_tags[j - 1] != "O"
                        and pred_tags[j - 1].endswith(entity.label)
                    ):
                        pred_tags[j] = f"I-{entity.label}"
                    else:
                        pred_tags[j] = f"B-{entity.label}"

        true_labels.append(true_tags)
        pred_labels.append(pred_tags)

        # Save qualitative examples (first 10 with entities)
        if i < 20 and any(t != "O" for t in true_tags):
            qualitative.append(
                {
                    "german": german_text,
                    "english": english_text,
                    "true_entities": [
                        tokens[j] for j, t in enumerate(true_tags) if t.startswith("B-")
                    ],
                    "pred_entities_en": [e.text for e in entities],
                    "true_tags": true_tags,
                    "pred_tags": pred_tags,
                }
            )

    total_time = time.perf_counter() - t0

    metrics = compute_ner_metrics(true_labels, pred_labels)
    return (
        {
            "overall_f1": metrics.overall_f1,
            "overall_precision": metrics.overall_precision,
            "overall_recall": metrics.overall_recall,
            "per_entity": metrics.per_entity,
            "num_samples": metrics.num_samples,
        },
        total_time,
        qualitative,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare cross-lingual NER vs translate-then-NER")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max test samples (default: 200)",
    )
    parser.add_argument(
        "--output",
        default="results/ner_comparison.json",
        help="Output path (default: results/ner_comparison.json)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  NER Approach Comparison: Cross-Lingual vs Translate-then-NER")
    print("=" * 70)

    # Load test data
    print(f"\nLoading WikiANN German test data (max {args.max_samples} samples)...")
    _, test_data = load_ner_eval(max_samples=args.max_samples)
    print(f"Loaded {len(test_data)} test samples")

    # --- Approach 1: Cross-Lingual NER (direct German) ---
    print("\n" + "-" * 70)
    print("Approach 1: Cross-Lingual NER (XLM-RoBERTa on German)")
    print("-" * 70)

    ner = NERExtractor()
    ner.load()

    crosslingual_metrics, crosslingual_time = evaluate_crosslingual_ner(ner, test_data)
    ner.unload()

    print(f"  F1:        {crosslingual_metrics['overall_f1']:.4f}")
    print(f"  Precision: {crosslingual_metrics['overall_precision']:.4f}")
    print(f"  Recall:    {crosslingual_metrics['overall_recall']:.4f}")
    print(f"  Time:      {crosslingual_time:.1f}s")

    # --- Approach 2: Translate-then-NER ---
    print("\n" + "-" * 70)
    print("Approach 2: Translate-then-NER (MarianMT DE->EN + XLM-RoBERTa)")
    print("-" * 70)

    translator = Translator()
    translator.load()
    ner.load()

    translate_metrics, translate_time, qualitative = evaluate_translate_then_ner(
        translator, ner, test_data
    )
    ner.unload()
    translator.unload()

    print(f"  F1:        {translate_metrics['overall_f1']:.4f}")
    print(f"  Precision: {translate_metrics['overall_precision']:.4f}")
    print(f"  Recall:    {translate_metrics['overall_recall']:.4f}")
    print(f"  Time:      {translate_time:.1f}s")

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    f1_diff = crosslingual_metrics["overall_f1"] - translate_metrics["overall_f1"]
    speedup = translate_time / crosslingual_time if crosslingual_time > 0 else 0

    print(f"  {'Metric':<25} {'Cross-Lingual':<18} {'Translate-then-NER':<18}")
    print(f"  {'-' * 25} {'-' * 18} {'-' * 18}")
    print(
        f"  {'F1':<25} {crosslingual_metrics['overall_f1']:<18.4f} "
        f"{translate_metrics['overall_f1']:<18.4f}"
    )
    print(
        f"  {'Precision':<25} {crosslingual_metrics['overall_precision']:<18.4f} "
        f"{translate_metrics['overall_precision']:<18.4f}"
    )
    print(
        f"  {'Recall':<25} {crosslingual_metrics['overall_recall']:<18.4f} "
        f"{translate_metrics['overall_recall']:<18.4f}"
    )
    print(f"  {'Total Time':<25} {crosslingual_time:<18.1f} {translate_time:<18.1f}")
    print(f"  {'Speedup':<25} {speedup:.1f}x faster")
    print(f"  {'F1 Advantage':<25} +{f1_diff:.4f}")

    # Show qualitative examples of entity boundary issues
    if qualitative:
        print("\n" + "-" * 70)
        print("  QUALITATIVE EXAMPLES (Entity Boundary Issues)")
        print("-" * 70)
        for ex in qualitative[:5]:
            print(f"\n  DE: {ex['german'][:80]}...")
            print(f"  EN: {ex['english'][:80]}...")
            print(f"  True entities: {ex['true_entities']}")
            print(f"  Pred entities (EN): {ex['pred_entities_en']}")

    # Per-entity comparison
    print("\n" + "-" * 70)
    print("  PER-ENTITY F1 COMPARISON")
    print("-" * 70)
    all_entities = set(
        list(crosslingual_metrics.get("per_entity", {}).keys())
        + list(translate_metrics.get("per_entity", {}).keys())
    )
    print(f"  {'Entity':<10} {'Cross-Lingual':<18} {'Translate-then-NER':<18}")
    print(f"  {'-' * 10} {'-' * 18} {'-' * 18}")
    for entity in sorted(all_entities):
        cl_f1 = crosslingual_metrics.get("per_entity", {}).get(entity, {}).get("f1", 0)
        tr_f1 = translate_metrics.get("per_entity", {}).get(entity, {}).get("f1", 0)
        print(f"  {entity:<10} {cl_f1:<18.4f} {tr_f1:<18.4f}")

    print("=" * 70)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "crosslingual": {
            "metrics": crosslingual_metrics,
            "time_s": round(crosslingual_time, 2),
        },
        "translate_then_ner": {
            "metrics": translate_metrics,
            "time_s": round(translate_time, 2),
            "qualitative_examples": qualitative[:10],
        },
        "comparison": {
            "f1_advantage": round(f1_diff, 4),
            "speedup_factor": round(speedup, 1),
            "num_samples": len(test_data),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
