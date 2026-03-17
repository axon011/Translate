"""Compare cross-lingual NER vs translate-then-NER on live scraped German articles."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.models.ner import NERExtractor
from src.models.translator import Translator


def main() -> None:
    # Load the scraped results
    results_path = Path("results/scraped_results.json")
    with open(results_path, encoding="utf-8") as f:
        scraped = json.load(f)

    # Use the first article (longest, most entities)
    articles = [a for a in scraped if a["word_count"] > 10]

    ner = NERExtractor()
    translator = Translator()

    comparison = []

    for article in articles:
        title = article["title"]
        german_text = article["pipeline_result"]["original_text"]
        crosslingual_entities = article["pipeline_result"]["entities"]

        print(f"\n{'=' * 90}")
        print(f"  ARTICLE: {title[:80]}")
        print(f"{'=' * 90}")

        # --- Approach 1: Cross-lingual (already done) ---
        print("\n  [APPROACH 1] Cross-Lingual NER (direct on German)")
        print(f"  Entities found: {len(crosslingual_entities)}")
        for e in crosslingual_entities:
            if e["text"].strip():
                print(f"    {e['label']:<5} | {e['text']:<30} | conf: {e['score']:.4f}")

        # --- Approach 2: Translate then NER ---
        print("\n  [APPROACH 2] Translate-then-NER (DE -> EN -> NER)")

        # Translate
        translator.load()
        t0 = time.perf_counter()
        english_text = translator.translate(german_text)
        translate_time = (time.perf_counter() - t0) * 1000
        translator.unload()

        print(f"  Translated text ({len(english_text.split())} words, {translate_time:.0f}ms):")
        print(f"    {english_text[:200]}...")

        # NER on English
        ner.load()
        t0 = time.perf_counter()
        en_entities = ner.extract(english_text)
        ner_time = (time.perf_counter() - t0) * 1000
        ner.unload()

        print(f"\n  Entities found: {len(en_entities)}")
        for e in en_entities:
            if e.text.strip():
                print(f"    {e.label:<5} | {e.text:<30} | conf: {e.score:.4f}")

        # --- Comparison ---
        print(f"\n  {'COMPARISON':=^60}")

        cl_persons = [e for e in crosslingual_entities if e["label"] == "PER" and e["text"].strip()]
        cl_locs = [e for e in crosslingual_entities if e["label"] == "LOC" and e["text"].strip()]
        cl_orgs = [e for e in crosslingual_entities if e["label"] == "ORG" and e["text"].strip()]

        en_persons = [e for e in en_entities if e.label == "PER" and e.text.strip()]
        en_locs = [e for e in en_entities if e.label == "LOC" and e.text.strip()]
        en_orgs = [e for e in en_entities if e.label == "ORG" and e.text.strip()]

        print(f"  {'Metric':<25} {'Cross-Lingual':>15} {'Translate+NER':>15}")
        print(f"  {'-' * 25} {'-' * 15} {'-' * 15}")
        print(f"  {'PER entities':<25} {len(cl_persons):>15} {len(en_persons):>15}")
        print(f"  {'LOC entities':<25} {len(cl_locs):>15} {len(en_locs):>15}")
        print(f"  {'ORG entities':<25} {len(cl_orgs):>15} {len(en_orgs):>15}")
        print(f"  {'Total entities':<25} {len(crosslingual_entities):>15} {len(en_entities):>15}")

        # Show what cross-lingual found that translate missed
        cl_texts = {e["text"].strip().lower() for e in crosslingual_entities if e["text"].strip()}
        en_texts = {e.text.strip().lower() for e in en_entities if e.text.strip()}

        only_crosslingual = cl_texts - en_texts
        only_translate = en_texts - cl_texts

        if only_crosslingual:
            print(f"\n  Only in Cross-Lingual: {', '.join(sorted(only_crosslingual))}")
        if only_translate:
            print(f"  Only in Translate+NER: {', '.join(sorted(only_translate))}")

        comparison.append(
            {
                "title": title,
                "crosslingual_count": len(crosslingual_entities),
                "translate_ner_count": len(en_entities),
                "crosslingual_persons": [e["text"] for e in cl_persons],
                "translate_persons": [e.text for e in en_persons],
                "crosslingual_locs": [e["text"] for e in cl_locs],
                "translate_locs": [e.text for e in en_locs],
                "translate_time_ms": translate_time,
                "en_ner_time_ms": ner_time,
            }
        )

    # Final verdict
    print(f"\n{'=' * 90}")
    print("  VERDICT")
    print(f"{'=' * 90}")
    total_cl = sum(c["crosslingual_count"] for c in comparison)
    total_en = sum(c["translate_ner_count"] for c in comparison)
    print(f"  Cross-Lingual NER found {total_cl} total entities across {len(comparison)} articles")
    print(f"  Translate-then-NER found {total_en} total entities across {len(comparison)} articles")
    if total_cl > total_en:
        print(
            f"  -> Cross-Lingual wins: +{total_cl - total_en} more entities, no translation overhead"
        )
    elif total_en > total_cl:
        print(f"  -> Translate-then-NER found +{total_en - total_cl} more entities")
    else:
        print("  -> Tie in entity count")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
