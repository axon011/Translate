"""Run all evaluations: NER (GermEval), Classification (10kGNAD test), ASR (FLEURS).

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --component ner
    python -m scripts.evaluate --component asr --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = get_logger("evaluate")


def evaluate_ner(max_samples: int = 1000):
    """Evaluate NER on WikiANN German."""
    from src.data.dataset import load_ner_eval
    from src.evaluation.metrics import compute_ner_metrics
    from src.models.ner import NERExtractor

    logger.info("Evaluating NER on WikiANN German...", extra={"component": "evaluate"})

    _, test_data = load_ner_eval(max_samples=max_samples)
    ner = NERExtractor()
    ner.load()

    true_labels = []
    pred_labels = []

    for sample in test_data:
        tokens = sample["tokens"]
        true_tags = sample["ner_tags"]  # Already string BIO tags

        # Run NER on the full sentence
        text = " ".join(tokens)
        entities = ner.extract(text)

        # Map predictions back to token-level BIO tags using overlap-based alignment.
        # Previous approach used strict containment (token fully inside entity span),
        # which missed tokens at entity boundaries due to subword tokenization offsets.
        # New approach: tag a token if >50% of its characters overlap with an entity.
        pred_tags = ["O"] * len(tokens)

        # Build token character spans
        token_spans = []
        pos = 0
        for token in tokens:
            idx = text.find(token, pos)
            if idx == -1:
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

                # Compute character overlap between token and entity
                overlap_start = max(tok_start, entity.start)
                overlap_end = min(tok_end, entity.end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > tok_len * 0.5:
                    prefix = "B" if first_token else "I"
                    pred_tags[i] = f"{prefix}-{entity.label}"
                    first_token = False

        true_labels.append(true_tags)
        pred_labels.append(pred_tags)

    metrics = compute_ner_metrics(true_labels, pred_labels)
    ner.unload()

    return {
        "overall_f1": metrics.overall_f1,
        "overall_precision": metrics.overall_precision,
        "overall_recall": metrics.overall_recall,
        "per_entity": metrics.per_entity,
        "num_samples": metrics.num_samples,
    }


def evaluate_classification(model_path: str = "models/event_classifier"):
    """Evaluate classifier on 10kGNAD test set."""
    from src.data.dataset import LABEL2ID, load_10kgnad
    from src.evaluation.metrics import compute_classification_metrics
    from src.models.classifier import EventClassifier

    logger.info("Evaluating Classifier on 10kGNAD test...", extra={"component": "evaluate"})

    _, _, test_split = load_10kgnad(val_ratio=0)

    classifier = EventClassifier(model_path=model_path)
    classifier.load()

    pred_labels = []
    pred_probs = []
    label_names = list(LABEL2ID.keys())

    for text in test_split.texts:
        result = classifier.classify(text)
        pred_labels.append(LABEL2ID[result.label])
        # Collect prediction probabilities in label index order for ROC-AUC
        pred_probs.append([result.all_scores[name] for name in label_names])

    metrics = compute_classification_metrics(
        true_labels=test_split.labels,
        pred_labels=pred_labels,
        label_names=label_names,
        pred_probs=pred_probs,
    )

    classifier.unload()

    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "roc_auc": metrics.roc_auc,
        "mcc": metrics.mcc,
        "per_class": metrics.per_class,
        "num_samples": metrics.num_samples,
    }


def evaluate_asr(max_samples: int = 50):
    """Evaluate ASR using local TTS-generated German audio.

    Uses gTTS to generate German speech from reference texts, then
    transcribes with our Whisper model and computes WER. This avoids
    dependency on FLEURS (broken with latest datasets library).

    Falls back to FLEURS if local audio is not available.
    """
    import glob
    import json as json_module

    from src.evaluation.metrics import compute_wer
    from src.models.asr import ASRModel

    audio_dir = Path("data/eval_audio")
    refs_file = audio_dir / "references.json"

    # Check if local evaluation audio exists
    if refs_file.exists():
        logger.info(
            "Evaluating ASR on local TTS-generated German audio...",
            extra={"component": "evaluate"},
        )

        with open(refs_file, encoding="utf-8") as f:
            ref_map = json_module.load(f)

        asr = ASRModel()
        asr.load()

        references = []
        hypotheses = []

        audio_files = sorted(glob.glob(str(audio_dir / "sample_*.mp3")))
        for audio_path in audio_files[:max_samples]:
            sample_id = Path(audio_path).stem
            ref_text = ref_map[sample_id]
            result = asr.transcribe(audio_path)
            references.append(ref_text)
            hypotheses.append(result.text)

        metrics = compute_wer(references, hypotheses)
        asr.unload()

        return {
            "wer": metrics.wer,
            "cer": metrics.cer,
            "num_samples": metrics.num_samples,
            "source": "tts_generated",
        }
    else:
        # Fallback: try FLEURS
        import tempfile

        import soundfile as sf

        from src.data.dataset import load_fleurs_german

        logger.info(
            f"Evaluating ASR on FLEURS German ({max_samples} samples)...",
            extra={"component": "evaluate"},
        )

        samples = load_fleurs_german(max_samples=max_samples)
        asr = ASRModel()
        asr.load()

        references = []
        hypotheses = []

        for sample in samples:
            audio = sample["audio"]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio["array"], audio["sampling_rate"])
                result = asr.transcribe(tmp.name)
                Path(tmp.name).unlink()

            references.append(sample["transcription"])
            hypotheses.append(result.text)

        metrics = compute_wer(references, hypotheses)
        asr.unload()

        return {
            "wer": metrics.wer,
            "cer": metrics.cer,
            "num_samples": metrics.num_samples,
            "source": "fleurs_de",
        }


def evaluate_translation(max_samples: int = 50):
    """Evaluate translation quality using BLEU score.

    Uses a small set of German-English parallel sentences to measure
    translation quality with sacrebleu.
    """
    from src.models.translator import Translator

    logger.info("Evaluating Translation (BLEU)...", extra={"component": "evaluate"})

    # Parallel German-English test sentences (curated for news domain)
    parallel_corpus = [
        {
            "de": "Die Bundeskanzlerin traf sich mit dem französischen Präsidenten in Berlin.",
            "en": "The Federal Chancellor met with the French President in Berlin.",
        },
        {
            "de": "Der Deutsche Aktienindex stieg um drei Prozent auf ein neues Rekordhoch.",
            "en": "The German stock index rose by three percent to a new record high.",
        },
        {
            "de": "Bayern München gewann das Spiel gegen Borussia Dortmund mit zwei zu eins.",
            "en": "Bayern Munich won the game against Borussia Dortmund two to one.",
        },
        {
            "de": "Forscher der Technischen Universität entwickelten einen neuen Algorithmus für künstliche Intelligenz.",
            "en": "Researchers at the Technical University developed a new algorithm for artificial intelligence.",
        },
        {
            "de": "Die Europäische Zentralbank hat den Leitzins unverändert gelassen.",
            "en": "The European Central Bank has left the key interest rate unchanged.",
        },
        {
            "de": "Der Außenminister forderte eine diplomatische Lösung des Konflikts.",
            "en": "The Foreign Minister called for a diplomatic solution to the conflict.",
        },
        {
            "de": "Die Arbeitslosenquote sank im vergangenen Monat auf den niedrigsten Stand seit zehn Jahren.",
            "en": "The unemployment rate fell last month to the lowest level in ten years.",
        },
        {
            "de": "Das neue Gesetz zur Datenschutzreform tritt nächsten Monat in Kraft.",
            "en": "The new data protection reform law takes effect next month.",
        },
        {
            "de": "Wissenschaftler warnen vor den Folgen des Klimawandels für die Landwirtschaft.",
            "en": "Scientists warn of the consequences of climate change for agriculture.",
        },
        {
            "de": "Die Regierung plant massive Investitionen in den Ausbau der digitalen Infrastruktur.",
            "en": "The government plans massive investments in the expansion of digital infrastructure.",
        },
    ]

    translator = Translator()
    translator.load()

    hypotheses = []
    references = []

    for pair in parallel_corpus[:max_samples]:
        translated = translator.translate(pair["de"])
        hypotheses.append(translated)
        references.append(pair["en"])

    translator.unload()

    # Compute BLEU and ChrF scores using sacrebleu
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])

    # Compute BERTScore
    try:
        from bert_score import score as bert_score_fn

        bert_P, bert_R, bert_F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
        bert_scores = {
            "precision": round(bert_P.mean().item(), 4),
            "recall": round(bert_R.mean().item(), 4),
            "f1": round(bert_F1.mean().item(), 4),
        }
    except ImportError:
        logger.warning(
            "bert-score not installed, skipping BERTScore",
            extra={"component": "evaluate"},
        )
        bert_scores = None

    # Also compute per-sentence scores for error analysis
    per_sentence = []
    for i, (hyp, ref) in enumerate(zip(hypotheses, references, strict=True)):
        sent_bleu = sacrebleu.sentence_bleu(hyp, [ref])
        per_sentence.append(
            {
                "source": parallel_corpus[i]["de"],
                "reference": ref,
                "hypothesis": hyp,
                "bleu": round(sent_bleu.score, 2),
            }
        )

    logger.info(
        f"Translation BLEU: {bleu.score:.2f}, ChrF: {chrf.score:.2f} on {len(hypotheses)} samples",
        extra={"component": "evaluate"},
    )

    result = {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "bleu_signature": str(bleu),
        "num_samples": len(hypotheses),
        "per_sentence": per_sentence,
    }
    if bert_scores is not None:
        result["bert_score"] = bert_scores

    return result


def evaluate_summarization(max_samples: int = 50):
    """Evaluate summarization quality using ROUGE scores.

    Translates German texts to English, summarizes them, and compares
    with reference summaries using ROUGE-1, ROUGE-2, and ROUGE-L.
    """
    from src.evaluation.metrics import compute_rouge
    from src.models.summarizer import Summarizer
    from src.models.translator import Translator

    logger.info("Evaluating Summarization (ROUGE)...", extra={"component": "evaluate"})

    # Curated German news texts with reference English summaries
    eval_data = [
        {
            "de": (
                "Die Bundeskanzlerin hat sich bei einem Treffen in Berlin mit dem "
                "französischen Präsidenten auf eine engere Zusammenarbeit in der "
                "Verteidigungspolitik geeinigt. Beide Seiten betonten die Bedeutung "
                "einer gemeinsamen europäischen Sicherheitsstrategie. Die Vereinbarung "
                "umfasst gemeinsame Rüstungsprojekte und den Austausch von Militärpersonal."
            ),
            "reference": (
                "The German Chancellor and French President agreed on closer defense "
                "cooperation in Berlin, including joint armament projects and military "
                "personnel exchanges as part of a common European security strategy."
            ),
        },
        {
            "de": (
                "Der Deutsche Aktienindex erreichte am Freitag ein neues Allzeithoch "
                "und schloss bei über 18.000 Punkten. Analysten führen den Anstieg auf "
                "positive Quartalszahlen großer Unternehmen und die Hoffnung auf eine "
                "baldige Zinssenkung der Europäischen Zentralbank zurück. Besonders "
                "Technologiewerte verzeichneten starke Gewinne."
            ),
            "reference": (
                "The German stock index hit a new all-time high above 18,000 points, "
                "driven by strong quarterly earnings and expectations of ECB rate cuts, "
                "with technology stocks leading the gains."
            ),
        },
        {
            "de": (
                "Bayern München sicherte sich mit einem überzeugenden Sieg gegen Borussia "
                "Dortmund die Tabellenführung in der Bundesliga. Der Stürmer erzielte "
                "zwei Tore in der ersten Halbzeit, während die Abwehr keine Gegentreffer "
                "zuließ. Der Trainer lobte die Mannschaft für ihre Disziplin und taktische "
                "Umsetzung."
            ),
            "reference": (
                "Bayern Munich took the Bundesliga lead with a dominant win over Borussia "
                "Dortmund, with the striker scoring twice in the first half and the defense "
                "keeping a clean sheet."
            ),
        },
        {
            "de": (
                "Forscher der Technischen Universität München haben einen Durchbruch "
                "in der Quantencomputertechnologie erzielt. Das Team entwickelte einen "
                "neuen Algorithmus, der Berechnungen bis zu hundertmal schneller "
                "durchführen kann als bisherige Methoden. Die Ergebnisse wurden in der "
                "Fachzeitschrift Nature veröffentlicht."
            ),
            "reference": (
                "Researchers at TU Munich achieved a quantum computing breakthrough, "
                "developing an algorithm up to 100 times faster than existing methods, "
                "with results published in Nature."
            ),
        },
        {
            "de": (
                "Die Bundesregierung hat ein umfassendes Klimaschutzpaket verabschiedet, "
                "das Investitionen in Höhe von 50 Milliarden Euro in erneuerbare Energien "
                "vorsieht. Das Paket beinhaltet den beschleunigten Ausbau von Wind- und "
                "Solarenergie sowie Förderprogramme für Elektromobilität. Umweltverbände "
                "begrüßten die Maßnahmen, forderten jedoch noch ambitioniertere Ziele."
            ),
            "reference": (
                "The German government passed a 50 billion euro climate package for "
                "renewable energy expansion, including accelerated wind and solar "
                "development and EV subsidies, though environmental groups called for "
                "more ambitious targets."
            ),
        },
    ]

    translator = Translator()
    translator.load()

    summarizer = Summarizer()
    summarizer.load()

    references = []
    hypotheses = []

    for item in eval_data[:max_samples]:
        # Translate DE -> EN (summarizer is English-only)
        translated = translator.translate(item["de"])
        # Summarize the translated text
        summary_result = summarizer.summarize(translated)
        references.append(item["reference"])
        hypotheses.append(summary_result.summary)

    summarizer.unload()
    translator.unload()

    metrics = compute_rouge(references, hypotheses)

    return {
        "rouge1": metrics.rouge1,
        "rouge2": metrics.rouge2,
        "rougeL": metrics.rougeL,
        "num_samples": metrics.num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Run evaluations")
    parser.add_argument(
        "--component",
        choices=["ner", "classifier", "asr", "translation", "summarization", "all"],
        default="all",
        help="Which component to evaluate",
    )
    parser.add_argument("--model-path", type=str, default="models/event_classifier")
    parser.add_argument("--max-samples", type=int, default=50, help="Max ASR/NER samples")
    args = parser.parse_args()

    results = {}

    if args.component in ("ner", "all"):
        results["ner"] = evaluate_ner(max_samples=args.max_samples)

    if args.component in ("classifier", "all"):
        results["classifier"] = evaluate_classification(args.model_path)

    if args.component in ("asr", "all"):
        results["asr"] = evaluate_asr(args.max_samples)

    if args.component in ("translation", "all"):
        results["translation"] = evaluate_translation(args.max_samples)

    if args.component in ("summarization", "all"):
        results["summarization"] = evaluate_summarization(args.max_samples)

    # Save
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)
    with open(results_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
