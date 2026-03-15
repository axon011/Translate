"""Train the event classifier on 10kGNAD with MLflow experiment tracking.

Usage:
    python -m scripts.train_classifier
    python -m scripts.train_classifier --epochs 5 --batch-size 4 --lr 2e-5
    python -m scripts.train_classifier --no-mlflow  # disable tracking
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import LABEL2ID, load_10kgnad
from src.evaluation.metrics import compute_classification_metrics
from src.models.classifier import EventClassifier
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger("train")

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Train event classifier on 10kGNAD")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--save-path", type=str, default="models/event_classifier", help="Save path"
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    args = parser.parse_args()

    use_mlflow = MLFLOW_AVAILABLE and not args.no_mlflow

    config = get_config()

    # Setup MLflow
    if use_mlflow:
        mlflow.set_experiment("news-classifier")
        logger.info("MLflow tracking enabled", extra={"component": "train"})

    # 1. Load data
    logger.info("Loading 10kGNAD dataset...", extra={"component": "train"})
    train_split, val_split, test_split = load_10kgnad(val_ratio=args.val_ratio)

    logger.info(
        f"Dataset: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test",
        extra={"component": "train"},
    )

    # Start MLflow run (or use no-op context)
    run_context = mlflow.start_run() if use_mlflow else _nullcontext()
    with run_context:
        # Log hyperparameters
        train_params = {
            "model": config.classifier.model_id,
            "epochs": args.epochs or config.classifier.train.num_epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.lr or config.classifier.train.learning_rate,
            "weight_decay": config.classifier.train.weight_decay,
            "warmup_steps": config.classifier.train.warmup_steps,
            "max_length": config.classifier.max_length,
            "num_labels": config.classifier.num_labels,
            "val_ratio": args.val_ratio,
            "train_samples": len(train_split),
            "val_samples": len(val_split),
            "test_samples": len(test_split),
        }
        if use_mlflow:
            mlflow.log_params(train_params)

        # 2. Initialize classifier
        classifier = EventClassifier(device=args.device)
        classifier.load()

        # 3. Train
        logger.info("Starting training...", extra={"component": "train"})
        history = classifier.train(
            train_texts=train_split.texts,
            train_labels=train_split.labels,
            val_texts=val_split.texts if val_split.texts else None,
            val_labels=val_split.labels if val_split.labels else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_accum_steps=args.grad_accum,
            save_path=args.save_path,
        )

        # Log training curves to MLflow
        if use_mlflow:
            for epoch, loss in enumerate(history.get("train_loss", []), 1):
                mlflow.log_metric("train_loss", loss, step=epoch)
            for epoch, val_acc in enumerate(history.get("val_accuracy", []), 1):
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # 4. Evaluate on test set
        logger.info("Evaluating on test set...", extra={"component": "train"})

        # Reload the saved model for clean evaluation
        eval_classifier = EventClassifier(model_path=args.save_path, device=args.device)
        eval_classifier.load()

        pred_labels = []
        pred_probs = []
        label_names = list(LABEL2ID.keys())

        for text in test_split.texts:
            result = eval_classifier.classify(text)
            pred_labels.append(LABEL2ID[result.label])
            pred_probs.append([result.all_scores[name] for name in label_names])

        metrics = compute_classification_metrics(
            true_labels=test_split.labels,
            pred_labels=pred_labels,
            label_names=label_names,
            pred_probs=pred_probs,
        )

        # Log test metrics to MLflow
        if use_mlflow:
            log_dict = {
                "test_accuracy": metrics.accuracy,
                "test_macro_f1": metrics.macro_f1,
                "test_balanced_accuracy": metrics.balanced_accuracy,
                "test_mcc": metrics.mcc,
            }
            if metrics.roc_auc is not None:
                log_dict["test_roc_auc"] = metrics.roc_auc
            mlflow.log_metrics(log_dict)
            for cls_name, cls_metrics in metrics.per_class.items():
                mlflow.log_metric(f"test_f1_{cls_name.lower()}", cls_metrics["f1"])
                mlflow.log_metric(f"test_precision_{cls_name.lower()}", cls_metrics["precision"])
                mlflow.log_metric(f"test_recall_{cls_name.lower()}", cls_metrics["recall"])

            # Log model artifact
            mlflow.log_artifacts(args.save_path, artifact_path="model")

        # 5. Save results
        results = {
            "training_history": history,
            "test_metrics": {
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "balanced_accuracy": metrics.balanced_accuracy,
                "roc_auc": metrics.roc_auc,
                "mcc": metrics.mcc,
                "per_class": metrics.per_class,
                "confusion_matrix": metrics.confusion_matrix,
                "num_samples": metrics.num_samples,
            },
            "config": train_params,
        }

        results_path = Path("results")
        results_path.mkdir(exist_ok=True)
        with open(results_path / "classifier_results.json", "w") as f:
            json.dump(results, f, indent=2)

        if use_mlflow:
            mlflow.log_artifact(str(results_path / "classifier_results.json"))

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Model saved to: {args.save_path}")
        print(f"Test Accuracy:      {metrics.accuracy:.4f}")
        print(f"Test Macro F1:      {metrics.macro_f1:.4f}")
        print(f"Test Balanced Acc:  {metrics.balanced_accuracy:.4f}")
        print(f"Test MCC:           {metrics.mcc:.4f}")
        if metrics.roc_auc is not None:
            print(f"Test ROC-AUC:       {metrics.roc_auc:.4f}")
        print("\nPer-class metrics:")
        for cls_name, cls_metrics in metrics.per_class.items():
            print(
                f"  {cls_name:<12s} F1={cls_metrics['f1']:.4f}  "
                f"P={cls_metrics['precision']:.4f}  R={cls_metrics['recall']:.4f}"
            )
        if use_mlflow:
            print(f"\nMLflow run: {mlflow.active_run().info.run_id}")
            print("View results: mlflow ui --port 5000")
        print("=" * 60)

        eval_classifier.unload()


class _nullcontext:
    """Minimal no-op context manager for when MLflow is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
