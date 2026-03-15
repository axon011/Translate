"""Configuration management using dataclasses and YAML.

Loads settings from configs/default.yaml and provides typed access
to all model, training, and deployment parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Find the project root (directory containing configs/)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "configs").is_dir():
            return current
        current = current.parent
    # Fallback to cwd
    return Path.cwd()


PROJECT_ROOT = _project_root()


@dataclass
class ASRConfig:
    model_id: str = "primeline/whisper-tiny-german-1224"
    compute_type: str = "int8"
    device: str = "cuda"
    language: str = "de"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0


@dataclass
class NERConfig:
    model_id: str = "xlm-roberta-large-finetuned-conll03-german"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512
    precision: str = "fp16"
    aggregation_strategy: str = "simple"


@dataclass
class TrainConfig:
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 500


@dataclass
class ClassifierConfig:
    model_id: str = "distilbert-base-multilingual-cased"
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512
    num_labels: int = 4
    labels: list[str] = field(
        default_factory=lambda: ["Political", "Economic", "Sports", "Technology"]
    )
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class SummarizerConfig:
    model_id: str = "sshleifer/distilbart-cnn-12-6"
    device: str = "cuda"
    max_length: int = 130
    min_length: int = 30
    length_penalty: float = 2.0
    num_beams: int = 4
    early_stopping: bool = True


@dataclass
class TranslatorConfig:
    model_id: str = "Helsinki-NLP/opus-mt-de-en"
    device: str = "cuda"
    max_length: int = 512
    num_beams: int = 4


@dataclass
class LangDetectConfig:
    languages: list[str] = field(default_factory=lambda: ["de", "en"])
    confidence_threshold: float = 0.8


@dataclass
class ClassificationDatasetConfig:
    name: str = "10kGNAD"
    hf_id: str = "philschmid/10kGNAD"
    category_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "Inland": "Political",
            "International": "Political",
            "Wirtschaft": "Economic",
            "Etat": "Economic",
            "Sport": "Sports",
            "Web": "Technology",
            "Wissenschaft": "Technology",
        }
    )


@dataclass
class NEREvalDatasetConfig:
    name: str = "WikiANN-de"
    hf_id: str = "wikiann"
    language: str = "de"


@dataclass
class ASREvalDatasetConfig:
    name: str = "FLEURS"
    hf_id: str = "google/fleurs"
    language: str = "de_de"
    split: str = "test"


@dataclass
class DatasetsConfig:
    classification: ClassificationDatasetConfig = field(default_factory=ClassificationDatasetConfig)
    ner_eval: NEREvalDatasetConfig = field(default_factory=NEREvalDatasetConfig)
    asr_eval: ASREvalDatasetConfig = field(default_factory=ASREvalDatasetConfig)


@dataclass
class HardwareConfig:
    max_vram_gb: int = 4
    gpu_device: str = "cuda:0"
    enable_sequential_loading: bool = True
    clear_cache_between_models: bool = True


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_audio_file_mb: int = 25
    max_text_length: int = 10000


@dataclass
class BenchmarkConfig:
    warmup_runs: int = 3
    num_runs: int = 10
    measure_vram: bool = True
    measure_latency: bool = True
    percentiles: list[int] = field(default_factory=lambda: [50, 95, 99])
    output_dir: str = "benchmark_results"


@dataclass
class PathsConfig:
    cache_dir: str = "C:\\hf_cache"
    model_dir: str = "models"
    data_dir: str = "data"
    results_dir: str = "results"


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire pipeline."""

    asr: ASRConfig = field(default_factory=ASRConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    lang_detect: LangDetectConfig = field(default_factory=LangDetectConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    api: APIConfig = field(default_factory=APIConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _apply_dict(obj: Any, data: dict) -> None:
    """Recursively apply dict values onto a dataclass instance."""
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dict(current, value)
        else:
            # Coerce to the dataclass field's declared type
            if isinstance(current, float) and isinstance(value, str):
                value = float(value)
            elif isinstance(current, int) and isinstance(value, str):
                value = int(value)
            setattr(obj, key, value)


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. Defaults to configs/default.yaml.

    Returns:
        PipelineConfig with all settings populated.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    config = PipelineConfig()

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _apply_dict(config, raw)

    # Set HF_HOME env var for model caching
    os.environ.setdefault("HF_HOME", config.paths.cache_dir)

    return config


# Singleton for convenience
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    """Get the global configuration (loads once, caches)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
