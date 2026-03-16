"""Tests for configuration loading."""

from src.utils.config import PipelineConfig, load_config


class TestConfig:
    def test_loads_default_config(self):
        config = load_config()
        assert isinstance(config, PipelineConfig)

    def test_ner_config(self):
        config = load_config()
        assert "xlm-roberta" in config.ner.model_id
        assert config.ner.precision == "fp16"

    def test_classifier_config(self):
        config = load_config()
        assert "multilingual" in config.classifier.model_id
        assert config.classifier.num_labels == 4
        assert len(config.classifier.labels) == 4

    def test_asr_config(self):
        config = load_config()
        assert "whisper" in config.asr.model_id
        assert config.asr.compute_type == "int8"

    def test_summarizer_config(self):
        config = load_config()
        assert "distilbart" in config.summarizer.model_id

    def test_hardware_constraints(self):
        config = load_config()
        assert config.hardware.max_vram_gb == 4
        assert config.hardware.enable_sequential_loading is True

    def test_api_config(self):
        config = load_config()
        assert config.api.port == 8000
        assert config.api.max_audio_file_mb == 25

    def test_dataset_mapping(self):
        config = load_config()
        mapping = config.datasets.classification.category_mapping
        assert mapping["Inland"] == "Political"
        assert mapping["Wirtschaft"] == "Economic"
        assert mapping["Sport"] == "Sports"
        assert mapping["Web"] == "Technology"
