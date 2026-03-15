# Multilingual News NLP Pipeline

End-to-end pipeline for processing German news content: speech-to-text, named entity recognition, event classification, translation, and summarization. Built with production constraints in mind (4GB VRAM GPU, sequential model loading).

## Architecture

```
Audio (German) --> Whisper ASR --> Language Detection --> Cross-Lingual NER
                                                              |
                                                              v
                                                    Event Classification
                                                              |
                                                              v
                                                  Translation (DE -> EN)
                                                              |
                                                              v
                                                       Summarization
                                                              |
                                                              v
                                                     Structured JSON
                                                     (via FastAPI)
```

### Design Decisions

1. **Cross-lingual NER over translate-then-NER** -- Translating German text to English before NER corrupts entity boundaries (e.g., compound nouns split incorrectly). XLM-RoBERTa handles German directly with zero-shot cross-lingual transfer.

2. **Sequential model loading** -- With 4GB VRAM (RTX 3050), models are loaded one at a time with explicit `load()`/`unload()` and `torch.cuda.empty_cache()`. The largest model (XLM-RoBERTa NER) uses ~1067 MB.

3. **CTranslate2 for Whisper** -- The German-finetuned Whisper model is converted to CTranslate2 format with int8 quantization, reducing memory footprint while maintaining accuracy.

4. **FP16 inference** -- All transformer models use float16 precision, halving VRAM usage without measurable quality loss.

## Models

| Component | Model | VRAM | Latency (P50) |
|-----------|-------|------|---------------|
| ASR | whisper-tiny-german (CTranslate2, int8) | ~0 MB* | ~200ms |
| NER | xlm-roberta-large-finetuned-conll03-german (FP16) | 1067 MB | 43ms |
| Classifier | distilbert-base-multilingual-cased (fine-tuned) | 525 MB | 7ms |
| Summarizer | sshleifer/distilbart-cnn-12-6 | 787 MB | 578ms |
| Translator | Helsinki-NLP/opus-mt-de-en | 521 MB | 185ms |

*CTranslate2 manages its own memory pool outside PyTorch's CUDA allocator.

## Evaluation Results

| Component | Metric | Score | Dataset |
|-----------|--------|-------|---------|
| NER | F1 (overall) | 0.630 | WikiANN German (500 samples) |
| NER | F1 (PER) | 0.745 | WikiANN German |
| NER | F1 (LOC) | 0.649 | WikiANN German |
| NER | F1 (ORG) | 0.528 | WikiANN German |
| Classifier | Accuracy | 93.55% | 10kGNAD (806 test samples) |
| Classifier | Macro F1 | 94.07% | 10kGNAD |
| Classifier | Balanced Accuracy | — | 10kGNAD |
| Classifier | ROC-AUC (macro) | — | 10kGNAD |
| Classifier | MCC | — | 10kGNAD |
| ASR | WER | 12.10% | TTS-generated German (15 samples) |
| ASR | CER | — | TTS-generated German (15 samples) |
| Translation | BLEU | — | Curated DE→EN parallel sentences (10 samples) |
| Translation | ChrF | — | Curated DE→EN parallel sentences |
| Translation | BERTScore F1 | — | Curated DE→EN parallel sentences |
| Summarization | ROUGE-1 | — | Curated German news (5 samples) |
| Summarization | ROUGE-2 | — | Curated German news |
| Summarization | ROUGE-L | — | Curated German news |

### Classifier Per-Class Performance

| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Political | 0.926 | 0.912 | 0.941 |
| Economic | 0.909 | 0.934 | 0.885 |
| Sports | 0.983 | 0.983 | 0.983 |
| Technology | 0.945 | 0.939 | 0.951 |

### Notes on NER Evaluation

The NER F1 of 0.63 reflects token-to-character alignment challenges between WikiANN's tokenization and the model's subword tokenization. Qualitative performance is substantially better -- the model correctly extracts entities like "Angela Merkel" (PER), "Berlin" (LOC), "Europaeische Union" (ORG) from German text with high confidence.

## Production Benchmarks

Measured on NVIDIA GeForce RTX 3050 Laptop GPU (4096 MB VRAM), 10 runs with 3 warmup, CUDA-synchronized timing:

| Component | Mean Latency | P95 Latency | Peak VRAM | Throughput |
|-----------|-------------|-------------|-----------|------------|
| NER | 43.8 ms | 64.5 ms | 1067 MB | 22.9 items/s |
| Classifier | 6.9 ms | 8.3 ms | 525 MB | 145.3 items/s |
| Summarizer | 577.8 ms | 630.8 ms | 787 MB | 1.7 items/s |
| Translator | 197.1 ms | 270.1 ms | 521 MB | 5.1 items/s |

### MLflow Experiment Tracking

Training runs are tracked with MLflow, logging hyperparameters, per-epoch loss/accuracy curves, and test metrics:

```bash
# Train with MLflow tracking (default)
make train

# View results
make mlflow  # opens MLflow UI at http://localhost:5000
```

Tracked metrics include: `train_loss`, `val_accuracy`, `test_accuracy`, `test_macro_f1`, and per-class F1/precision/recall. Model artifacts are logged for reproducibility.

## Setup

### Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA support
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Create virtual environment
uv venv --python 3.12

# Activate (Windows)
.\venv\Scripts\activate

# Install PyTorch with CUDA first (critical -- must be before other deps)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
uv pip install -r requirements.txt

# Re-pin PyTorch CUDA (uv may replace with CPU-only)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
```

### Environment Variables

```bash
export HF_HOME=C:/hf_cache  # Avoid Windows path length issues
```

### Convert Whisper Model (one-time)

```bash
python -c "
import ctranslate2
converter = ctranslate2.converters.TransformersConverter('primeline/whisper-tiny-german-1224')
converter.convert('models/whisper-tiny-german-ct2', quantization='int8')
"
```

### Train Classifier (one-time)

```bash
python -m scripts.train_classifier
```

## Usage

### FastAPI Server

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `GET /health` -- Health check with GPU status
- `GET /models` -- List loaded models and memory usage
- `POST /extract` -- Text processing (NER + classification)
- `POST /asr/transcribe` -- Audio transcription
- `POST /pipeline` -- Full pipeline (text or audio input)

### Python API

```python
from src.pipeline import NewsPipeline

pipeline = NewsPipeline(sequential_mode=True)

result = pipeline.run(
    text="Angela Merkel traf sich mit Emmanuel Macron in Berlin.",
    include_summary=True
)

print(result.entities)       # [Entity(text='Angela Merkel', label='PER', ...)]
print(result.classification) # ClassificationResult(label='Political', score=0.99)
print(result.summary)        # "Angela Merkel met with Emmanuel Macron in Berlin..."
```

### Running Evaluations

```bash
# All evaluations (NER + Classifier + ASR + Translation + Summarization)
python -m scripts.evaluate

# Individual components
python -m scripts.evaluate --component ner --max-samples 500
python -m scripts.evaluate --component classifier
python -m scripts.evaluate --component asr --max-samples 15
python -m scripts.evaluate --component translation
python -m scripts.evaluate --component summarization
```

### Error Analysis Notebook

See [`notebooks/demo_and_error_analysis.ipynb`](notebooks/demo_and_error_analysis.ipynb) for:
- End-to-end pipeline demo with 4 German news articles
- NER error analysis (compound nouns, abbreviations, nested entities)
- Classifier confidence analysis on ambiguous texts
- Translation quality inspection on challenging cases
- VRAM usage profiling across sequential model loading

### Running Benchmarks

```bash
python -m scripts.run_benchmark
```

### Running Tests

```bash
# All tests (requires GPU)
python -m pytest tests/ -v

# CPU-only tests
python -m pytest tests/test_config.py tests/test_preprocessing.py tests/test_langdetect.py -v
```

## Project Structure

```
.
├── configs/
│   └── default.yaml          # Model, dataset, hardware configuration
├── data/
│   └── eval_audio/           # TTS-generated evaluation audio
├── docker/
│   ├── Dockerfile            # Multi-stage build with CUDA
│   └── docker-compose.yml    # GPU-enabled compose
├── models/
│   ├── event_classifier/     # Fine-tuned 4-class classifier
│   └── whisper-tiny-german-ct2/  # CTranslate2 Whisper model
├── results/
│   ├── evaluation_results.json
│   ├── classifier_results.json
│   └── asr_results.json
├── scripts/
│   ├── train_classifier.py
│   ├── evaluate.py
│   └── run_benchmark.py
├── src/
│   ├── api/
│   │   └── app.py            # FastAPI endpoints
│   ├── data/
│   │   ├── dataset.py        # Dataset loaders (10kGNAD, WikiANN, FLEURS)
│   │   └── preprocessing.py  # Text cleaning, normalization
│   ├── evaluation/
│   │   ├── benchmark.py      # Latency, VRAM, throughput benchmarks
│   │   └── metrics.py        # NER F1, classification metrics, WER
│   ├── models/
│   │   ├── asr.py            # Whisper ASR (faster-whisper / CTranslate2)
│   │   ├── classifier.py     # Event classifier (DistilBERT)
│   │   ├── ner.py            # Named entity recognition (XLM-RoBERTa)
│   │   ├── summarizer.py     # Summarization (DistilBART)
│   │   └── translator.py     # DE->EN translation (MarianMT)
│   ├── utils/
│   │   ├── config.py         # Dataclass config with YAML loading
│   │   └── logging.py        # JSON/console logging, TimingContext
│   ├── langdetect_util.py    # Language detection
│   └── pipeline.py           # Main pipeline orchestration
├── tests/                    # 53 tests (all passing)
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Tech Stack

- **NLP**: Transformers (XLM-RoBERTa, DistilBERT, DistilBART, MarianMT)
- **ASR**: faster-whisper (CTranslate2 backend)
- **API**: FastAPI + Uvicorn
- **ML**: PyTorch 2.6 (CUDA 12.4), HuggingFace Datasets
- **Evaluation**: seqeval (NER), jiwer (WER/CER), sacrebleu (BLEU/ChrF), bert-score, rouge-score, scikit-learn (classification)
- **MLOps**: MLflow (experiment tracking, model registry), Docker, GitHub Actions CI
- **Testing**: pytest (53 tests)
- **Tooling**: uv, ruff, Docker
