# Multilingual News NLP Pipeline

End-to-end NLP system for processing German news content — from raw audio to structured insights. Handles speech-to-text, named entity recognition, event classification, translation, and summarization through a unified FastAPI service with a live web dashboard.

Built for a 4GB VRAM constraint (RTX 3050), demonstrating production-grade model orchestration on consumer hardware.

## Key Results

| Metric | Value |
|--------|-------|
| NER F1 (German, WikiANN) | **0.647** |
| Event Classification Accuracy | **93.4%** |
| Cross-lingual NER vs translate-then-NER | **+13% F1, 8.4x faster** |
| ASR Word Error Rate | **12.1%** |
| End-to-end latency (text → structured JSON) | **~250ms** (without summary) |

## Architecture

```
Audio (German) ──► Whisper ASR ──► Language Detection ──► Cross-Lingual NER
                                                               │
                                                               ▼
                                                     Event Classification
                                                               │
                                                               ▼
                                                    Translation (DE → EN)
                                                               │
                                                               ▼
                                                        Summarization
                                                               │
                                                               ▼
                                                      Structured JSON
                                                       (via FastAPI)
```

### Design Decisions

**Cross-lingual NER over translate-then-NER** — Translating German text to English before NER corrupts entity boundaries. German compound words like "Bundesverfassungsgericht" become "Federal Constitutional Court" (1 word → 3), breaking character offsets. XLM-RoBERTa handles German directly with zero-shot cross-lingual transfer — **+13% F1 and 8.4x faster** on 200 samples.

**Smart VRAM caching** — NER + Classifier stay co-resident (~1.6GB), only evicted when summarization needs VRAM. Batch RSS processing uses stage-batched loading (4 model loads for N articles instead of 4N). Repeat request latency drops from ~15s to ~40ms.

**CTranslate2 for Whisper** — The German-finetuned Whisper model is converted to CTranslate2 format with int8 quantization, reducing memory footprint while maintaining accuracy.

**FP16 safetensors** — All transformer models pre-saved as FP16 locally. Halves disk I/O — NER load time dropped from 14.9s to 5.0s. Total model disk: ~2.1GB (down from ~4.1GB with FP32).

## Models

| Component | Model | VRAM | Latency (P50) |
|-----------|-------|------|---------------|
| ASR | whisper-tiny-german (CTranslate2, int8) | ~0 MB* | 289ms |
| NER | xlm-roberta-large-finetuned-conll03-german (FP16) | 1,067 MB | 43ms |
| Classifier | distilbert-base-multilingual-cased (fine-tuned) | 525 MB | 7ms |
| Translator | Helsinki-NLP/opus-mt-de-en (FP16) | 521 MB | 185ms |
| Summarizer | sshleifer/distilbart-cnn-12-6 (FP16) | 787 MB | 578ms |

*CTranslate2 manages its own memory pool outside PyTorch's CUDA allocator.

## Evaluation

### Named Entity Recognition (WikiANN German, 500 samples)

| Entity | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| PER | 0.798 | 0.801 | 0.794 |
| LOC | 0.664 | 0.609 | 0.730 |
| ORG | 0.524 | 0.754 | 0.402 |
| **Overall** | **0.647** | 0.643 | 0.650 |

PER scores highest — person names are consistent across languages. ORG is hardest because German organizational names are often long compound words that get partially tokenized.

### Event Classification (10kGNAD, 806 test samples)

| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Sports | 0.979 | 0.975 | 0.983 |
| Technology | 0.950 | 0.939 | 0.960 |
| Political | 0.921 | 0.918 | 0.925 |
| Economic | 0.907 | 0.925 | 0.889 |

**93.4% accuracy, 93.9% macro F1** — achieved in 3 epochs with gradient accumulation (effective batch size 16).

### Cross-Lingual NER vs Translate-then-NER (200 samples)

| Metric | Cross-Lingual | Translate-then-NER |
|--------|--------------|-------------------|
| F1 | **0.681** | 0.551 |
| PER F1 | **0.813** | 0.644 |
| LOC F1 | **0.686** | 0.560 |
| ORG F1 | **0.574** | 0.485 |
| Time | **9.3s** | 77.7s |

### ASR & Summarization

| Component | Metric | Score |
|-----------|--------|-------|
| ASR | WER | 12.1% |
| Summarization | ROUGE-1 / ROUGE-2 / ROUGE-L | 0.523 / 0.227 / 0.381 |

## Production Benchmarks

Measured on NVIDIA RTX 3050 Laptop GPU (4GB VRAM), 10 runs with 3 warmup, CUDA-synchronized timing:

| Component | Mean | P95 | Peak VRAM | Throughput |
|-----------|------|-----|-----------|------------|
| NER | 43.8ms | 64.5ms | 1,067 MB | 22.9 items/s |
| Classifier | 6.9ms | 8.3ms | 525 MB | 145.3 items/s |
| Summarizer | 577.8ms | 630.8ms | 787 MB | 1.7 items/s |
| Translator | 197.1ms | 270.1ms | 521 MB | 5.1 items/s |
| ASR | 304.0ms | 395.1ms | ~0 MB* | 3.3 items/s |

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
uv venv --python 3.11
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# PyTorch with CUDA (must be installed before other deps)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Dependencies
uv pip install -r requirements.txt
```

### One-time Setup

```bash
# Convert Whisper model to CTranslate2 format
python -c "
import ctranslate2
converter = ctranslate2.converters.TransformersConverter('primeline/whisper-tiny-german-1224')
converter.convert('models/whisper-tiny-german-ct2', quantization='int8')
"

# Train the event classifier
python -m scripts.train_classifier
```

### Run

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the web dashboard.

### Docker

```bash
cd docker
docker compose up --build
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard with entity highlighting |
| `GET` | `/health` | Health check with GPU status |
| `GET` | `/models` | Loaded models and memory usage |
| `POST` | `/extract` | NER + classification on text |
| `POST` | `/scrape` | URL scraping + NER + classification |
| `POST` | `/rss` | Batch RSS feed processing |
| `POST` | `/asr/transcribe` | Audio transcription |
| `POST` | `/pipeline` | Full pipeline (text or audio) |

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

## Evaluation & Benchmarks

```bash
# Run all evaluations
python -m scripts.evaluate

# Individual components
python -m scripts.evaluate --component ner --max-samples 500
python -m scripts.evaluate --component classifier
python -m scripts.evaluate --component asr --max-samples 15

# Latency benchmarks
python -m scripts.run_benchmark

# Cross-lingual vs translate-then-NER comparison
python -m scripts.compare_ner_approaches
```

## Tests

```bash
# All tests (requires GPU)
python -m pytest tests/ -v

# CPU-only tests
python -m pytest tests/test_config.py tests/test_preprocessing.py tests/test_langdetect.py -v
```

53 tests covering config loading, text preprocessing, language detection, NER, classification, translation, and API endpoints.

## Project Structure

```
├── configs/default.yaml              # Model, dataset, hardware config
├── docker/
│   ├── Dockerfile                    # Multi-stage build with CUDA 12.4
│   └── docker-compose.yml            # GPU-enabled compose
├── models/
│   ├── event_classifier/             # Fine-tuned DistilBERT (4 classes)
│   └── whisper-tiny-german-ct2/      # CTranslate2 int8 Whisper
├── notebooks/
│   └── demo_and_error_analysis.ipynb # Pipeline demo + error analysis
├── results/                          # Saved evaluation results (JSON)
├── scripts/
│   ├── train_classifier.py           # Classifier training
│   ├── evaluate.py                   # Component evaluations
│   ├── run_benchmark.py              # Latency + VRAM benchmarks
│   └── compare_ner_approaches.py     # Cross-lingual vs translate-then-NER
├── src/
│   ├── api/app.py                    # FastAPI + web dashboard
│   ├── models/                       # ASR, NER, classifier, translator, summarizer
│   ├── evaluation/                   # Metrics + benchmarking
│   ├── data/                         # Dataset loaders, preprocessing, scraping
│   └── pipeline.py                   # Pipeline orchestration
├── tests/                            # 53 tests
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Tech Stack

**NLP:** Transformers (XLM-RoBERTa, DistilBERT, DistilBART, MarianMT) · faster-whisper (CTranslate2)

**Backend:** FastAPI · Uvicorn · PyTorch 2.6 (CUDA 12.4)

**Evaluation:** seqeval · jiwer (WER) · sacrebleu · bert-score · rouge-score · scikit-learn

**Infrastructure:** Docker (multi-stage, GPU-enabled) · MLflow · pytest · uv · ruff

## License

MIT
