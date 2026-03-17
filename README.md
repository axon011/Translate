# Multilingual News NLP Pipeline

End-to-end pipeline for processing German news content: speech-to-text, named entity recognition, event classification, translation, and summarization. Includes a live web dashboard and RSS feed processing. Built with production constraints in mind (4GB VRAM, smart VRAM caching).

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

2. **Smart VRAM caching** -- With 4GB VRAM (RTX 3050), NER + Classifier stay loaded together (~1.6GB), only evicted when summarization needs VRAM. Batch RSS processing uses stage-batched loading (4 model loads for N articles instead of 4N). Repeat request latency ~40ms vs ~15s with full reload.

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

### NER (WikiANN German, 500 samples)

| Entity | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| PER | 0.798 | 0.801 | 0.794 |
| LOC | 0.664 | 0.609 | 0.730 |
| ORG | 0.524 | 0.754 | 0.402 |
| **Overall** | **0.647** | 0.643 | 0.650 |

PER scores highest because person names are consistent across languages. ORG is hardest -- German organizational names are often long compound words that get partially tokenized. F1 improved from 0.63 to 0.647 after switching to overlap-based token alignment (>50% character overlap).

### Classifier (10kGNAD, 806 test samples)

| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Sports | 0.979 | 0.975 | 0.983 |
| Technology | 0.950 | 0.939 | 0.960 |
| Political | 0.921 | 0.918 | 0.925 |
| Economic | 0.907 | 0.925 | 0.889 |

**Overall: 93.4% accuracy, 93.9% macro F1** -- achieved in just 3 epochs with gradient accumulation (effective batch size 16). Sports is easiest (distinct vocabulary), Economic is hardest (overlaps with Political on fiscal policy topics). Training loss: 0.58 → 0.20 → 0.12.

### Cross-Lingual NER vs Translate-then-NER (200 samples)

| Metric | Cross-Lingual | Translate-then-NER |
|--------|--------------|-------------------|
| F1 | **0.681** | 0.551 |
| PER F1 | **0.813** | 0.644 |
| LOC F1 | **0.686** | 0.560 |
| ORG F1 | **0.574** | 0.485 |
| Time | **9.3s** | 77.7s |

Cross-lingual wins by **+13% F1** and is **8.4x faster**. Translation corrupts entity boundaries -- German compound words like "Bundesverfassungsgericht" become "Federal Constitutional Court" (1 word → 3), breaking character offsets and entity spans.

### ASR & Summarization

| Component | Metric | Score |
|-----------|--------|-------|
| ASR | WER | 12.1% (15 TTS-generated German samples) |
| Summarization | ROUGE-1 / ROUGE-2 / ROUGE-L | 0.523 / 0.227 / 0.381 |

ASR errors are mostly format differences ("drei" vs "3") and minor spelling variations. ROUGE scores reflect abstractive generation -- the model paraphrases rather than copying, so exact n-gram overlap is naturally lower.

## Production Benchmarks

Measured on NVIDIA GeForce RTX 3050 Laptop GPU (4096 MB VRAM), 10 runs with 3 warmup, CUDA-synchronized timing:

| Component | Mean Latency | P95 Latency | Peak VRAM | Throughput |
|-----------|-------------|-------------|-----------|------------|
| NER | 43.8 ms | 64.5 ms | 1067 MB | 22.9 items/s |
| Classifier | 6.9 ms | 8.3 ms | 525 MB | 145.3 items/s |
| Summarizer | 577.8 ms | 630.8 ms | 787 MB | 1.7 items/s |
| Translator | 197.1 ms | 270.1 ms | 521 MB | 5.1 items/s |
| ASR | 304.0 ms | 395.1 ms | ~0 MB* | 3.3 items/s |

*ASR benchmarked on 7.2s German audio clip.

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Create virtual environment
uv venv --python 3.11

# Activate (Windows)
.\.venv\Scripts\activate

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
# Visit http://localhost:8000 for the web dashboard
```

**Endpoints:**
- `GET /` -- Web dashboard UI (entity highlighting, classification bars, timing breakdown)
- `GET /health` -- Health check with GPU status
- `GET /models` -- List loaded models and memory usage
- `POST /extract` -- Text processing (NER + classification + summary)
- `POST /scrape` -- URL scraping + NER + classification
- `POST /rss` -- RSS feed processing (batch, stage-optimized)
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
│   ├── run_benchmark.py
│   ├── compare_ner_approaches.py
│   ├── compare_live.py
│   ├── scrape_and_process.py
│   └── scrap.py
├── src/
│   ├── api/
│   │   ├── app.py            # FastAPI endpoints + async inference
│   │   └── static/
│   │       └── index.html    # Web dashboard UI
│   ├── data/
│   │   ├── dataset.py        # Dataset loaders (10kGNAD, WikiANN, FLEURS)
│   │   ├── preprocessing.py  # Text cleaning, normalization
│   │   └── scraper.py        # Web scraping + RSS feeds
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
- **Web UI**: Single-page dashboard (vanilla HTML/CSS/JS, served from FastAPI)
- **Web Scraping**: requests + BeautifulSoup4, feedparser (RSS)
- **MLOps**: MLflow (experiment tracking, model registry), Docker
- **Testing**: pytest (53 tests)
- **Tooling**: uv, ruff, Docker
