.PHONY: install train evaluate benchmark test lint api docker clean mlflow evaluate-summary scrape

# Install dependencies
install:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
	pip install -r requirements.txt

# Train classifier on 10kGNAD
train:
	python -m scripts.train_classifier --epochs 3 --batch-size 4 --grad-accum 4

# Run all evaluations
evaluate:
	python -m scripts.evaluate --component all

# Run NER evaluation only
evaluate-ner:
	python -m scripts.evaluate --component ner

# Run classifier evaluation only
evaluate-cls:
	python -m scripts.evaluate --component classifier

# Run ASR evaluation only
evaluate-asr:
	python -m scripts.evaluate --component asr --max-samples 50

# Run translation evaluation (BLEU, ChrF, BERTScore)
evaluate-translation:
	python -m scripts.evaluate --component translation

# Run summarization evaluation (ROUGE)
evaluate-summary:
	python -m scripts.evaluate --component summarization

# Run production benchmarks
benchmark:
	python -m scripts.run_benchmark

# Run tests
test:
	pytest tests/ -v --tb=short

# Run tests without GPU-dependent tests
test-fast:
	pytest tests/test_config.py tests/test_preprocessing.py tests/test_langdetect.py -v

# Lint
lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

# Format
format:
	ruff format src/ tests/ scripts/

# Start API server
api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Docker build and run
docker:
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down

# MLflow UI
mlflow:
	mlflow ui --port 5000

# Scrape news articles and process through pipeline
scrape:
	python -m scripts.scrape_and_process --max-articles 5

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf benchmark_results/ results/ .pytest_cache/
