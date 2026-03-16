"""FastAPI application for serving the NLP pipeline.

Endpoints:
    POST /extract        - Text → NER + Classification
    POST /scrape         - URL → Scrape + NER + Classification
    POST /asr/transcribe - Audio file → Transcribed text
    POST /pipeline       - Audio → Full pipeline (ASR + NER + Classification)
    GET  /health         - Health check, GPU status
    GET  /models         - Model info and status
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger("api")


# --- Pydantic Models ---


class EntityResponse(BaseModel):
    text: str
    label: str
    confidence: float
    start: int
    end: int


class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]


class SummaryResponse(BaseModel):
    summary: str
    input_length: int
    output_length: int


class ExtractRequest(BaseModel):
    text: str = Field(..., max_length=10000, description="Input text (German or English)")
    include_summary: bool = Field(default=True, description="Whether to generate summary")


class ExtractResponse(BaseModel):
    request_id: str
    language_detected: str
    entities: list[EntityResponse]
    classification: ClassificationResponse
    summary: SummaryResponse | None = None
    processing_time_ms: float
    timings: dict[str, float] = {}


class TranscriptionResponse(BaseModel):
    request_id: str
    text: str
    language: str
    duration_s: float
    segments: list[dict]
    confidence: float
    processing_time_ms: float


class FullPipelineResponse(BaseModel):
    request_id: str
    transcribed_text: str
    language_detected: str
    entities: list[EntityResponse]
    classification: ClassificationResponse
    summary: SummaryResponse | None = None
    processing_time_ms: float
    timings: dict[str, float]


class ScrapeRequest(BaseModel):
    url: str = Field(..., description="URL of the article to scrape")
    include_summary: bool = Field(default=True, description="Whether to generate summary")


class ScrapeResponse(BaseModel):
    request_id: str
    url: str
    word_count: int
    language_detected: str
    entities: list[EntityResponse]
    classification: ClassificationResponse
    summary: SummaryResponse | None = None
    processing_time_ms: float
    timings: dict[str, float] = {}


class RssRequest(BaseModel):
    source: str = Field(..., description="RSS feed name (e.g. 'Top News', 'Tagesschau')")
    max_articles: int = Field(default=3, ge=1, le=10, description="Max articles to scrape")
    include_summary: bool = Field(default=True, description="Whether to generate summaries")


class RssArticleResult(BaseModel):
    title: str
    url: str
    language_detected: str
    entities: list[EntityResponse]
    classification: ClassificationResponse
    summary: SummaryResponse | None = None
    word_count: int


class RssResponse(BaseModel):
    source: str
    results: list[RssArticleResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str | None = None
    gpu_memory_total_mb: float | None = None
    gpu_memory_used_mb: float | None = None


class ModelInfoResponse(BaseModel):
    models: list[dict[str, Any]]


# --- Pipeline singleton ---

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline import NewsPipeline

        _pipeline = NewsPipeline()
    return _pipeline


# --- App lifecycle ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Starting API server", extra={"component": "api"})
    yield
    logger.info("Shutting down API server", extra={"component": "api"})


# --- App ---


config = get_config()

app = FastAPI(
    title="News NLP Pipeline API",
    description="Multilingual news processing: NER, Classification, Summarization",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dashboard static directory
_static_dir = Path(__file__).resolve().parent / "static"


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with GPU status."""
    gpu_available = torch.cuda.is_available()

    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_name=torch.cuda.get_device_name(0) if gpu_available else None,
        gpu_memory_total_mb=(
            round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
            if gpu_available
            else None
        ),
        gpu_memory_used_mb=(
            round(torch.cuda.memory_allocated() / 1024**2, 1) if gpu_available else None
        ),
    )


@app.get("/models", response_model=ModelInfoResponse)
async def model_info():
    """List all models and their status."""
    cfg = get_config()
    pipeline = get_pipeline()

    models = [
        {
            "name": "NER",
            "model_id": cfg.ner.model_id,
            "loaded": pipeline.ner.is_loaded,
            "device": cfg.ner.device,
            "precision": cfg.ner.precision,
        },
        {
            "name": "Classifier",
            "model_id": cfg.classifier.model_id,
            "loaded": pipeline.classifier.is_loaded,
            "device": cfg.classifier.device,
        },
        {
            "name": "Summarizer",
            "model_id": cfg.summarizer.model_id,
            "loaded": pipeline.summarizer.is_loaded,
            "device": cfg.summarizer.device,
        },
        {
            "name": "Translator",
            "model_id": cfg.translator.model_id,
            "loaded": pipeline.translator.is_loaded,
            "device": cfg.translator.device,
        },
        {
            "name": "ASR",
            "model_id": cfg.asr.model_id,
            "loaded": False,  # ASR is created per-request
            "device": cfg.asr.device,
            "compute_type": cfg.asr.compute_type,
        },
    ]

    return ModelInfoResponse(models=models)


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    """Extract entities, classify event, and optionally summarize text.

    Input: German or English news text
    Output: Entities, classification, optional summary
    """
    pipeline = get_pipeline()

    try:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(
            partial(pipeline.run, request.text, include_summary=request.include_summary)
        )
        total_ms = (time.perf_counter() - t0) * 1000

        entities = [
            EntityResponse(
                text=e.text,
                label=e.label,
                confidence=e.score,
                start=e.start,
                end=e.end,
            )
            for e in result.entities
        ]

        classification = ClassificationResponse(
            label=result.classification.label,
            confidence=result.classification.score,
            all_scores=result.classification.all_scores,
        )

        summary = None
        if result.summary:
            summary = SummaryResponse(
                summary=result.summary.summary,
                input_length=result.summary.input_length,
                output_length=result.summary.output_length,
            )

        return ExtractResponse(
            request_id=result.request_id,
            language_detected=result.detected_language,
            entities=entities,
            classification=classification,
            summary=summary,
            processing_time_ms=round(total_ms, 1),
            timings=result.timings,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Pipeline error: {e}", extra={"component": "api"})
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}") from e


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest):
    """Scrape an article from a URL and process through the NLP pipeline.

    Input: URL of a news article
    Output: Entities, classification, optional summary, word count
    """
    from src.data.scraper import ScrapeError, scrape_article

    pipeline = get_pipeline()

    def _scrape_and_run():
        art = scrape_article(request.url)
        res = pipeline.run(art["cleaned_text"], include_summary=request.include_summary)
        return art, res

    try:
        t0 = time.perf_counter()
        article, result = await asyncio.to_thread(_scrape_and_run)
        total_ms = (time.perf_counter() - t0) * 1000

        entities = [
            EntityResponse(
                text=e.text,
                label=e.label,
                confidence=e.score,
                start=e.start,
                end=e.end,
            )
            for e in result.entities
        ]

        classification = ClassificationResponse(
            label=result.classification.label,
            confidence=result.classification.score,
            all_scores=result.classification.all_scores,
        )

        summary = None
        if result.summary:
            summary = SummaryResponse(
                summary=result.summary.summary,
                input_length=result.summary.input_length,
                output_length=result.summary.output_length,
            )

        return ScrapeResponse(
            request_id=result.request_id,
            url=request.url,
            word_count=article["word_count"],
            language_detected=result.detected_language,
            entities=entities,
            classification=classification,
            summary=summary,
            processing_time_ms=round(total_ms, 1),
            timings=result.timings,
        )

    except ScrapeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Scrape pipeline error: {e}", extra={"component": "api"})
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}") from e


@app.post("/asr/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File()):
    """Transcribe an audio file to text using Whisper.

    Accepts: wav, mp3, flac, ogg (max 25MB)
    """
    from src.models.asr import ASRModel

    # Validate file
    max_size = config.api.max_audio_file_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {config.api.max_audio_file_mb}MB",
        )

    # Save temp file
    import tempfile

    suffix = Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()

        asr = ASRModel()
        asr.load()
        result = asr.transcribe(tmp_path)
        asr.unload()

        total_ms = (time.perf_counter() - t0) * 1000

        return TranscriptionResponse(
            request_id=str(uuid.uuid4())[:8],
            text=result.text,
            language=result.language,
            duration_s=result.duration_s,
            segments=result.segments,
            confidence=result.confidence,
            processing_time_ms=round(total_ms, 1),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail="Invalid audio file") from e
    except Exception as e:
        logger.error(f"ASR error: {e}", extra={"component": "api"})
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/pipeline", response_model=FullPipelineResponse)
async def full_pipeline(file: UploadFile = File()):
    """Full pipeline: Audio → ASR → NER → Classification → Summary.

    Accepts: wav, mp3, flac, ogg (max 25MB)
    """
    pipeline = get_pipeline()

    # Validate and save file
    max_size = config.api.max_audio_file_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {config.api.max_audio_file_mb}MB",
        )

    import tempfile

    suffix = Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        transcribed_text, result = pipeline.run_with_audio(tmp_path)
        total_ms = (time.perf_counter() - t0) * 1000

        entities = [
            EntityResponse(
                text=e.text,
                label=e.label,
                confidence=e.score,
                start=e.start,
                end=e.end,
            )
            for e in result.entities
        ]

        classification = ClassificationResponse(
            label=result.classification.label,
            confidence=result.classification.score,
            all_scores=result.classification.all_scores,
        )

        summary = None
        if result.summary:
            summary = SummaryResponse(
                summary=result.summary.summary,
                input_length=result.summary.input_length,
                output_length=result.summary.output_length,
            )

        return FullPipelineResponse(
            request_id=result.request_id,
            transcribed_text=transcribed_text,
            language_detected=result.detected_language,
            entities=entities,
            classification=classification,
            summary=summary,
            processing_time_ms=round(total_ms, 1),
            timings=result.timings,
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}", extra={"component": "api"})
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/rss", response_model=RssResponse)
async def process_rss(request: RssRequest):
    """Scrape articles from an RSS feed and process through the NLP pipeline.

    Input: RSS feed name and max articles
    Output: List of processed article results
    """
    from src.data.scraper import RSS_FEEDS, ScrapeError, scrape_from_rss

    if request.source not in RSS_FEEDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown feed: {request.source}. Available: {list(RSS_FEEDS.keys())}",
        )

    pipeline = get_pipeline()

    def _scrape_rss_and_run():
        feed_url = RSS_FEEDS[request.source]
        articles = scrape_from_rss(
            feed_url, max_articles=request.max_articles, delay=1.0
        )

        # Batch process: loads each model once for all articles
        texts = [a["cleaned_text"] for a in articles]
        pipeline_results = pipeline.run_batch(
            texts, include_summary=request.include_summary
        )

        results = []
        for article, result in zip(articles, pipeline_results):
            entities = [
                EntityResponse(
                    text=e.text,
                    label=e.label,
                    confidence=e.score,
                    start=e.start,
                    end=e.end,
                )
                for e in result.entities
            ]

            classification = ClassificationResponse(
                label=result.classification.label,
                confidence=result.classification.score,
                all_scores=result.classification.all_scores,
            )

            summary = None
            if result.summary:
                summary = SummaryResponse(
                    summary=result.summary.summary,
                    input_length=result.summary.input_length,
                    output_length=result.summary.output_length,
                )

            results.append(
                RssArticleResult(
                    title=article["title"],
                    url=article["url"],
                    language_detected=result.detected_language,
                    entities=entities,
                    classification=classification,
                    summary=summary,
                    word_count=article["word_count"],
                )
            )
        return results

    try:
        t0 = time.perf_counter()
        results = await asyncio.to_thread(_scrape_rss_and_run)
        total_ms = (time.perf_counter() - t0) * 1000

        return RssResponse(
            source=request.source,
            results=results,
            processing_time_ms=round(total_ms, 1),
        )

    except ScrapeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.error(f"RSS pipeline error: {e}", extra={"component": "api"})
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}") from e


@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the dashboard UI."""
    return FileResponse(str(_static_dir / "index.html"))
