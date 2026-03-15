"""Production benchmarking: latency, VRAM, throughput for every component.

Measures real-world performance with proper CUDA synchronization,
warmup runs, and percentile reporting (P50, P95, P99).
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger("benchmark")


@dataclass
class LatencyStats:
    """Latency statistics from a benchmark run."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    num_runs: int


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    peak_vram_mb: float
    allocated_vram_mb: float
    reserved_vram_mb: float
    cpu_rss_mb: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a component."""

    component: str
    model_id: str
    latency: LatencyStats
    memory: MemoryStats
    throughput_items_per_sec: float
    device: str


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def measure_gpu_memory() -> MemoryStats:
    """Get current GPU and CPU memory stats."""
    cpu_rss = psutil.Process().memory_info().rss / 1024**2

    if torch.cuda.is_available():
        return MemoryStats(
            peak_vram_mb=round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            allocated_vram_mb=round(torch.cuda.memory_allocated() / 1024**2, 1),
            reserved_vram_mb=round(torch.cuda.memory_reserved() / 1024**2, 1),
            cpu_rss_mb=round(cpu_rss, 1),
        )
    return MemoryStats(
        peak_vram_mb=0.0,
        allocated_vram_mb=0.0,
        reserved_vram_mb=0.0,
        cpu_rss_mb=round(cpu_rss, 1),
    )


def time_function(
    fn,
    *args,
    warmup: int | None = None,
    num_runs: int | None = None,
    **kwargs,
) -> LatencyStats:
    """Time a function with warmup and CUDA synchronization.

    Args:
        fn: Function to benchmark.
        warmup: Number of warmup runs.
        num_runs: Number of timed runs.

    Returns:
        LatencyStats with mean, std, percentiles.
    """
    config = get_config().benchmark
    warmup = warmup if warmup is not None else config.warmup_runs
    num_runs = num_runs if num_runs is not None else config.num_runs
    device = get_device()

    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        fn(*args, **kwargs)

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    sorted_times = sorted(times)

    def percentile(p: int) -> float:
        idx = int(len(sorted_times) * p / 100)
        idx = min(idx, len(sorted_times) - 1)
        return round(sorted_times[idx], 2)

    return LatencyStats(
        mean_ms=round(statistics.mean(times), 2),
        std_ms=round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
        min_ms=round(min(times), 2),
        max_ms=round(max(times), 2),
        p50_ms=percentile(50),
        p95_ms=percentile(95),
        p99_ms=percentile(99),
        num_runs=num_runs,
    )


def cleanup_gpu() -> None:
    """Free GPU memory between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_ner(text: str | None = None) -> BenchmarkResult:
    """Benchmark NER component."""
    from src.models.ner import NERExtractor

    cleanup_gpu()

    test_text = text or (
        "Angela Merkel traf sich mit Emmanuel Macron in Berlin, "
        "um den wirtschaftlichen Aufschwung der Europäischen Union zu besprechen."
    )

    logger.info("Benchmarking NER...", extra={"component": "benchmark"})

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ner = NERExtractor()
    ner.load()
    memory = measure_gpu_memory()

    latency = time_function(ner.extract, test_text)
    throughput = 1000.0 / latency.mean_ms if latency.mean_ms > 0 else 0.0

    result = BenchmarkResult(
        component="NER",
        model_id=ner.config.model_id,
        latency=latency,
        memory=memory,
        throughput_items_per_sec=round(throughput, 1),
        device=ner.device,
    )

    ner.unload()

    logger.info(
        f"NER: {latency.mean_ms:.1f}ms mean, {latency.p95_ms:.1f}ms P95, "
        f"{memory.peak_vram_mb:.0f}MB peak VRAM",
        extra={"component": "benchmark"},
    )

    return result


def benchmark_classifier(text: str | None = None) -> BenchmarkResult:
    """Benchmark event classifier component."""
    from src.models.classifier import EventClassifier

    cleanup_gpu()

    test_text = text or (
        "Die Aktienmärkte erlebten einen deutlichen Abschwung, nachdem "
        "die Zentralbank höhere Zinssätze angekündigt hatte."
    )

    logger.info("Benchmarking Classifier...", extra={"component": "benchmark"})

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Use fine-tuned model if available
    from pathlib import Path

    model_path = "models/event_classifier" if Path("models/event_classifier").exists() else None
    classifier = EventClassifier(model_path=model_path)
    classifier.load()
    memory = measure_gpu_memory()

    latency = time_function(classifier.classify, test_text)
    throughput = 1000.0 / latency.mean_ms if latency.mean_ms > 0 else 0.0

    result = BenchmarkResult(
        component="Classifier",
        model_id=classifier.config.model_id,
        latency=latency,
        memory=memory,
        throughput_items_per_sec=round(throughput, 1),
        device=classifier.device,
    )

    classifier.unload()

    logger.info(
        f"Classifier: {latency.mean_ms:.1f}ms mean, {latency.p95_ms:.1f}ms P95, "
        f"{memory.peak_vram_mb:.0f}MB peak VRAM",
        extra={"component": "benchmark"},
    )

    return result


def benchmark_summarizer(text: str | None = None) -> BenchmarkResult:
    """Benchmark summarizer component."""
    from src.models.summarizer import Summarizer

    cleanup_gpu()

    test_text = text or (
        "The German chancellor announced new measures at the economic summit "
        "in Berlin to promote digital transformation across all sectors. "
        "The plan includes substantial investments in artificial intelligence "
        "research and development, as well as initiatives to improve "
        "digital infrastructure in rural areas. Industry leaders welcomed "
        "the announcement, calling it a significant step forward."
    )

    logger.info("Benchmarking Summarizer...", extra={"component": "benchmark"})

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    summarizer = Summarizer()
    summarizer.load()
    memory = measure_gpu_memory()

    latency = time_function(summarizer.summarize, test_text)
    throughput = 1000.0 / latency.mean_ms if latency.mean_ms > 0 else 0.0

    result = BenchmarkResult(
        component="Summarizer",
        model_id=summarizer.config.model_id,
        latency=latency,
        memory=memory,
        throughput_items_per_sec=round(throughput, 1),
        device=summarizer.device,
    )

    summarizer.unload()

    logger.info(
        f"Summarizer: {latency.mean_ms:.1f}ms mean, {latency.p95_ms:.1f}ms P95, "
        f"{memory.peak_vram_mb:.0f}MB peak VRAM",
        extra={"component": "benchmark"},
    )

    return result


def benchmark_translator(text: str | None = None) -> BenchmarkResult:
    """Benchmark translator component."""
    from src.models.translator import Translator

    cleanup_gpu()

    test_text = text or (
        "Der Bundeskanzler hat auf dem Wirtschaftsgipfel neue Maßnahmen zur "
        "Förderung der digitalen Transformation angekündigt."
    )

    logger.info("Benchmarking Translator...", extra={"component": "benchmark"})

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    translator = Translator()
    translator.load()
    memory = measure_gpu_memory()

    latency = time_function(translator.translate, test_text)
    throughput = 1000.0 / latency.mean_ms if latency.mean_ms > 0 else 0.0

    result = BenchmarkResult(
        component="Translator",
        model_id=translator.config.model_id,
        latency=latency,
        memory=memory,
        throughput_items_per_sec=round(throughput, 1),
        device=translator.device,
    )

    translator.unload()

    logger.info(
        f"Translator: {latency.mean_ms:.1f}ms mean, {latency.p95_ms:.1f}ms P95, "
        f"{memory.peak_vram_mb:.0f}MB peak VRAM",
        extra={"component": "benchmark"},
    )

    return result


def benchmark_asr(audio_path: str | None = None) -> BenchmarkResult:
    """Benchmark ASR component.

    Note: CTranslate2 manages its own memory pool outside PyTorch's CUDA
    allocator, so VRAM tracking shows only PyTorch-allocated memory.
    """
    from src.models.asr import ASRModel

    cleanup_gpu()

    # Use first eval audio sample if no path provided
    if audio_path is None:
        test_path = Path("data/eval_audio/sample_000.mp3")
        if not test_path.exists():
            logger.warning(
                "No eval audio found, skipping ASR benchmark",
                extra={"component": "benchmark"},
            )
            return None
        audio_path = str(test_path)

    logger.info("Benchmarking ASR...", extra={"component": "benchmark"})

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    asr = ASRModel()
    asr.load()
    memory = measure_gpu_memory()

    latency = time_function(asr.transcribe, audio_path)
    throughput = 1000.0 / latency.mean_ms if latency.mean_ms > 0 else 0.0

    result = BenchmarkResult(
        component="ASR",
        model_id=asr.config.model_id,
        latency=latency,
        memory=memory,
        throughput_items_per_sec=round(throughput, 1),
        device=asr.device,
    )

    asr.unload()

    logger.info(
        f"ASR: {latency.mean_ms:.1f}ms mean, {latency.p95_ms:.1f}ms P95, "
        f"{memory.peak_vram_mb:.0f}MB peak VRAM (PyTorch only, CT2 manages own pool)",
        extra={"component": "benchmark"},
    )

    return result


def run_all_benchmarks(output_path: str | None = None) -> dict[str, BenchmarkResult]:
    """Run benchmarks for all components sequentially.

    Args:
        output_path: Optional path to save results as JSON.

    Returns:
        Dict mapping component name to BenchmarkResult.
    """
    device = get_device()
    logger.info(f"Starting benchmarks on {device}", extra={"component": "benchmark"})

    if device == "cuda":
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}, "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB",
            extra={"component": "benchmark"},
        )

    results = {}
    results["ner"] = benchmark_ner()
    results["classifier"] = benchmark_classifier()
    results["summarizer"] = benchmark_summarizer()
    results["translator"] = benchmark_translator()

    asr_result = benchmark_asr()
    if asr_result is not None:
        results["asr"] = asr_result

    # Save results
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: asdict(v) for k, v in results.items()}
        with open(output_file, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {output_path}", extra={"component": "benchmark"})

    # Print summary table
    print("\n" + "=" * 80)
    print(
        f"{'Component':<15} {'Latency (mean)':<18} {'P95':<12} {'Peak VRAM':<12} {'Throughput':<15}"
    )
    print("=" * 80)
    for _name, r in results.items():
        print(
            f"{r.component:<15} {r.latency.mean_ms:>8.1f} ms      "
            f"{r.latency.p95_ms:>8.1f} ms "
            f"{r.memory.peak_vram_mb:>8.0f} MB  "
            f"{r.throughput_items_per_sec:>8.1f} items/s"
        )
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_all_benchmarks(output_path="benchmark_results/results.json")
