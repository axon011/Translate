"""Run production benchmarks for all pipeline components.

Usage:
    python -m scripts.run_benchmark
    python -m scripts.run_benchmark --output benchmark_results/results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run production benchmarks")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/results.json",
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    from src.evaluation.benchmark import run_all_benchmarks

    run_all_benchmarks(output_path=args.output)


if __name__ == "__main__":
    main()
