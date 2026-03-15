"""Scrape news articles and process them through the NLP pipeline.

Usage:
    # Scrape from English RSS feed
    python -m scripts.scrape_and_process --source "Top News" --max-articles 3

    # Scrape from German RSS feed (cross-lingual demo)
    python -m scripts.scrape_and_process --source "Tagesschau" --max-articles 3

    # Scrape a single URL (English or German)
    python -m scripts.scrape_and_process --url "https://www.tagesschau.de/..."

    # Without summarization
    python -m scripts.scrape_and_process --source "Technology" --no-summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.scraper import (
    RSS_FEEDS,
    ScrapeError,
    scrape_article,
    scrape_from_rss,
)
from src.pipeline import NewsPipeline
from src.utils.logging import get_logger

logger = get_logger("scrape_and_process")


def process_articles(
    articles: list[dict],
    include_summary: bool = True,
) -> list[dict]:
    """Run scraped articles through the NLP pipeline.

    Args:
        articles: List of article dicts from scraper.
        include_summary: Whether to generate summaries.

    Returns:
        List of combined scrape metadata + pipeline results.
    """
    pipeline = NewsPipeline()
    results = []

    for i, article in enumerate(articles):
        print(f"\n[{i + 1}/{len(articles)}] Processing: {article['title'][:60]}...")

        try:
            result = pipeline.run(
                article["cleaned_text"],
                include_summary=include_summary,
            )

            combined = {
                "url": article["url"],
                "title": article["title"],
                "scraped_at": article["scraped_at"],
                "word_count": article["word_count"],
                "pipeline_result": result.to_dict(),
            }
            results.append(combined)

        except Exception as e:
            logger.error(
                f"Pipeline error for {article['url']}: {e}",
                extra={"component": "scrape_and_process"},
            )
            continue

    return results


def print_summary_table(results: list[dict]) -> None:
    """Print a summary table of processed articles."""
    print(f"\n{'=' * 90}")
    print(f"  RESULTS SUMMARY  ({len(results)} articles)")
    print(f"{'=' * 90}")
    print(f"{'Title':<35} {'Lang':<5} {'Classification':<18} {'Entities':<10} {'Words':<8}")
    print(f"{'-' * 35} {'-' * 5} {'-' * 18} {'-' * 10} {'-' * 8}")

    for r in results:
        title = r["title"][:33] + ".." if len(r["title"]) > 35 else r["title"]
        pr = r["pipeline_result"]
        lang = pr["detected_language"].upper()
        label = pr["classification"]["label"]
        confidence = pr["classification"]["score"]
        entity_count = len(pr["entities"])
        words = r["word_count"]
        print(
            f"{title:<35} {lang:<5} {label} ({confidence:.0%}){'':<3} {entity_count:<10} {words:<8}"
        )

    # Print summary if available
    has_summaries = any(r["pipeline_result"].get("summary") for r in results)
    if has_summaries:
        print(f"\n{'=' * 90}")
        print("  SUMMARIES")
        print(f"{'=' * 90}")
        for r in results:
            summary = r["pipeline_result"].get("summary")
            if summary:
                title = r["title"][:60]
                print(f"\n  [{r['pipeline_result']['detected_language'].upper()}] {title}")
                print(f"  {summary['summary'][:200]}")

    print(f"{'=' * 90}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape news articles and process through NLP pipeline"
    )
    parser.add_argument(
        "--source",
        default="Top News",
        choices=list(RSS_FEEDS.keys()),
        help="RSS feed: English (Top News, World, Technology, Business, Sports) "
        "or German (Tagesschau, Spiegel, DW German, ZDF)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=5,
        help="Maximum articles to scrape (default: 5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between requests (default: 2.0)",
    )
    parser.add_argument(
        "--include-summary",
        action="store_true",
        default=True,
        dest="include_summary",
        help="Include summarization (default: True)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_false",
        dest="include_summary",
        help="Skip summarization",
    )
    parser.add_argument(
        "--output",
        default="results/scraped_results.json",
        help="Output JSON file path (default: results/scraped_results.json)",
    )
    parser.add_argument(
        "--url",
        help="Scrape a single URL instead of RSS feed",
    )
    args = parser.parse_args()

    # Scrape articles
    if args.url:
        print(f"Scraping single URL: {args.url}")
        try:
            article = scrape_article(args.url)
            articles = [article]
        except ScrapeError as e:
            print(f"Error: {e}")
            return
    else:
        feed_url = RSS_FEEDS[args.source]
        print(f"Scraping RSS feed: {args.source} ({feed_url})")
        print(f"Max articles: {args.max_articles}, delay: {args.delay}s")
        articles = scrape_from_rss(
            feed_url,
            max_articles=args.max_articles,
            delay=args.delay,
        )

    if not articles:
        print("No articles scraped. Exiting.")
        return

    print(f"\nScraped {len(articles)} articles. Running pipeline...")

    # Process through pipeline
    results = process_articles(articles, include_summary=args.include_summary)

    if not results:
        print("No articles processed successfully. Exiting.")
        return

    # Print summary table
    print_summary_table(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
