# requirements: pip install feedparser requests beautifulsoup4

"""Lightweight Reuters RSS headline viewer.

For full article scraping and pipeline processing, use:
    python -m scripts.scrape_and_process
"""

import json
from datetime import datetime

import feedparser

from src.data.scraper import REUTERS_RSS_FEEDS


def scrape_reuters_rss(feed_name: str = "Top News", max_headlines: int = 20) -> list[dict]:
    """
    Scrape Reuters headlines from their official RSS feed.

    Args:
        feed_name: One of the keys in REUTERS_RSS_FEEDS
        max_headlines: Maximum number of headlines to return

    Returns:
        List of dicts with keys: title, link, published, summary
    """
    url = REUTERS_RSS_FEEDS.get(feed_name)
    if not url:
        raise ValueError(f"Invalid feed. Choose from: {list(REUTERS_RSS_FEEDS.keys())}")

    feed = feedparser.parse(url)

    if feed.bozo:  # feedparser sets bozo=True on parse errors
        raise ConnectionError(f"Failed to parse RSS feed: {feed.bozo_exception}")

    headlines = []
    for entry in feed.entries[:max_headlines]:
        headlines.append(
            {
                "title": entry.get("title", "N/A"),
                "link": entry.get("link", "N/A"),
                "published": entry.get("published", "N/A"),
                "summary": entry.get("summary", "N/A"),
            }
        )

    return headlines


def display_headlines(headlines: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  REUTERS HEADLINES  --  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}\n")
    for i, item in enumerate(headlines, 1):
        print(f"[{i:02d}] {item['title']}")
        print(f"      {item['link']}")
        print(f"      {item['published']}\n")


def save_to_json(headlines: list[dict], filename: str = "reuters_headlines.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(headlines, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(headlines)} headlines to '{filename}'")


if __name__ == "__main__":
    headlines = scrape_reuters_rss(feed_name="Top News", max_headlines=15)
    display_headlines(headlines)
    save_to_json(headlines)
