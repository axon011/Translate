"""Web scraping module for fetching real news articles.

Fetches full article text from news websites and RSS feeds,
cleans and prepares it for the NLP pipeline.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import feedparser
import requests
from bs4 import BeautifulSoup

from src.data.preprocessing import clean_news_text, truncate_text
from src.utils.logging import get_logger

logger = get_logger("scraper")

# English RSS feeds (BBC — Reuters blocks scrapers)
BBC_RSS_FEEDS = {
    "Top News": "https://feeds.bbci.co.uk/news/rss.xml",
    "World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Sports": "https://feeds.bbci.co.uk/sport/rss.xml",
}

# German RSS feeds
GERMAN_RSS_FEEDS = {
    "Tagesschau": "https://www.tagesschau.de/index~rss2.xml",
    "Spiegel": "https://www.spiegel.de/schlagzeilen/index.rss",
    "DW German": "https://rss.dw.com/rdf/rss-de-all",
    "ZDF": "https://www.zdf.de/rss/zdf/nachrichten",
}

# Combined feed registry
RSS_FEEDS = {**BBC_RSS_FEEDS, **GERMAN_RSS_FEEDS}

# Legacy alias
REUTERS_RSS_FEEDS = BBC_RSS_FEEDS

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class ScrapeError(Exception):
    """Raised on HTTP, connection, or text extraction failures."""


def fetch_article_text(url: str, timeout: int = 15) -> str:
    """Fetch raw HTML from a URL.

    Args:
        url: Article URL.
        timeout: Request timeout in seconds.

    Returns:
        Raw HTML string.

    Raises:
        ScrapeError: On HTTP or connection errors.
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise ScrapeError(f"Failed to fetch {url}: {e}") from e


def extract_text_from_html(html: str) -> str:
    """Extract article text from HTML.

    Strips script, style, nav, and footer elements. Extracts text
    from <article> tags if present, otherwise falls back to <body> <p> tags.

    Args:
        html: Raw HTML string.

    Returns:
        Extracted plain text.

    Raises:
        ScrapeError: If no meaningful text could be extracted.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Try <article> first
    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
    else:
        # Fall back to body paragraphs
        body = soup.find("body")
        paragraphs = body.find_all("p") if body else soup.find_all("p")

    text = " ".join(p.get_text(strip=True) for p in paragraphs)
    text = " ".join(text.split())  # normalize whitespace

    if not text or len(text) < 50:
        raise ScrapeError("Could not extract meaningful text from HTML")

    return text


def scrape_article(url: str, timeout: int = 15) -> dict:
    """Scrape a single article: fetch, extract, clean, and truncate.

    Args:
        url: Article URL.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: url, title, cleaned_text, scraped_at, word_count.

    Raises:
        ScrapeError: On fetch or extraction failure.
    """
    html = fetch_article_text(url, timeout=timeout)
    raw_text = extract_text_from_html(html)

    # Try to extract title from HTML
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    cleaned = clean_news_text(raw_text)
    cleaned = truncate_text(cleaned, max_words=500)

    logger.info(
        f"Scraped article: {len(cleaned.split())} words from {url}",
        extra={"component": "scraper"},
    )

    return {
        "url": url,
        "title": title,
        "cleaned_text": cleaned,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "word_count": len(cleaned.split()),
    }


def scrape_from_rss(
    feed_url: str,
    max_articles: int = 5,
    delay: float = 2.0,
) -> list[dict]:
    """Scrape full articles from an RSS feed.

    Args:
        feed_url: RSS feed URL.
        max_articles: Maximum number of articles to scrape.
        delay: Seconds to wait between requests.

    Returns:
        List of article dicts from scrape_article().
    """
    feed = feedparser.parse(feed_url)

    if feed.bozo and not feed.entries:
        raise ScrapeError(f"Failed to parse RSS feed: {feed.bozo_exception}")

    articles = []
    for i, entry in enumerate(feed.entries[:max_articles]):
        url = entry.get("link")
        if not url:
            continue

        try:
            article = scrape_article(url)
            # Override title with RSS title if available
            if entry.get("title"):
                article["title"] = entry["title"]
            articles.append(article)
        except ScrapeError as e:
            logger.warning(
                f"Skipping article {url}: {e}",
                extra={"component": "scraper"},
            )
            continue

        # Polite delay between requests
        if i < len(feed.entries[:max_articles]) - 1:
            time.sleep(delay)

    logger.info(
        f"Scraped {len(articles)}/{max_articles} articles from RSS feed",
        extra={"component": "scraper"},
    )

    return articles
