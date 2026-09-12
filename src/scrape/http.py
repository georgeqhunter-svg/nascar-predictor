"""Polite HTTP client with disk caching and retries.

All scrapers should go through `fetch()`. It:
  * caches raw responses to data/raw/ keyed by URL,
  * throttles to ~1 req/sec across the process,
  * retries transient failures with exponential backoff.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import cloudscraper  # optional; only needed if scraping the Cloudflare-protected racing_reference
except ImportError:
    cloudscraper = None  # type: ignore

log = logging.getLogger(__name__)

# Repo root is two levels up from this file (src/scrape/http.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw"

# Realistic Chrome UA — Cloudflare-protected sites often block obvious bot UAs
# outright before even trying the JS challenge.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

# Cloudscraper handles Cloudflare's JS challenge and rotates plausible browser
# TLS fingerprints. One shared session so we amortize challenge cookies.
# Falls back to plain requests if cloudscraper isn't installed.
if cloudscraper is not None:
    _SCRAPER = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )
else:
    _SCRAPER = requests.Session()
    _SCRAPER.headers.update({"User-Agent": USER_AGENT})

_MIN_INTERVAL_SEC = 1.0
_last_request_at: float = 0.0


def _url_to_path(url: str, subdir: str) -> Path:
    """Deterministic cache path for a URL under data/raw/<subdir>/."""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    # Keep a human-readable hint alongside the hash.
    tail = url.rstrip("/").rsplit("/", 1)[-1] or "index"
    safe_tail = "".join(c if c.isalnum() or c in "-_." else "_" for c in tail)[:80]
    out = RAW_ROOT / subdir / f"{safe_tail}__{h}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _throttle() -> None:
    global _last_request_at
    now = time.monotonic()
    wait = _MIN_INTERVAL_SEC - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _get(url: str) -> str:
    _throttle()
    log.debug("GET %s", url)
    resp = _SCRAPER.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch(url: str, *, subdir: str, force: bool = False) -> str:
    """Return the response body for `url`, using disk cache under data/raw/<subdir>/."""
    path = _url_to_path(url, subdir)
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")
    body = _get(url)
    path.write_text(body, encoding="utf-8")
    return body


def cached_path(url: str, subdir: str) -> Optional[Path]:
    """Return the on-disk cache path for `url` if it exists, else None."""
    p = _url_to_path(url, subdir)
    return p if p.exists() else None
