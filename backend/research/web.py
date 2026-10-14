"""Public web search and page fetch for the deep pass."""
from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from backend.config import get_settings
from backend.schemas import FetchedPage, WebHit

# Set by the research pipeline so fetches cache under runs/<job_id>/pages/.
CURRENT_RUN_DIR: ContextVar[Path | None] = ContextVar("plan4me_run_dir", default=None)

BLOCKED_HOST_PARTS = (
    "libgen",
    "annas-archive",
    "z-lib",
    "zlib",
    "pdfdrive",
    "sci-hub",
)

_BANNED_QUERY_BITS = ("pdf", "filetype:pdf", "libgen", "download", "free book")

USER_AGENT = "plan4me-research"


def url_allowed(url: str) -> bool:
    """False for blocked hosts and for URLs whose path ends in .pdf."""
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if any(part in host for part in BLOCKED_HOST_PARTS):
        return False
    if (parsed.path or "").lower().endswith(".pdf"):
        return False
    return True


def rejection_reason(url: str) -> str | None:
    """Why a hit must not be fetched, or None when a fetch is allowed.

    YouTube is skipped here even though `url_allowed` permits it: captions
    were already the normal pass.
    """
    if not url or not isinstance(url, str):
        return "blocked host"
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    if any(part in host for part in BLOCKED_HOST_PARTS):
        return "blocked host"
    if path.endswith(".pdf"):
        return "pdf not fetched"
    if host == "youtu.be" or host.endswith(".youtu.be") or "youtube.com" in host:
        return "youtube skipped"
    if not host:
        return "blocked host"
    return None


def query_allowed(query: str) -> bool:
    lowered = query.lower()
    return not any(bit in lowered for bit in _BANNED_QUERY_BITS)


def search_web(query: str, max_results: int) -> list[WebHit]:
    """Search the public web. Provider comes from WEB_SEARCH_PROVIDER."""
    if not query_allowed(query):
        return []
    settings = get_settings()
    provider = (settings.web_search_provider or "duckduckgo").strip().lower()
    if provider == "tavily":
        return _tavily_search(query, max_results, settings.tavily_api_key)
    return _duckduckgo_search(query, max_results)


def fetch_page(url: str) -> FetchedPage:
    """Download a page and extract its main text. PDFs and blocked hosts are not fetched."""
    if not url_allowed(url):
        reason = "pdf not fetched"
        parsed = urlparse(url.strip()) if url else None
        if parsed is not None and not (parsed.path or "").lower().endswith(".pdf"):
            reason = "blocked host"
        return FetchedPage(url=url, title="", text="", word_count=0, error=reason)

    cached = _read_cache(url)
    if cached is not None:
        return cached

    import httpx
    import trafilatura

    try:
        with httpx.Client(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = client.get(url)
            final_url = str(response.url)
            content_type = response.headers.get("content-type", "")
            if not url_allowed(final_url) or "application/pdf" in content_type.lower():
                page = FetchedPage(
                    url=url, title="", text="", word_count=0, error="pdf not fetched"
                )
                _write_cache(url, page)
                return page
            html = response.text
        text = trafilatura.extract(html) or ""
        title = ""
        meta = trafilatura.extract_metadata(html)
        if meta is not None and getattr(meta, "title", None):
            title = meta.title or ""
        page = FetchedPage(
            url=url,
            title=title,
            text=text,
            word_count=len(text.split()),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - a failed fetch is a rejected page
        page = FetchedPage(
            url=url, title="", text="", word_count=0, error=str(exc)[:300]
        )
    _write_cache(url, page)
    return page


def _duckduckgo_search(query: str, max_results: int) -> list[WebHit]:
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=max_results) or []
    hits: list[WebHit] = []
    for row in rows:
        hits.append(
            WebHit(
                title=row.get("title") or "",
                url=row.get("href") or row.get("url") or "",
                snippet=row.get("body") or row.get("snippet") or "",
            )
        )
    return hits


def _tavily_search(query: str, max_results: int, api_key: str) -> list[WebHit]:
    if not api_key.strip():
        raise RuntimeError("TAVILY_API_KEY is empty")
    import httpx

    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        },
        timeout=20.0,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    hits: list[WebHit] = []
    for item in payload.get("results", []):
        hits.append(
            WebHit(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("content") or "",
            )
        )
    return hits


def _cache_file(url: str) -> Path | None:
    run_dir = CURRENT_RUN_DIR.get()
    if run_dir is None:
        return None
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return run_dir / "pages" / f"{digest}.json"


def _read_cache(url: str) -> FetchedPage | None:
    path = _cache_file(url)
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return FetchedPage.model_validate(data)


def _write_cache(url: str, page: FetchedPage) -> None:
    path = _cache_file(url)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = page.model_dump()
    payload["fetched_at"] = datetime.now(timezone.utc).date().isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
