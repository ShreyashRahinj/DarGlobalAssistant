"""Robots-aware scraper for DarGlobal and Wasalt.

Run locally (not inside the final image):
    python scraper/scrape.py

Produces data/pages.jsonl with fields:
    url, title, source, text, fetched_at
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
import yaml
from bs4 import BeautifulSoup

# Optional extractors
try:
    import trafilatura  # type: ignore
    HAS_TRAFILATURA = True
except Exception:  # noqa: BLE001
    HAS_TRAFILATURA = False

try:
    from selectolax.parser import HTMLParser  # type: ignore
    HAS_SELECTOLAX = True
except Exception:  # noqa: BLE001
    HAS_SELECTOLAX = False

CONFIG_PATH = Path(__file__).resolve().parent / "sources.yaml"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "pages.jsonl"

HEADERS = {
    "User-Agent": (
        "DarGlobal-Wasalt-Assistant/1.0 "
        "(educational assignment; contact careers@darglobal.co.uk)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}

SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".zip", ".docx", ".xlsx")
SKIP_PATH_FRAGMENTS = (
    "/wp-admin/", "/cart/", "/checkout/", "/my-account/", "/login/",
    "/wp-login/", "/wp-content/uploads/", "/feed/", "/search",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    # Drop fragments, keep query only if meaningful (we keep as-is for now)
    clean = parsed._replace(fragment="")
    return urlunparse(clean)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def robots_allowed(url: str, session: httpx.Client, cache: dict) -> bool:
    domain = urlparse(url).netloc
    if domain not in cache:
        rp = RobotFileParser()
        robots_url = f"https://{domain}/robots.txt"
        try:
            r = session.get(robots_url, timeout=15, headers=HEADERS)
            rp.parse(r.text.splitlines())
        except Exception:  # noqa: BLE001
            # If robots.txt can't be fetched, default to cautious allow
            cache[domain] = None
            return True
        cache[domain] = rp
    rp = cache[domain]
    if rp is None:
        return True
    ua = HEADERS["User-Agent"]
    return rp.can_fetch(ua, url)


def fetch(session: httpx.Client, url: str) -> Optional[str]:
    try:
        r = session.get(url, timeout=20, headers=HEADERS, follow_redirects=True)
        if r.status_code != 200:
            return None
        ct = r.headers.get("content-type", "")
        if "text/html" not in ct:
            return None
        return r.text
    except Exception:  # noqa: BLE001
        return None


def extract_text(html: str, url: str) -> tuple[str, str]:
    """Return (title, main_text). Trafilatura preferred, selectolax fallback."""
    title = ""
    main_text = ""

    if HAS_TRAFILATURA:
        try:
            downloaded = trafilatura.extract(
                html, include_comments=False, include_tables=False,
                no_fallback=False, url=url,
            )
            if downloaded:
                main_text = downloaded.strip()
        except Exception:  # noqa: BLE001
            main_text = ""

    if not main_text and HAS_SELECTOLAX:
        try:
            tree = HTMLParser(html)
            # Remove noise
            for tag in tree.css("script, style, nav, footer, header, .cookie, .cookie-banner, .newsletter, aside"):
                tag.decompose()
            # Try main/article first
            main = tree.css_first("main") or tree.css_first("article") or tree
            main_text = main.text(separator="\n").strip()
        except Exception:  # noqa: BLE001
            main_text = ""

    if not main_text:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.body or soup
            main_text = main.get_text(separator="\n").strip()
        except Exception:  # noqa: BLE001
            main_text = ""

    # Title
    try:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    except Exception:  # noqa: BLE001
        title = ""

    # Collapse whitespace
    main_text = re.sub(r"\n{3,}", "\n\n", main_text)
    main_text = re.sub(r"[ \t]+", " ", main_text)
    return title, main_text


def should_skip(url: str, patterns: List[str]) -> bool:
    low = url.lower()
    if any(ext in low for ext in SKIP_EXTENSIONS):
        return True
    if any(frag in low for frag in SKIP_PATH_FRAGMENTS):
        return True
    return False


def discover_urls(sitemap_url: str, session: httpx.Client) -> List[str]:
    """Fetch sitemap.xml (plain or .xml.gz) and return <loc> URLs."""
    urls: List[str] = []
    try:
        r = session.get(sitemap_url, timeout=30, headers=HEADERS)
        if r.status_code != 200:
            return urls
        content = r.content
        # Handle gzipped sitemaps (.xml.gz) that httpx doesn't auto-decompress.
        # Some hosts serve plain XML under a .gz name; fall back to the raw body.
        if sitemap_url.lower().endswith(".gz") or r.headers.get(
            "content-encoding", ""
        ).lower() == "gzip":
            try:
                content = gzip.decompress(content)
            except Exception:  # noqa: BLE001
                if not content.lstrip().startswith(b"<"):
                    return urls
        text = content.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return urls

    # Simple XML parse of <loc>
    for m in re.finditer(r"<loc>\s*(.*?)\s*</loc>", text, re.DOTALL | re.IGNORECASE):
        u = normalize_url(m.group(1))
        if u.startswith("http"):
            urls.append(u)
    return urls


def crawl(source_key: str, source_cfg: dict, session: httpx.Client,
          robots_cache: dict, seen_hashes: set, out_path: Path,
          max_pages: int) -> int:
    base_url = source_cfg["base_url"]
    sitemap = source_cfg.get("sitemap")
    delay = source_cfg.get("delay_sec", 1.5)
    timeout = source_cfg.get("timeout_sec", 20)
    max_depth = source_cfg.get("max_depth", 2)

    discovered: List[tuple[str, int]] = []  # (url, depth)
    if sitemap:
        sitemap_urls = discover_urls(sitemap, session)
        for u in sitemap_urls:
            discovered.append((u, 0))
    discovered.append((base_url, 0))

    crawled = 0
    visited = set()
    queue = list(discovered)

    with open(out_path, "a", encoding="utf-8") as out:
        while queue and crawled < max_pages:
            url, depth = queue.pop(0)
            norm = normalize_url(url)
            if norm in visited:
                continue
            visited.add(norm)

            if should_skip(norm, source_cfg.get("skip_patterns", [])):
                continue
            if not robots_allowed(norm, session, robots_cache):
                continue

            html = fetch(session, norm)
            if html is None:
                continue

            title, text = extract_text(html, norm)
            if not text or len(text) < 100:
                continue

            ch = content_hash(text)
            if ch in seen_hashes:
                continue
            seen_hashes.add(ch)

            record = {
                "url": norm,
                "title": title,
                "source": source_key,
                "text": text,
                "fetched_at": now_iso(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            crawled += 1
            print(f"[{source_key}] {crawled}/{max_pages} {depth} {norm}")

            if depth < max_depth:
                # Enqueue same-domain links from this page
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        link = urljoin(norm, a["href"])
                        link = normalize_url(link)
                        if urlparse(link).netloc == urlparse(base_url).netloc:
                            if link not in visited and not should_skip(link, source_cfg.get("skip_patterns", [])):
                                queue.append((link, depth + 1))
                except Exception:  # noqa: BLE001
                    pass

            time.sleep(delay)

    return crawled


def main() -> None:
    cfg = load_config()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    robots_cache: dict = {}
    seen_hashes: set = set()

    # If re-running, load existing hashes to avoid duplicates across runs
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        seen_hashes.add(content_hash(obj.get("text", "")))
        except Exception:  # noqa: BLE001
            pass

    total = 0
    with httpx.Client(trust_env=False) as session:
        for source_key, source_cfg in cfg["source_filter"].items():
            if source_cfg.get("enabled", True) is False:
                print(f"[{source_key}] skipped (disabled in sources.yaml)")
                continue
            mp = int(source_cfg.get("max_pages", 150))
            c = crawl(source_key, source_cfg, session, robots_cache, seen_hashes, OUTPUT_PATH, mp)
            total += c
            print(f"[{source_key}] done: {c} new pages")

    print(f"Total new pages: {total}. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
