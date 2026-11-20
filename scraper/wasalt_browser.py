"""Collect public Wasalt pages through a real browser (Cloudflare blocks plain HTTP).

wasalt.sa answers non-browser clients with a Cloudflare challenge, so the httpx
scraper cannot read it. This collector drives Chromium via Playwright instead.

Policy: follows https://wasalt.sa/llms.txt — non-commercial LLM use, public page
content only, moderate crawl rate (1.5 s delay), no contact-detail harvesting,
and attribution ("Source: Wasalt.sa") is shown next to every citation in the app.

Run locally (not inside the final image):
    pip install playwright
    python -m playwright install chromium
    python scraper/wasalt_browser.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "pages.jsonl"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

SEEDS = [
    "https://wasalt.sa/en",
    "https://wasalt.sa/en/corporate",
    "https://wasalt.sa/en/s/privacy-policy",
    "https://wasalt.sa/en/s/terms-of-use",
    "https://wasalt.sa/en/s/terms-of-advertising",
    "https://wasalt.sa/en/properties-for-sale-in-riyadh",
    "https://wasalt.sa/en/properties-for-sale-in-jeddah",
    "https://wasalt.sa/en/properties-for-sale-in-makkah",
    "https://wasalt.sa/en/properties-for-sale-in-madinah",
    "https://wasalt.sa/en/properties-for-sale-in-dammam",
    "https://wasalt.sa/en/properties-for-sale-in-al-khobar",
    "https://wasalt.sa/en/apartments-for-sale-in-riyadh",
    "https://wasalt.sa/en/villas-for-sale-in-riyadh",
    "https://wasalt.sa/en/lands-for-sale-in-riyadh",
    "https://wasalt.sa/en/lands-for-sale-in-jeddah",
    "https://wasalt.sa/en/properties-for-rent-in-riyadh",
    "https://wasalt.sa/en/properties-for-rent-in-jeddah",
    "https://wasalt.sa/en/apartments-for-rent-in-riyadh",
    "https://wasalt.sa/en/villas-for-rent-in-riyadh",
]

BLOG_URL = "https://blog.wasalt.sa/en/"

BLOCK_TAGS = (
    "script", "style", "noscript", "svg", "iframe", "nav", "footer",
    "header", "aside", "form", "button",
)
LINE_TAGS = (
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
    "section", "article", "ul", "ol", "table",
)
PHONE_RE = re.compile(r"(?:\+?966|00966|0)[\s\-()]?\d{1,2}[\s\-()]?\d{3}[\s\-()]?\d{3,4}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def sanitize(text: str) -> str:
    """Drop contact details and raw links; keep the readable public content."""
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def extract(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(BLOCK_TAGS):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for el in soup.find_all(LINE_TAGS):
        el.append("\n")
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return title, sanitize(main.get_text())


def load_seen() -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    hashes: set[str] = set()
    if not OUT_PATH.exists():
        return urls, hashes
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            urls.add(str(obj.get("url", "")).rstrip("/"))
            text = obj.get("text", "")
            if text:
                hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return urls, hashes


def wait_ready(page, timeout_sec: float = 25.0) -> bool:
    """Wait out the Cloudflare interstitial and the client-side render."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            title = page.title() or ""
            length = page.evaluate("document.body ? document.body.innerText.length : 0")
        except Exception:  # noqa: BLE001
            page.wait_for_timeout(500)
            continue
        if "just a moment" not in title.lower() and length > 400:
            return True
        page.wait_for_timeout(700)
    return False


def absolute_links(page, pattern: re.Pattern[str], limit: int) -> list[str]:
    try:
        hrefs = page.evaluate(
            "() => Array.from(document.querySelectorAll('a[href]'), (a) => a.href)"
        )
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for href in hrefs:
        if pattern.search(href) and href not in out:
            out.append(href)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public Wasalt pages via Chromium")
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--headless", action="store_true", help="usually worsens Cloudflare checks")
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen_urls, seen_hashes = load_seen()

    stored = 0
    skipped = 0
    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            locale="en-US",
            viewport={"width": 1366, "height": 900},
            user_agent=USER_AGENT,
        )
        page = context.new_page()

        try:
            page.goto("https://wasalt.sa/en", wait_until="domcontentloaded", timeout=60000)
            if not wait_ready(page):
                print("Could not get past the Cloudflare check; aborting.", file=sys.stderr)
                browser.close()
                sys.exit(1)

            queue = list(SEEDS)
            pdp_candidates: list[str] = []
            blog_candidates: list[str] = []
            blog_visited = False
            index = 0

            while index < len(queue) and stored < args.max_pages:
                url = queue[index]
                index += 1
                if url.rstrip("/") in seen_urls:
                    skipped += 1
                    continue

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    if not wait_ready(page):
                        errors.append(f"{url}: challenge/timeout")
                        continue
                    page.wait_for_timeout(1200)
                    html = page.content()
                except PlaywrightTimeout:
                    errors.append(f"{url}: timeout")
                    continue
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{url}: {str(exc)[:100]}")
                    continue

                title, text = extract(html)
                if len(text) < 150:
                    errors.append(f"{url}: too little text ({len(text)})")
                    continue

                text = text[:3500]
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    skipped += 1
                    continue

                record = {
                    "url": url,
                    "title": title,
                    "source": "wasalt",
                    "text": text,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(OUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_urls.add(url.rstrip("/"))
                seen_hashes.add(content_hash)
                stored += 1
                print(f"[wasalt] {stored}/{args.max_pages} {len(text):5d} chars {url}")

                if "/properties-" in url or "/apartments-" in url or "/villas-" in url or "/lands-" in url:
                    new_pdps = absolute_links(page, re.compile(r"/en/property/"), 12)
                    for link in new_pdps:
                        if link not in pdp_candidates:
                            pdp_candidates.append(link)

                if not blog_visited:
                    blog_visited = True
                    try:
                        page.goto(BLOG_URL, wait_until="domcontentloaded", timeout=60000)
                        if wait_ready(page):
                            page.wait_for_timeout(1200)
                            blog_candidates = absolute_links(
                                page, re.compile(r"blog\.wasalt\.sa/en/[^/?#]+/?$"), 6
                            )
                    except Exception:  # noqa: BLE001
                        pass

                time.sleep(args.delay)

            for pdp in pdp_candidates[:4]:
                if stored >= args.max_pages:
                    break
                if pdp.rstrip("/") in seen_urls:
                    continue
                try:
                    page.goto(pdp, wait_until="domcontentloaded", timeout=60000)
                    if not wait_ready(page):
                        errors.append(f"{pdp}: challenge/timeout")
                        continue
                    page.wait_for_timeout(1200)
                    title, text = extract(page.content())
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{pdp}: {str(exc)[:100]}")
                    continue
                if len(text) < 150:
                    continue
                text = text[:3500]
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    continue
                record = {
                    "url": pdp,
                    "title": title,
                    "source": "wasalt",
                    "text": text,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(OUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_urls.add(pdp.rstrip("/"))
                seen_hashes.add(content_hash)
                stored += 1
                print(f"[wasalt] {stored}/{args.max_pages} {len(text):5d} chars {pdp}")
                time.sleep(args.delay)

            for article in blog_candidates[:3]:
                if stored >= args.max_pages:
                    break
                if article.rstrip("/") in seen_urls:
                    continue
                try:
                    page.goto(article, wait_until="domcontentloaded", timeout=60000)
                    if not wait_ready(page):
                        errors.append(f"{article}: challenge/timeout")
                        continue
                    page.wait_for_timeout(1200)
                    title, text = extract(page.content())
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{article}: {str(exc)[:100]}")
                    continue
                if len(text) < 150:
                    continue
                text = text[:3500]
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    continue
                record = {
                    "url": article,
                    "title": title,
                    "source": "wasalt",
                    "text": text,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(OUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen_urls.add(article.rstrip("/"))
                seen_hashes.add(content_hash)
                stored += 1
                print(f"[wasalt] {stored}/{args.max_pages} {len(text):5d} chars {article}")
                time.sleep(args.delay)
        finally:
            browser.close()

    print(f"Stored {stored} new Wasalt pages ({skipped} skipped, {len(errors)} errors).")
    for err in errors[:10]:
        print(f"  - {err}")


if __name__ == "__main__":
    main()
