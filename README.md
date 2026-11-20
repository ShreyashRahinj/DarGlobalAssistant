# DarGlobal & Wasalt Assistant

A grounded, hybrid-retrieval RAG chatbot over public pages from **DarGlobal** (real estate developer) and **Wasalt** (Saudi proptech). Built as a free, deployed, stand-out technical assessment.

**Live URL:** *(set after you deploy to Render)*  
**Repo:** *(your GitHub public link)*

---

## What this does

- Answers questions about DarGlobal and Wasalt **only from scraped public pages**.
- Every answer includes **citations [1], [2]…** with source URLs and snippets.
- Hybrid retrieval: **BM25 + embeddings (bge-small-en-v1.5)**, merged with **reciprocal rank fusion (RRF)**.
- LLM calls go through **OpenRouter free models** with a **fallback chain**, retries, backoff, and an in-memory response cache.
- **Model pickers in the UI:** pick a specific free chat model (the most reliable one is preselected; no auto-start), pick the local or an OpenRouter embedding model, and add **any other model ID** by typing it. Custom IDs are remembered in the browser.
- Source filter: All / DarGlobal / Wasalt. Handles comparison questions ("how do they differ?").
- Refuses out-of-scope questions and never invents prices or figures.
- Minimal vanilla-JS chat UI, health endpoint, and Docker packaging.

---

## Architecture

```
scrape.py (httpx + trafilatura/selectolax, robots-aware)
   -> data/pages.jsonl  (url, title, source, text, fetched_at)
   -> chunk (400 tokens, 60 overlap, title prepended for retrieval)

wasalt_browser.py (Playwright, Cloudflare-safe; see below)
   -> same pages.jsonl format, source="wasalt"

FastAPI app
   /chat                    : retrieve (BM25 top-20 + embed top-20, RRF,
                              max 2 chunks/page, legal pages only when asked,
                              balanced per-source for comparison questions)
                              -> threshold -> prompt -> OpenRouter (fallback chain)
                              -> answer + citations + model used
   /health                  : status, chunk count, OpenRouter configured?
   /sources                 : available source filters
   /models                  : static catalog of free chat + embedding models
   /embedding-models/warm   : start building a corpus index for a model
   /embedding-models/status : indexing progress for a model
   /                        : static single-page chat UI (vanilla HTML/JS)

data/index/ : BM25 (shared) + one cached embedding matrix per embedding model
Dockerfile  -> Render free web service (port 7860); also runs as an HF PRO Docker Space

eval/run_eval.py -> retrieval hit-rate report (README)
```

```
User
  |
[FastAPI /chat]
  |
Retriever (BM25 + bge-small embeddings + RRF)
  |
data/pages.jsonl  (snapshot of public pages)
  |
OpenRouter (free models, fallback chain)
```

---

## Repo layout

```
darglobal-wasalt-assistant/
  app/          main.py   rag.py   llm.py   embedding_backends.py   config.py   static/index.html
  scraper/      scrape.py  wasalt_browser.py  sources.yaml
  data/         pages.jsonl  index/          <- snapshot ships in repo; refresh via scraper/
  eval/         questions.jsonl  run_eval.py  results.md
  Dockerfile  requirements.txt  README.md  .dockerignore  .env.example
```

---

## One-command run (locally)

```bash
git clone <your-repo>
cd darglobal-wasalt-assistant

# 1. Copy env and add your OpenRouter key
cp .env.example .env
# edit .env: OPENROUTER_API_KEY=sk-or-...

# 2. Build and run
docker build -t darglobal-wasalt-assistant .
docker run --rm -p 7860:7860 --env-file .env darglobal-wasalt-assistant
```

Open `http://localhost:7860`. Health: `http://localhost:7860/health`.

Without Docker (uses the same command the container runs):

```bash
python -m venv env
env/Scripts/activate          # Windows; use source env/bin/activate on macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Locally, `app/config.py` loads `.env` automatically (via `python-dotenv`), so no shell export is needed.

---

## Choosing models in the UI

**Chat model.** The most reliable free model found during testing (`particle-ai/particle-1:free`) is preselected, so the app never starts in an "auto" mode. If the selected model fails (429/5xx/timeout/empty), the app still falls back through the rest of the chain, so a bad choice never takes the assistant down. The answer footer shows which model actually answered. Change the default with `OPENROUTER_DEFAULT_MODEL`.

Reasoning models get `reasoning.max_tokens` (default 256 via `REASONING_MAX_TOKENS`) so their hidden thinking cannot consume the whole answer budget; set it to `0` to stop sending that field.

**Embedding model.** The local `bge-small-en-v1.5` model is the default: it is offline, fast and has no rate limits. Picking one of the OpenRouter embedding models (or typing any other embedding ID) builds a corpus index for that model on first use. The UI shows indexing progress and blocks sending until the index is ready. BM25 stays shared; only the embedding matrix is rebuilt, and it is cached in memory and under `data/index/`.

**Adding new model IDs.** The catalogs in `config.py` are static (no live discovery), so when the free list changes you have two options:
- Type the new ID in *+ Add custom model…* on the page. It is saved in your browser's `localStorage` and sent per request — no server change needed.
- Add it to `DEFAULT_FREE_MODELS` / `OPENROUTER_EMBED_MODELS` in `app/config.py`, or set the `OPENROUTER_MODELS` / `OPENROUTER_EMBED_MODELS` environment variables, to make it visible to everyone.

---

## How to refresh data (you do this)

The repo ships a snapshot in `data/pages.jsonl` (DarGlobal + Wasalt). To refresh it, run the scrapers; both append to the same file and skip pages they have already saved (content hash / URL).

```bash
# 1. First check (required):
#    - https://darglobal.co.uk/robots.txt  and  https://wasalt.sa/llms.txt
#    - https://darglobal.co.uk/sitemap.xml

# 2. Install scraper deps
pip install -r requirements.txt        # pyyaml, beautifulsoup4, trafilatura, httpx
pip install playwright && python -m playwright install chromium   # Wasalt only

# 3a. DarGlobal (plain HTTP scraper, robots-aware, 1.5 s delay)
#     Wasalt is marked `enabled: false` in sources.yaml because HTTP gets a
#     Cloudflare challenge; it is collected by the browser scraper instead.
python scraper/scrape.py

# 3b. Wasalt (Cloudflare-protected: needs a real browser)
python scraper/wasalt_browser.py --max-pages 30 --delay 1.5

# 4. Delete the cached index (it detects staleness, but this is deterministic)
rm -rf data/index

# 5. Restart the app; commit data/pages.jsonl and rebuild/redeploy
```

**Why Wasalt needs a different scraper.** `wasalt.sa` sits behind Cloudflare bot protection, so plain `httpx` gets a JS challenge instead of HTML. `scraper/wasalt_browser.py` drives headful Chromium with Playwright (run it on a machine with a display, or pass `--headless`), waits out the interstitial, visits the landing/corporate/legal/listing pages plus a capped set of property-detail and blog links, and stores only public page text.

**Wasalt usage terms.** `wasalt.sa/llms.txt` permits non-commercial LLM use of public listing content with moderate crawl rates and attribution, and asks that phone/WhatsApp/contact details not be collected at scale and search results not be replicated. The collector therefore: caps pages, uses a 1.5 s delay, strips URLs/emails/Saudi phone numbers from stored text, follows only public pages, and the UI cites "Source: Wasalt.sa". Listing prices/availability come from the snapshot and may have changed; the prompt tells the model to say so when quoting them.

Scraping ethics (both scrapers): respects `robots.txt`, 1.5 s delay, same-domain only, depth 2, capped page count, public pages only, no login-gated or personal data, descriptive user-agent with contact.

---

## Deploy (free): Render

Hugging Face now requires a paid plan (PRO, $9/mo) to create **Docker or Gradio** Spaces; free accounts only get Static Spaces, which cannot run the FastAPI backend. Render's free Docker web service runs this repo unchanged.

1. Push this repo to GitHub (public or private).
2. On https://render.com: **New + → Web Service → Build and deploy from a Git repository**, pick the repo. Render detects the `Dockerfile`.
3. Instance Type: **Free**. Health Check Path: `/health`.
4. **Environment**: add `OPENROUTER_API_KEY` (use the secret/secret-file field). Optional: `OPENROUTER_DEFAULT_MODEL`, `REASONING_MAX_TOKENS`.
5. Create the service. First build ~10–15 min; logs end with `Loaded index with N chunks`.
6. Live URL: `https://<service-name>.onrender.com`. Try `/health`, then ask a question.

Free-tier behavior: 512 MB RAM / 0.1 CPU; the service sleeps after ~15 min without traffic (first visit afterwards takes ~30–60 s); disk is ephemeral, but the retrieval index is baked into the image during the build, so waking never re-embeds the corpus.

**Alternative:** the same Dockerfile works as a Hugging Face Docker Space on a PRO account (16 GB RAM, no sleep), if you prefer that over Render.

---

## Design decisions (read like an engineering note)

| Decision | Why |
|---|---|
| Local embeddings (fastembed bge-small) | Free, offline, no API key, no rate limits; ~100–200 MB image; stays the default |
| Per-model embedding index cache | Switching to an OpenRouter embedding model re-embeds just the corpus chunks, reuses BM25, and caches a normalised matrix per model |
| Custom model IDs in `localStorage` | The deployed app stays stateless; users can adopt new free IDs without a redeploy |
| Hybrid BM25 + embeddings + RRF | BM25 catches exact terms; embeddings catch meaning; RRF merges robustly |
| Fallback chain across free OpenRouter models | Free models hit 429/5xx often; trying the next model keeps the app alive |
| `reasoning.max_tokens=256` cap (`REASONING_MAX_TOKENS`) | Free reasoning models were spending the whole output budget on hidden thinking and returning empty answers |
| Title prepended to chunk text | Listing pages share boilerplate, so the title (city/type) is what makes "villas for rent in Riyadh" retrievable |
| Max 2 chunks per URL in top-k | One long page used to fill all six slots; the cap widens coverage |
| Legal pages only for legal questions | Privacy/terms pages match words like "terms" and used to outrank topical pages |
| Balanced per-source retrieval for comparisons | "How do DarGlobal and Wasalt differ" otherwise returned only one company |
| Browser-based Wasalt collector | Cloudflare serves a JS challenge to plain HTTP clients; Playwright gets real HTML |
| In-memory LRU cache on normalised question | Repeat demo questions never re-hit the API |
| Bake `pages.jsonl` into image | Deployed app never depends on live scraping or network to the two sites |
| No `torch` | Keeps image small; fastembed uses ONNX |
| Threshold on embedding score | Weak queries return "I don't have that" instead of hallucinating |

---

## Limitations / honesty

- Answers are based on a **snapshot of public pages** (173 pages / 303 chunks at last build), may be out of date. "Data last refreshed" shown in UI. Listing prices/availability change; the assistant says so when quoting them.
- **OpenRouter free tier is capped at 50 chat requests/day per key** (1,000/day after adding 10 credits). The fallback chain burns one request per model attempt, so a day of testing can exhaust it. When it does, `/chat` still returns retrieval-backed sources with a friendly "temporarily unavailable" answer, and the log shows `free-models-per-day` with the reset time.
- Free models can be slow or unavailable; the app falls back across several and shows a friendly message if all are exhausted.
- OpenRouter embedding models have to index the corpus on first use (about a minute for ~300 chunks, longer on free limits). The local model avoids this entirely and remains the default; OpenRouter also runs content-safety screening on embedding inputs (handled with split-on-422, but a page can still be replaced with a placeholder vector).
- The similarity threshold is tuned for `bge-small`; API embedding models may need `SIMILARITY_THRESHOLD` adjusted.
- The model list is static; it does not auto-discover today's free models. Custom IDs (UI) or editing `config.py` cover changes.
- **Not production-ready.** Next steps: scheduled refresh, reranker, streaming, auth, observability.
- Embedding model (`BAAI/bge-small-en-v1.5`) is downloaded at image build (~100–200 MB).

---

## Eval results

Run `python eval/run_eval.py` locally after building the index. It reports retrieval hit rate over 20 held-out questions (expected source: darglobal / wasalt / both).

See `eval/results.md` for the latest numbers.

Latest local run: **20/20 (100%)** on the merged DarGlobal + Wasalt snapshot. This is a retrieval hit-rate benchmark only (was the expected source in the top-k chunks), not an answer-accuracy score; groundedness is checked manually against `/chat`.

---

## Email to send

**Holding (early):** see `DarGlobal_Assignment_Plan.md` section 9 template — send now with location, current package, expected package, notice period, and "URL follows today."

**Final (today):** same details + live URL, GitHub repo, 3-line summary, note that first load takes ~a minute, and 3 example questions.

---

## Security

- `OPENROUTER_API_KEY` is never committed. Set it as a Space secret / env var. `.env.example` is committed; `.env` is not.
- Scraped page text is treated as untrusted data; the system prompt instructs the model to ignore instructions hidden inside it.
- No personal data is scraped or stored.
