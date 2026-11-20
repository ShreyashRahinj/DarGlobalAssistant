# Deploy on Render — phone checklist

Everything here can be done in a phone browser (Chrome/Safari). No terminal needed.
The app is already on GitHub: `ShreyashRahinj/DarGlobalAssistant`.

## 1. Sign in

- Open `https://render.com` → **Sign in with GitHub**.
- Have your GitHub 2FA / authenticator ready.

## 2. Create the Web Service

1. **New + → Web Service → Build and deploy from a Git repository**.
2. Connect GitHub if prompted, then select **`ShreyashRahinj/DarGlobalAssistant`**.
3. If the page is cramped or a button won't tap, enable the browser's
   **"Request desktop site"** — Render's creation flow behaves better that way.
4. Settings:
   - **Name:** `darglobal-assistant` (this becomes the URL)
   - **Runtime:** Docker (auto-detected — leave the Dockerfile path empty)
   - **Instance Type:** Free
   - **Health Check Path:** `/health`
   - **Region:** closest to you (e.g. Frankfurt or Singapore)

## 3. Add the API key

On the same page, **Environment Variables → Add Environment Variable**:

| Key | Value |
|---|---|
| `OPENROUTER_API_KEY` | your `sk-or-...` key (paste from a password manager) |

Optional variables (only if you want to change defaults):

| Key | Default | Meaning |
|---|---|---|
| `OPENROUTER_DEFAULT_MODEL` | `particle-ai/particle-1:free` | chat model preselected in the UI |
| `EMBED_THREADS` | `1` | ONNX threads for local embeddings (keep at 1 on free instances) |
| `REASONING_MAX_TOKENS` | `256` | cap on hidden "thinking" tokens for reasoning models |

## 4. Deploy

- Tap **Create Web Service**.
- Open the **Logs** tab. First build takes ~10–15 minutes.
- Success looks like:
  - `Loaded index with 303 chunks`
  - `Application startup complete.`

## 5. Verify

1. Open `https://<name>.onrender.com/health`.
   Expected: `"status":"ok"`, `"indexed_chunks":303`, `"openrouter_configured":true`,
   `"sources":["darglobal","wasalt"]`.
2. Open `https://<name>.onrender.com` and check the UI loads and the chat-model
   dropdown preselects `particle-ai/particle-1:free (default)`.
3. Ask a question, e.g. "What services does Wasalt provide?".
   - Use the source filter chips to test DarGlobal / Wasalt / All.
   - A comparison question ("How do DarGlobal and Wasalt differ?") should return
     sources from both sites.

## 6. Timing notes

- **Free instance sleeps after ~15 minutes without traffic.** The next visit
  takes ~30–60 s to wake. The retrieval index is baked into the image, so waking
  never re-embeds the corpus.
- **OpenRouter free quota:** 50 chat requests/day per key, reset at 00:00 UTC
  (05:30 IST). If you see "the AI answering service is temporarily unavailable",
  check the Render logs for `free-models-per-day`. Adding 10 credits to
  OpenRouter raises the limit to 1,000 requests/day.
- Sources are still returned when generation is quota-blocked, so retrieval can
  be demonstrated independently.

## 7. Updating the app later

- **Code change:** commit and push to GitHub. Render redeploys automatically.
- **Data refresh:** run `scraper/scrape.py` (DarGlobal) and
  `scraper/wasalt_browser.py` (Wasalt) locally, commit `data/pages.jsonl`,
  push. Keep `data/index/` out of git — the image rebuilds it at build time.
- **Live URL:** once deployed, edit the **Live URL** line in `README.md` via
  github.com in the browser (pencil icon) and commit.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Build fails on `pip install` | Tap **Manual Deploy → Clear build cache & deploy** |
| Logs show `Could not download model ... Permission denied (os error 13)` or `Error logging to file "/home/appuser/.cache/...` | Fixed in the image (HOME/HF cache dirs). Push the latest Dockerfile, then **Manual Deploy → Deploy latest commit** |
| `openrouter_configured: false` | Re-check the `OPENROUTER_API_KEY` environment variable, then **Manual Deploy → Deploy latest commit** |
| Instance out of memory | Keep `EMBED_THREADS=1`; usage measured ~70 MB idle, ~265 MB after the first query |
| Space/site slow first visit | Free-instance wake-up (~30–60 s), not a bug |
| Render asks for a card | Free web services should not need one; if prompted, pause and use Koyeb's free instance instead |
