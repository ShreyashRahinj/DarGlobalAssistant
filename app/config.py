"""Configuration for the DarGlobal / Wasalt assistant."""

import os
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # env vars still work (Docker / HF Spaces secrets)
    pass


class Settings:
    def __init__(self):
        # OpenRouter. Embeddings default to local fastembed; API embeddings are optional.
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        self.OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Retrieval
        self.TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "20"))
        self.TOP_K_EMBED = int(os.getenv("TOP_K_EMBED", "20"))
        self.FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "6"))
        self.RRF_K = int(os.getenv("RRF_K", "60"))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))

        # Follow-up context window
        self.FOLLOW_UP_TURNS = int(os.getenv("FOLLOW_UP_TURNS", "3"))

        # Free reasoning models can spend the whole completion budget "thinking"
        # and return empty content. Cap reasoning tokens per call (0 = don't send).
        self.REASONING_MAX_TOKENS = int(os.getenv("REASONING_MAX_TOKENS", "256"))

        # Cache
        self.CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "512"))

        # ONNX intra-op threads for the local embedding model. One thread keeps
        # RSS low on free 512 MB instances; raise it on bigger machines.
        self.EMBED_THREADS = int(os.getenv("EMBED_THREADS", "1"))

        # OpenRouter embeddings (optional alternative to the local model).
        self.EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
        self.EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
        self.EMBED_TIMEOUT_SEC = float(os.getenv("EMBED_TIMEOUT_SEC", "120"))
        self.MAX_EMBEDDING_STATES = int(os.getenv("MAX_EMBEDDING_STATES", "12"))

        # Scraping ethics (documented in README)
        self.REQUEST_DELAY_SEC = float(os.getenv("REQUEST_DELAY_SEC", "1.5"))
        self.USER_AGENT = os.getenv(
            "USER_AGENT",
            "DarGlobal-Wasalt-Assistant/1.0 "
            "(educational assignment; contact careers@darglobal.co.uk)",
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Module-level aliases (kept so other modules can `from config import X`).
TOP_K_BM25 = settings.TOP_K_BM25
TOP_K_EMBED = settings.TOP_K_EMBED
FINAL_TOP_K = settings.FINAL_TOP_K
RRF_K = settings.RRF_K
SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD
FOLLOW_UP_TURNS = settings.FOLLOW_UP_TURNS
REASONING_MAX_TOKENS = settings.REASONING_MAX_TOKENS
CACHE_MAX_SIZE = settings.CACHE_MAX_SIZE
EMBED_THREADS = settings.EMBED_THREADS
REQUEST_DELAY_SEC = settings.REQUEST_DELAY_SEC
USER_AGENT = settings.USER_AGENT
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = settings.OPENROUTER_BASE_URL


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or default


# Model preference order (IDs ending in :free). Discovered from OpenRouter free list.
# Override via env OPENROUTER_MODELS, or per-request via the UI custom model input.
DEFAULT_FREE_MODELS = _env_list(
    "OPENROUTER_MODELS",
    [
        "particle-ai/particle-1:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "nvidia/nemotron-3.5-lightning:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "inclusionai/ling-3.2-flash:free",
        "baidu/ernie-5.5-flash-preview:free",
    ],
)

# Preselected chat model in the UI and used for API calls without an explicit
# model. Still tried first in the fallback chain, so a failure rolls to others.
DEFAULT_CHAT_MODEL = (
    os.getenv("OPENROUTER_DEFAULT_MODEL", "").strip() or DEFAULT_FREE_MODELS[0]
)

# Free OpenRouter embedding models (observed today). Kept as alternatives; the app
# defaults to local fastembed (offline, no API key, no rate limits) for embeddings.
OPENROUTER_EMBED_MODELS = _env_list(
    "OPENROUTER_EMBED_MODELS",
    [
        "nvidia/nemotron-3.5-embed-1b:free",
        "nvidia/nemotron-3.5-embed-8b:free",
    ],
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60

SOURCES = ["darglobal", "wasalt"]

# The local fastembed model is the default; OpenRouter embedding IDs are optional
# alternatives that re-index the corpus on first use (see app/rag.py).
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL


def pretty_model_name(model_id: str) -> str:
    """Human-friendly label for a model ID, e.g. 'google/gemma-4-31b-it:free'."""
    base = model_id.split("/")[-1]
    base = base.replace(":free", "")
    return base.replace("-", " ").replace("_", " ").strip()
