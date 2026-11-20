"""Embedding backends: local fastembed (default) and OpenRouter's API.

Both return a float32 matrix with one row per input text. Normalisation is done
by the caller (Retriever) so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Sequence

import numpy as np

logger = logging.getLogger("darglobal.embeddings")

ProgressFn = Callable[[int, int], None]


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce vectors."""


class LocalEmbedder:
    """Wraps fastembed (ONNX, offline, no API key)."""

    def __init__(self, model_name: str, threads: int | None = None):
        self.model_name = model_name
        self.threads = threads
        self._model = None

    def _lazy_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            from app.config import settings

            threads = self.threads if self.threads is not None else settings.EMBED_THREADS
            logger.info("Loading local embedding model %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name, threads=threads)
        return self._model

    def embed(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        model = self._lazy_model()
        vecs = [np.asarray(v, dtype="float32") for v in model.embed(list(texts), batch_size=batch_size)]
        if not vecs:
            raise EmbeddingError("No texts to embed")
        if progress:
            progress(len(texts), len(texts))
        return np.vstack(vecs)


class OpenRouterEmbedder:
    """Calls OpenRouter's OpenAI-compatible /embeddings endpoint.

    Batches inputs, retries on 429/5xx/timeouts, and splits a batch in half if
    the provider rejects it for being too large.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            if not self.api_key:
                raise EmbeddingError(
                    "OPENROUTER_API_KEY is not set, so OpenRouter embedding models "
                    "cannot be used. Use the local embedding model instead."
                )
            from openai import OpenAI

            import httpx

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
                # Same proxy-env caveat as the chat client (NO_PROXY with "[::1]").
                http_client=httpx.Client(trust_env=False),
            )
        return self._client

    def embed(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        client = self._lazy_client()
        out: List[List[float]] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            out.extend(self._embed_batch(client, batch))
            if progress:
                progress(min(start + batch_size, total), total)
        if not out:
            raise EmbeddingError("No texts to embed")
        return np.vstack([np.asarray(v, dtype="float32") for v in out])

    # ------------------------------------------------------------------ #
    def _embed_batch(self, client, batch: List[str], allow_placeholder: bool = True) -> List[List[float]]:
        try:
            resp = self._call_with_retry(client, batch)
        except EmbeddingError as exc:
            message = str(exc)
            if len(batch) > 1 and self._should_split(message):
                mid = len(batch) // 2
                logger.warning(
                    "Batch of %d rejected (%s); splitting and retrying",
                    len(batch),
                    message[:80],
                )
                return self._embed_batch(client, batch[:mid]) + self._embed_batch(
                    client, batch[mid:]
                )
            if (
                len(batch) == 1
                and allow_placeholder
                and self._is_content_safety(message)
            ):
                # A single chunk tripped the provider's screening. Keep row
                # alignment by embedding a neutral placeholder; BM25 still
                # matches the chunk, only its vector signal is lost.
                logger.warning("Input screened out; substituting a neutral vector")
                return self._embed_batch(client, ["[content unavailable]"], allow_placeholder=False)
            raise

        data = sorted(getattr(resp, "data", []), key=lambda d: getattr(d, "index", 0))
        vectors = [list(d.embedding) for d in data]
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"Provider returned {len(vectors)} vectors for {len(batch)} inputs"
            )
        return vectors

    def _call_with_retry(self, client, batch: List[str]):
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return client.embeddings.create(model=self.model_id, input=batch)
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = str(exc)
                lower = msg.lower()
                retryable = (
                    "429" in msg
                    or "rate limit" in lower
                    or any(code in msg for code in ("500", "502", "503", "504"))
                    or "timeout" in lower
                    or "timed out" in lower
                )
                if attempt < self.max_retries and retryable:
                    wait = (attempt + 1) * 2
                    logger.warning(
                        "Embedding model %s error (%s); retry %d in %ds",
                        self.model_id,
                        msg[:160],
                        attempt + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise EmbeddingError(msg) from exc
        raise EmbeddingError(str(last) if last else "Unknown embedding error")

    @staticmethod
    def _should_split(message: str) -> bool:
        lower = message.lower()
        return any(
            marker in lower
            for marker in (
                "too many",
                "too large",
                "maximum context",
                "context length",
                "413",
                "422",
                "token limit",
                "input rejected",
            )
        )

    @staticmethod
    def _is_content_safety(message: str) -> bool:
        lower = message.lower()
        return "content safety" in lower or "screening" in lower


def make_embedder(
    model_id: str,
    local_model_id: str,
    api_key: str,
    base_url: str,
    timeout: float = 120.0,
    max_retries: int = 3,
):
    """Return a LocalEmbedder for the configured local model, else an API embedder."""
    if model_id == local_model_id:
        return LocalEmbedder(model_id)
    return OpenRouterEmbedder(
        model_id, api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
    )
