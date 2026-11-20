"""LLM client for OpenRouter with fallback chain, retries, and response cache."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Iterable, List, Optional

import httpx
from openai import OpenAI

from app.config import (
    DEFAULT_FREE_MODELS,
    OPENROUTER_BASE_URL,
    settings,
)

logger = logging.getLogger("darglobal.llm")


class LRUCache:
    """Tiny in-memory LRU cache keyed on question, filter and model settings.

    Answers depend on the retrieval embedding model and the chat model override,
    so both are part of the key. Values are (answer, model_used) tuples.
    """

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._store: OrderedDict[str, tuple[str, str]] = OrderedDict()

    def _key(self, question: str, source_filter: str, suffix: str = "") -> str:
        raw = f"{question.strip().lower()}|{source_filter}|{suffix}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self, question: str, source_filter: str, suffix: str = ""
    ) -> Optional[tuple[str, str]]:
        key = self._key(question, source_filter, suffix)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(
        self,
        question: str,
        source_filter: str,
        answer: str,
        model_used: str,
        suffix: str = "",
    ) -> None:
        key = self._key(question, source_filter, suffix)
        self._store[key] = (answer, model_used)
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)


class OpenRouterClient:
    """Chat-completion client that falls back across free OpenRouter models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        models: Optional[Iterable[str]] = None,
        cache: Optional[LRUCache] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.base_url = base_url.rstrip("/")
        self.models = list(models) if models is not None else list(DEFAULT_FREE_MODELS)
        self.cache = cache or LRUCache(settings.CACHE_MAX_SIZE)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[OpenAI] = None
        self._session: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={"HTTP-Referer": "https://huggingface.co/spaces"},
                # trust_env=False: proxy env vars with IPv6 NO_PROXY entries (e.g.
                # "[::1]") crash httpx before any request is made.
                http_client=httpx.Client(trust_env=False),
                timeout=self.timeout,
                # Our own retry/backoff loop controls attempts; SDK retries would
                # multiply requests against rate-limited free endpoints.
                max_retries=0,
            )
        return self._client

    def _normalise_messages(
        self, messages: List[dict]
    ) -> List[dict]:
        out: List[dict] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role not in ("system", "user", "assistant"):
                role = "user"
            out.append({"role": role, "content": content})
        return out

    def chat(
        self,
        messages: List[dict],
        question: str = "",
        source_filter: str = "all",
        model_override: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        cache_suffix: str = "",
    ) -> str:
        """Return assistant text only (compatibility wrapper for chat_with_meta)."""
        return self.chat_with_meta(
            messages,
            question=question,
            source_filter=source_filter,
            model_override=model_override,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_suffix=cache_suffix,
        )[0]

    def chat_with_meta(
        self,
        messages: List[dict],
        question: str = "",
        source_filter: str = "all",
        model_override: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        cache_suffix: str = "",
    ) -> tuple[str, str]:
        """Return (assistant_text, model_used), trying models until one succeeds.

        If ``model_override`` is provided (e.g. typed by the user in the UI), it is
        tried first; on failure the client falls back through the default free chain.
        ``cache_suffix`` must include anything that changes the answer (embedding
        model, requested chat model) so cache hits stay correct.
        """
        normalised = self._normalise_messages(messages)

        # Build ordered model list: requested/default model first, then the
        # rest of the chain (deduped so a failed model is not retried twice).
        models_to_try: List[str] = []
        if model_override:
            custom = model_override.strip()
            if custom:
                models_to_try.append(custom)
        for model in self.models:
            if model not in models_to_try:
                models_to_try.append(model)

        # Cache lookup for cacheable (single-turn) questions.
        if question:
            cached = self.cache.get(question, source_filter, suffix=cache_suffix)
            if cached is not None:
                logger.info("Cache hit for question")
                return cached[0], f"{cached[1]} (cached)"

        if not self.api_key:
            logger.error("OPENROUTER_API_KEY is not configured")
            return (
                "The AI answering service is not configured yet: OPENROUTER_API_KEY "
                "is missing. Set it in the environment (or the Space secrets) and reload.",
                "unavailable",
            )

        extra_body = None
        if settings.REASONING_MAX_TOKENS > 0:
            extra_body = {"reasoning": {"max_tokens": settings.REASONING_MAX_TOKENS}}

        last_error: Optional[Exception] = None
        for model in models_to_try:
            for attempt in range(self.max_retries + 1):
                try:
                    logger.info("Calling model %s (attempt %d)", model, attempt + 1)
                    completion = self.client.chat.completions.create(
                        model=model,
                        messages=normalised,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body=extra_body,
                    )
                    text = completion.choices[0].message.content or ""
                    text = text.strip()
                    if not text:
                        # Some free models occasionally return an empty message;
                        # treat it as a failure so the chain tries the next model.
                        raise RuntimeError("Empty response from model")

                    if question:
                        self.cache.set(
                            question,
                            source_filter,
                            text,
                            model,
                            suffix=cache_suffix,
                        )
                    return text, model

                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    msg = str(exc)
                    is_rate = "429" in msg or "rate limit" in msg.lower()
                    is_server = any(code in msg for code in ("500", "502", "503", "504"))
                    is_timeout = "timeout" in msg.lower() or "timed out" in msg.lower()

                    if attempt < self.max_retries and (
                        is_rate or is_server or is_timeout
                    ):
                        wait = (attempt + 1) * 2
                        logger.warning(
                            "Model %s error (%s), retrying in %ds",
                            model,
                            msg[:120],
                            wait,
                        )
                        time.sleep(wait)
                        continue
                    break  # try next model

        # Friendly fallback message when everything is exhausted.
        logger.error("All models exhausted. Last error: %s", last_error)
        return (
            "I'm sorry — the AI answering service is temporarily unavailable "
            "(rate limits or model errors across all free models). "
            "Please try again in a minute or two."
        ), "unavailable"

    async def achat(self, *args, **kwargs) -> str:
        """Async wrapper (not used in the minimal build; kept for future streaming)."""
        return self.chat(*args, **kwargs)
