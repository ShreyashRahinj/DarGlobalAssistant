"""RAG pipeline: chunking, hybrid retrieval (BM25 + embeddings + RRF).

The local fastembed model is built at startup. Alternative OpenRouter embedding
models can be selected in the UI; their corpus matrix is built on first use in a
background thread (with progress) and cached in memory and under data/index/.
BM25 is shared because it depends only on the chunk text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    DEFAULT_EMBEDDING_MODEL,
    FINAL_TOP_K,
    RRF_K,
    SIMILARITY_THRESHOLD,
    SOURCES,
    TOP_K_BM25,
    TOP_K_EMBED,
    settings,
)
from app.embedding_backends import EmbeddingError, make_embedder

logger = logging.getLogger("darglobal.rag")


@dataclass
class Chunk:
    text: str
    source: str
    url: str
    title: str
    chunk_id: str
    embedding: Optional[np.ndarray] = None


@dataclass
class EmbeddingState:
    model: str
    status: str = "not_built"  # not_built | building | ready | error
    progress: int = 0
    total: int = 0
    dimension: int = 0
    error: str = ""
    updated_at: float = 0.0
    matrix: Optional[np.ndarray] = field(default=None, repr=False)

    def public(self) -> dict:
        return {
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "dimension": self.dimension,
            "error": self.error,
        }


class EmbeddingNotReady(RuntimeError):
    """Raised when retrieval is requested with a model that has no ready matrix."""

    def __init__(self, model: str, state: dict):
        self.model = model
        self.state = state
        super().__init__(f"Embedding model {model} is not ready")


LEGAL_URL_MARKERS = ("/s/privacy-policy", "/s/terms-of-use", "/s/terms-of-advertising")
LEGAL_QUERY_HINTS = {"privacy", "terms", "legal", "advertising", "cookies", "conditions"}
COMPARISON_HINTS = {"compare", "comparison", "differ", "difference", "vs", "versus"}


def _is_comparison_query(query: str) -> bool:
    tokens = set(_simple_tokenize(query))
    return bool(tokens & COMPARISON_HINTS) or {"darglobal", "wasalt"} <= tokens


def _is_legal_page(url: str) -> bool:
    return any(marker in url for marker in LEGAL_URL_MARKERS)


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def chunk_text(text: str, source: str, url: str, title: str, index: int) -> List[Chunk]:
    """Split a page into overlapping chunks of roughly CHUNK_SIZE_TOKENS words."""
    words = text.split()
    if not words:
        return []
    chunks: List[Chunk] = []
    prefix = f"{title}. " if title else ""
    start = 0
    seq = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE_TOKENS, len(words))
        # Title first: it carries the city/property type that queries match on.
        piece = prefix + " ".join(words[start:end])
        chunk_id = f"{index}-{seq}-{abs(hash(piece)) % 10_000_000}"
        chunks.append(
            Chunk(
                text=piece,
                source=source,
                url=url,
                title=title,
                chunk_id=chunk_id,
            )
        )
        seq += 1
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP_TOKENS
        if start < 0:
            start = 0
    return chunks


class Retriever:
    def __init__(
        self,
        index_dir: str | Path = "data/index",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.base_url = base_url or settings.OPENROUTER_BASE_URL
        self.batch_size = batch_size or settings.EMBED_BATCH_SIZE

        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._source_path: Optional[Path] = None
        self._embedders: Dict[str, object] = {}
        self._states: Dict[str, EmbeddingState] = {}
        self._lock = threading.Lock()
        self._build_semaphore = threading.BoundedSemaphore(2)
        self._built = False

    # ------------------------------------------------------------------ #
    # Build / load index
    # ------------------------------------------------------------------ #
    def build_from_jsonl(self, pages_jsonl: str | Path) -> int:
        """Read pages.jsonl, chunk every page, embed locally, and persist the index."""
        self._source_path = Path(pages_jsonl)
        pages = self._read_pages(pages_jsonl)
        all_chunks: List[Chunk] = []
        for i, page in enumerate(pages):
            all_chunks.extend(
                chunk_text(
                    page.get("text", ""),
                    page.get("source", "unknown"),
                    page.get("url", ""),
                    page.get("title", ""),
                    i,
                )
            )

        self.chunks = all_chunks
        if not self.chunks:
            logger.warning("No chunks built from %s", pages_jsonl)
            self._built = True
            return 0

        self._build_bm25()
        self._build_embeddings_sync(DEFAULT_EMBEDDING_MODEL)
        self._persist()
        self._built = True
        logger.info("Built index with %d chunks from %d pages", len(self.chunks), len(pages))
        return len(self.chunks)

    def load(self) -> bool:
        """Load a previously persisted index from data/index/."""
        meta_path = self.index_dir / "meta.json"
        chunks_path = self.index_dir / "chunks.jsonl"
        emb_path = self.index_dir / "embeddings.npy"
        if not (meta_path.exists() and chunks_path.exists() and emb_path.exists()):
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
        source = meta.get("source_path")
        if source and Path(source).exists():
            stat = Path(source).stat()
            # Docker layer export truncates mtimes to whole seconds, so compare
            # at second precision; the size check still catches content changes.
            mtime_delta = abs(stat.st_mtime_ns - int(meta.get("source_mtime_ns") or 0))
            if mtime_delta >= 1_000_000_000 or stat.st_size != meta.get("source_size"):
                logger.info("Index at %s is stale relative to %s", self.index_dir, source)
                return False

        try:
            chunks: List[Chunk] = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(Chunk(**json.loads(line)))
            matrix = np.load(emb_path)
            if matrix.shape[0] != len(chunks):
                raise ValueError(
                    f"Embedding matrix has {matrix.shape[0]} rows for {len(chunks)} chunks"
                )
            self.chunks = chunks
            self._states[DEFAULT_EMBEDDING_MODEL] = EmbeddingState(
                model=DEFAULT_EMBEDDING_MODEL,
                status="ready",
                progress=len(chunks),
                total=len(chunks),
                dimension=int(matrix.shape[1]),
                updated_at=time.time(),
                matrix=self._normalise(matrix.astype("float32")),
            )
            self._build_bm25()
            self._built = True
            logger.info("Loaded index with %d chunks", len(self.chunks))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load index: %s", exc)
            self.chunks = []
            return False

    def _persist(self) -> None:
        with open(self.index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(
                    json.dumps(
                        {
                            "text": c.text,
                            "source": c.source,
                            "url": c.url,
                            "title": c.title,
                            "chunk_id": c.chunk_id,
                        }
                    )
                    + "\n"
                )
        default_state = self._states.get(DEFAULT_EMBEDDING_MODEL)
        if default_state and default_state.matrix is not None:
            # Legacy single-matrix file used by load().
            np.save(self.index_dir / "embeddings.npy", default_state.matrix)
        meta = {
            "n_chunks": len(self.chunks),
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
        }
        if self._source_path and self._source_path.exists():
            stat = self._source_path.stat()
            meta["source_path"] = str(self._source_path)
            meta["source_mtime_ns"] = stat.st_mtime_ns
            meta["source_size"] = stat.st_size
        with open(self.index_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _read_pages(self, pages_jsonl: str | Path) -> List[dict]:
        pages: List[dict] = []
        seen_hashes = set()
        with open(pages_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                content_hash = hash(text)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
                pages.append(obj)
        return pages

    # ------------------------------------------------------------------ #
    # Sub-index builders
    # ------------------------------------------------------------------ #
    def _build_bm25(self) -> None:
        tokenised = [_simple_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenised)

    def _get_embedder(self, model_id: str):
        embedder = self._embedders.get(model_id)
        if embedder is None:
            embedder = make_embedder(
                model_id,
                local_model_id=DEFAULT_EMBEDDING_MODEL,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=settings.EMBED_TIMEOUT_SEC,
                max_retries=settings.EMBED_MAX_RETRIES,
            )
            self._embedders[model_id] = embedder
        return embedder

    def _state(self, model_id: str) -> EmbeddingState:
        state = self._states.get(model_id)
        if state is None:
            state = EmbeddingState(model=model_id)
            self._states[model_id] = state
        return state

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix / norms

    def _matrix_paths(self, model_id: str) -> Tuple[Path, Path]:
        slug = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:16]
        return (
            self.index_dir / f"emb-{slug}.npy",
            self.index_dir / f"emb-{slug}.json",
        )

    def _persist_matrix(self, model_id: str, matrix: np.ndarray) -> None:
        npy_path, meta_path = self._matrix_paths(model_id)
        try:
            np.save(npy_path, matrix)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": model_id,
                        "n_chunks": len(self.chunks),
                        "dimension": int(matrix.shape[1]),
                    },
                    f,
                )
        except OSError as exc:
            logger.warning("Could not persist embeddings for %s: %s", model_id, exc)

    def _load_matrix_from_disk(self, model_id: str) -> Optional[np.ndarray]:
        npy_path, meta_path = self._matrix_paths(model_id)
        if not (npy_path.exists() and meta_path.exists()):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            if info.get("model") != model_id or info.get("n_chunks") != len(self.chunks):
                return None
            matrix = np.load(npy_path)
            if matrix.shape[0] != len(self.chunks):
                return None
            return self._normalise(matrix.astype("float32"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load cached embeddings for %s: %s", model_id, exc)
            return None

    def _build_embeddings_sync(self, model_id: str) -> None:
        state = self._state(model_id)
        state.status = "building"
        state.total = len(self.chunks)
        self._build_state(model_id)

    def _build_state(self, model_id: str) -> None:
        state = self._state(model_id)
        # At most two corpus builds at a time so a public demo cannot burn the
        # free quota with parallel indexing jobs.
        with self._build_semaphore:
            self._build_state_inner(model_id, state)

    def _build_state_inner(self, model_id: str, state: EmbeddingState) -> None:
        try:
            embedder = self._get_embedder(model_id)
            texts = [c.text for c in self.chunks]
            state.total = len(texts)

            def on_progress(done: int, total: int) -> None:
                state.progress = done
                state.total = total

            matrix = embedder.embed(texts, batch_size=self.batch_size, progress=on_progress)
            if matrix.shape[0] != len(texts):
                raise EmbeddingError(
                    f"Embedder returned {matrix.shape[0]} vectors for {len(texts)} chunks"
                )
            matrix = self._normalise(matrix.astype("float32"))
            with self._lock:
                state.matrix = matrix
                state.dimension = int(matrix.shape[1])
                state.progress = len(texts)
                state.total = len(texts)
                state.status = "ready"
                state.error = ""
                state.updated_at = time.time()
            self._persist_matrix(model_id, matrix)
            logger.info(
                "Embedding model %s ready (%d chunks, %d dims)",
                model_id,
                len(texts),
                matrix.shape[1],
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                state.status = "error"
                state.error = str(exc)[:500]
                state.updated_at = time.time()
            logger.error("Failed to build embeddings for %s: %s", model_id, exc)

    # ------------------------------------------------------------------ #
    # Public embedding-model selection API
    # ------------------------------------------------------------------ #
    def warm_embedding_model(self, model_id: str | None) -> dict:
        """Ensure a matrix exists, building it in the background if needed."""
        model_id = (model_id or "").strip() or DEFAULT_EMBEDDING_MODEL
        if not self.chunks:
            return {
                "model": model_id,
                "status": "error",
                "progress": 0,
                "total": 0,
                "dimension": 0,
                "error": "No chunks indexed yet",
            }

        if model_id not in self._states and len(self._states) >= settings.MAX_EMBEDDING_STATES:
            return {
                "model": model_id,
                "status": "error",
                "progress": 0,
                "total": len(self.chunks),
                "dimension": 0,
                "error": (
                    "Too many embedding models have been tried in this session. "
                    "Restart the app or choose one of the listed models."
                ),
            }

        state = self._state(model_id)
        if state.status in ("ready", "building"):
            return state.public()

        with self._lock:
            if state.status in ("ready", "building"):
                return state.public()
            cached = self._load_matrix_from_disk(model_id)
            if cached is not None:
                state.matrix = cached
                state.dimension = int(cached.shape[1])
                state.progress = len(self.chunks)
                state.total = len(self.chunks)
                state.status = "ready"
                state.error = ""
                state.updated_at = time.time()
                return state.public()
            state.status = "building"
            state.progress = 0
            state.total = len(self.chunks)
            state.error = ""

        thread = threading.Thread(
            target=self._build_state, args=(model_id,), daemon=True, name="embed-build"
        )
        thread.start()
        return state.public()

    def embedding_status(self, model_id: str | None) -> dict:
        model_id = (model_id or "").strip() or DEFAULT_EMBEDDING_MODEL
        state = self._states.get(model_id)
        if state is None:
            return {
                "model": model_id,
                "status": "not_built",
                "progress": 0,
                "total": len(self.chunks),
                "dimension": 0,
                "error": "",
            }
        return state.public()

    def embedding_statuses(self) -> List[dict]:
        return [state.public() for state in self._states.values()]

    def _encode_query(self, query: str, model_id: str) -> np.ndarray:
        embedder = self._get_embedder(model_id)
        matrix = embedder.embed([query], batch_size=1)
        vec = np.asarray(matrix[0], dtype="float32")
        return vec / (np.linalg.norm(vec) + 1e-12)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        source_filter: str = "all",
        top_k_bm25: int = TOP_K_BM25,
        top_k_embed: int = TOP_K_EMBED,
        final_top_k: int = FINAL_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        embedding_model: str | None = None,
        balance_sources: bool | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """Return (chunk, fused_score) using hybrid BM25 + embeddings + RRF.

        Comparison questions (``balance_sources`` None = auto-detect) run once
        per source and merge, so one source cannot crowd out the other.
        """
        if not self._built or not self.chunks:
            return []

        if balance_sources is None:
            balance_sources = _is_comparison_query(query)
        if balance_sources and source_filter == "all" and len(SOURCES) > 1:
            per_source = max(1, final_top_k // len(SOURCES))
            merged: List[Tuple[Chunk, float]] = []
            seen: set[tuple[str, str]] = set()
            for src in SOURCES:
                for chunk, score in self.retrieve(
                    query,
                    source_filter=src,
                    top_k_bm25=top_k_bm25,
                    top_k_embed=top_k_embed,
                    final_top_k=per_source,
                    threshold=threshold,
                    embedding_model=embedding_model,
                    balance_sources=False,
                ):
                    key = (chunk.url, chunk.chunk_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append((chunk, score))
            merged.sort(key=lambda x: x[1], reverse=True)
            return self._cap_per_url(merged, final_top_k)

        model_id = (embedding_model or "").strip() or DEFAULT_EMBEDDING_MODEL
        matrix = self._ready_matrix(model_id)

        # Candidate set filtered by source. Policy/terms pages are boilerplate
        # that matches property questions weakly; only use them when asked about.
        legal_query = bool(set(_simple_tokenize(query)) & LEGAL_QUERY_HINTS)
        candidate_idx = [
            i for i, c in enumerate(self.chunks)
            if (source_filter == "all" or c.source == source_filter)
            and (legal_query or not _is_legal_page(c.url))
        ]
        if not candidate_idx:
            return []

        # BM25 scores over candidates.
        tokenised_q = _simple_tokenize(query)
        bm25_scores = self._bm25.get_scores(tokenised_q) if self._bm25 else np.zeros(len(self.chunks))
        bm25_rank = self._argsort_desc(bm25_scores, candidate_idx, top_k_bm25)

        # Embedding scores over candidates.
        try:
            qvec = self._encode_query(query, model_id)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingNotReady(
                model_id,
                {
                    "model": model_id,
                    "status": "error",
                    "progress": 0,
                    "total": len(self.chunks),
                    "dimension": 0,
                    "error": str(exc)[:500],
                },
            ) from exc
        if qvec.shape[0] != matrix.shape[1]:
            raise EmbeddingNotReady(
                model_id,
                {
                    "model": model_id,
                    "status": "error",
                    "progress": 0,
                    "total": len(self.chunks),
                    "dimension": int(matrix.shape[1]),
                    "error": "Query vector dimension does not match the index; rebuild required",
                },
            )
        emb_scores = matrix @ qvec  # cosine since both normalised
        emb_rank = self._argsort_desc(emb_scores, candidate_idx, top_k_embed)

        # Reciprocal Rank Fusion.
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(bm25_rank, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(emb_rank, start=1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        # Combine: normalise RRF + max raw signal, then threshold on embedding.
        fused: List[Tuple[Chunk, float]] = []
        for idx, score in rrf.items():
            chunk = self.chunks[idx]
            raw_signal = max(float(bm25_scores[idx]), float(emb_scores[idx]))
            combined = score + 0.5 * raw_signal
            # Threshold: require a decent embedding match so weak queries return nothing.
            if float(emb_scores[idx]) >= threshold:
                fused.append((chunk, combined))

        fused.sort(key=lambda x: x[1], reverse=True)
        return self._cap_per_url(fused, final_top_k)

    @staticmethod
    def _cap_per_url(
        results: List[Tuple[Chunk, float]], limit: int, max_per_url: int = 2
    ) -> List[Tuple[Chunk, float]]:
        """Keep top results while allowing at most max_per_url chunks per page."""
        out: List[Tuple[Chunk, float]] = []
        counts: Dict[str, int] = {}
        for chunk, score in results:
            key = chunk.url or chunk.chunk_id
            if counts.get(key, 0) >= max_per_url:
                continue
            counts[key] = counts.get(key, 0) + 1
            out.append((chunk, score))
            if len(out) >= limit:
                break
        return out

    def _ready_matrix(self, model_id: str) -> np.ndarray:
        state = self._states.get(model_id)
        if state is not None and state.status == "ready" and state.matrix is not None:
            matrix = state.matrix
        else:
            matrix = self._load_matrix_from_disk(model_id)
            if matrix is None:
                raise EmbeddingNotReady(model_id, self.embedding_status(model_id))
            state = self._state(model_id)
            with self._lock:
                state.matrix = matrix
                state.dimension = int(matrix.shape[1])
                state.progress = len(self.chunks)
                state.total = len(self.chunks)
                state.status = "ready"
                state.error = ""
                state.updated_at = time.time()
        if matrix.shape[0] != len(self.chunks):
            raise EmbeddingNotReady(
                model_id,
                {
                    **self.embedding_status(model_id),
                    "status": "error",
                    "error": "Embedding index size does not match the chunk store; rebuild required",
                },
            )
        return matrix

    @staticmethod
    def _argsort_desc(scores: np.ndarray, candidate_idx: List[int], top_k: int) -> List[int]:
        cand = np.array(candidate_idx)
        order = np.argsort(-scores[cand], kind="stable")
        return [int(cand[i]) for i in order[:top_k]]

    # Convenience for the API layer.
    def get_context(
        self,
        query: str,
        source_filter: str = "all",
        embedding_model: str | None = None,
        balance_sources: bool = False,
    ) -> Tuple[str, List[dict]]:
        """Return (context_string, sources_list) for the LLM prompt.

        With ``balance_sources`` and ``source_filter="all"``, retrieval runs per
        source and merges the results, so comparison questions always see both
        DarGlobal and Wasalt pages instead of one source crowding out the other.
        """
        results = self.retrieve(
            query,
            source_filter=source_filter,
            embedding_model=embedding_model,
            balance_sources=balance_sources,
        )
        if not results:
            return "", []
        parts: List[str] = []
        sources: List[dict] = []
        for i, (chunk, score) in enumerate(results, start=1):
            parts.append(f"[{i}] {chunk.text}\nSource: {chunk.title} ({chunk.url})")
            sources.append(
                {
                    "id": i,
                    "title": chunk.title,
                    "url": chunk.url,
                    "source": chunk.source,
                    "snippet": chunk.text[:300],
                    "score": round(float(score), 4),
                }
            )
        return "\n\n".join(parts), sources
