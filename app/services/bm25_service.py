"""
BM25 keyword search service.

Loads all stored chunks from ChromaDB at startup and builds an in-memory
BM25 index. The index is refreshed automatically after every upload so
new documents are immediately searchable.

Reciprocal Rank Fusion (RRF) is used to merge BM25 and vector rankings.
"""

import logging
from threading import Lock

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ── Internal state ─────────────────────────────────────────────────────────────

_lock = Lock()
_bm25: BM25Okapi | None = None
_corpus_docs: list[str] = []       # raw text of every chunk
_corpus_ids: list[str] = []        # ChromaDB id of every chunk
_corpus_metas: list[dict] = []     # metadata of every chunk


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def build_index(docs: list[str], ids: list[str], metas: list[dict]) -> None:
    """
    (Re)build the BM25 index from the provided documents.
    Called once at startup and after every upload.
    """
    global _bm25, _corpus_docs, _corpus_ids, _corpus_metas

    if not docs:
        logger.warning("BM25: no documents to index.")
        return

    tokenized = [_tokenize(d) for d in docs]

    with _lock:
        _bm25 = BM25Okapi(tokenized)
        _corpus_docs = docs
        _corpus_ids = ids
        _corpus_metas = metas

    logger.info("BM25 index built with %d chunks.", len(docs))


def bm25_search(
    query: str,
    n_results: int = 10,
    filter_ids: set[str] | None = None,
) -> list[dict]:
    """
    Search the BM25 index for the top-n chunks matching the query.

    Args:
        query: The search query string.
        n_results: Maximum number of results to return.
        filter_ids: If provided, only return chunks whose ChromaDB id
                    is in this set (used to apply metadata filters).

    Returns:
        List of dicts with keys: id, document, metadata, bm25_score, bm25_rank
    """
    with _lock:
        if _bm25 is None or not _corpus_docs:
            logger.warning("BM25 index is empty — skipping BM25 search.")
            return []

        scores = _bm25.get_scores(_tokenize(query))

    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )

    results = []
    rank = 1
    for idx, score in ranked:
        if score <= 0:
            break
        chunk_id = _corpus_ids[idx]
        if filter_ids is not None and chunk_id not in filter_ids:
            continue
        results.append({
            "id": chunk_id,
            "document": _corpus_docs[idx],
            "metadata": _corpus_metas[idx],
            "bm25_score": float(score),
            "bm25_rank": rank,
        })
        rank += 1
        if rank > n_results:
            break

    return results


def is_ready() -> bool:
    """Return True if the BM25 index has been built."""
    return _bm25 is not None