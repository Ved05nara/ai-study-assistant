"""
Cross-Encoder re-ranking service.

After hybrid retrieval, the top-k candidate chunks are re-scored by a
cross-encoder model that jointly encodes the (question, chunk) pair.
This is significantly more accurate than bi-encoder cosine/L2 distance
because the model can attend to both texts simultaneously.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~80 MB, fast on CPU (~50ms for 10 candidates)
  - Trained on MS MARCO passage ranking
"""

import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info("Loading cross-encoder model '%s'...", _MODEL_NAME)
        _model = CrossEncoder(_MODEL_NAME)
        logger.info("Cross-encoder loaded.")
    return _model


def rerank(
    question: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Re-rank a list of candidate chunks using the cross-encoder.

    Args:
        question: The user's question.
        candidates: List of dicts, each must have a "document" key with chunk text.
        top_k: Number of top results to return after re-ranking.

    Returns:
        Top-k candidates sorted by cross-encoder score (descending),
        each with an added "rerank_score" field.
    """
    if not candidates:
        return []

    model = _get_model()

    pairs = [(question, c["document"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]