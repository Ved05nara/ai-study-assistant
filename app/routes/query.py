import json
import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.chroma_service import collection, search_chunks, get_ids_for_filter
from app.services.embedding_service import model as embed_model
from app.services.bm25_service import bm25_search
from app.services.reranker_service import rerank
from app.services.llm_service import (
    generate_answer, generate_general_overview,
    stream_answer, stream_general_overview,
)
from app.services.memory_service import add_message, clear_history, get_history

logger = logging.getLogger(__name__)
router = APIRouter()

NO_ANSWER_MSG = "I could not find that information in the uploaded notes."

# ── Tuning knobs ───────────────────────────────────────────────────────────────
VECTOR_FETCH = 10        # how many chunks vector search retrieves
BM25_FETCH = 10          # how many chunks BM25 retrieves
RRF_K = 60               # RRF constant (higher = smoother rank fusion)
RERANK_TOP_K = 5         # candidates sent to cross-encoder, then to LLM
RELEVANCE_THRESHOLD = 0.0  # cross-encoder scores are unbounded; 0.0 keeps positives
SEARCH_HISTORY_TURNS = 2


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf_merge(
    vector_results: tuple[list, list, list],
    bm25_results: list[dict],
) -> list[dict]:
    """
    Merge vector search and BM25 results using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)

    Returns a unified list of candidate dicts sorted by RRF score (desc).
    Each dict has: document, metadata, rrf_score, vector_distance (optional).
    """
    v_docs, v_metas, v_dists = vector_results

    scores: dict[str, float] = {}
    candidates: dict[str, dict] = {}

    # Score vector results
    for rank, (doc, meta, dist) in enumerate(zip(v_docs, v_metas, v_dists), start=1):
        uid = meta.get("source", "") + str(meta.get("chunk_index", rank))
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (RRF_K + rank)
        candidates[uid] = {
            "document": doc,
            "metadata": meta,
            "vector_distance": dist,
            "rrf_score": 0.0,
        }

    # Score BM25 results
    for result in bm25_results:
        meta = result["metadata"]
        uid = meta.get("source", "") + str(meta.get("chunk_index", result["id"]))
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (RRF_K + result["bm25_rank"])
        if uid not in candidates:
            candidates[uid] = {
                "document": result["document"],
                "metadata": meta,
                "vector_distance": None,
                "rrf_score": 0.0,
            }

    # Write final RRF scores and sort
    for uid, score in scores.items():
        candidates[uid]["rrf_score"] = score

    return sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)


# ── Hybrid retrieval pipeline ──────────────────────────────────────────────────

def _build_search_query(question: str, history: list[dict]) -> str:
    recent = [m["content"] for m in history if m["role"] == "user"][-SEARCH_HISTORY_TURNS:]
    return " ".join(recent + [question])


def _hybrid_retrieve(
    question: str,
    history: list[dict],
    subject: str | None = None,
    semester: str | None = None,
    department: str | None = None,
    filename: str | None = None,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
      1. Vector search (ChromaDB)
      2. BM25 keyword search
      3. RRF merge
      4. Cross-encoder re-rank → top RERANK_TOP_K
    """
    search_query = _build_search_query(question, history)
    embedding = embed_model.encode(search_query).tolist()

    # 1. Vector search
    v_results = search_chunks(
        embedding, n_results=VECTOR_FETCH,
        subject=subject, semester=semester,
        department=department, filename=filename,
    )
    v_docs  = v_results.get("documents", [[]])[0] if v_results else []
    v_metas = v_results.get("metadatas", [[]])[0] if v_results else []
    v_dists = v_results.get("distances", [[]])[0] if v_results else []

    # 2. BM25 search (respects metadata filters via id allowlist)
    filter_ids = get_ids_for_filter(subject, semester, department, filename)
    bm25_results = bm25_search(search_query, n_results=BM25_FETCH, filter_ids=filter_ids)

    # 3. RRF merge
    merged = _rrf_merge((v_docs, v_metas, v_dists), bm25_results)

    if not merged:
        return []

    # 4. Cross-encoder re-rank
    reranked = rerank(question, merged, top_k=RERANK_TOP_K)

    # Filter out chunks with negative cross-encoder scores (clearly irrelevant)
    return [c for c in reranked if c.get("rerank_score", 0.0) >= RELEVANCE_THRESHOLD]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Utility endpoints ──────────────────────────────────────────────────────────

@router.get("/stats")
def stats():
    return {"total_chunks": collection.count()}


@router.get("/health")
def health():
    from app.services.bm25_service import is_ready
    return {
        "status": "running",
        "chunks": collection.count(),
        "bm25_ready": is_ready(),
    }


@router.post("/reset-chat")
def reset_chat(session_id: str = "default"):
    clear_history(session_id)
    return {"message": "Chat history cleared", "session_id": session_id}


# ── Request model ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str = Field(default="default")
    subject: str | None = Field(default=None)
    semester: str | None = Field(default=None)
    department: str | None = Field(default=None)
    filename: str | None = Field(default=None)


# ── Non-streaming /query ───────────────────────────────────────────────────────

@router.post("/query")
async def query_notes(request: QueryRequest):
    try:
        history = get_history(request.session_id)

        candidates = _hybrid_retrieve(
            request.question, history,
            subject=request.subject, semester=request.semester,
            department=request.department, filename=request.filename,
        )

        needs_general = not candidates

        if needs_general:
            overview = generate_general_overview(request.question)
            answer = (
                "I couldn't find that information in your uploaded notes, "
                f"but here's a general overview:\n\n{overview}"
            )
            add_message(request.session_id, "user", request.question)
            add_message(request.session_id, "assistant", answer)
            return {
                "question": request.question, "answer": answer,
                "sources": [], "chunks_data": [],
                "confidence_score": 0.0, "answer_source": "general",
                "session_id": request.session_id,
            }

        docs    = [c["document"] for c in candidates]
        metas   = [c["metadata"] for c in candidates]
        scores  = [c.get("rerank_score", 0.0) for c in candidates]

        start = time.time()
        answer = generate_answer(request.question, docs, history)
        logger.info("LLM response time: %.2f sec", time.time() - start)

        add_message(request.session_id, "user", request.question)
        add_message(request.session_id, "assistant", answer)

        sources = list({m.get("source", "N/A") for m in metas})
        chunks_data = [
            {
                "source": m.get("source", "N/A"),
                "page_number": m.get("page_number", 0),
                "rerank_score": round(s, 4),
                "text": d,
            }
            for d, m, s in zip(docs, metas, scores)
        ]
        confidence = round(max(0.0, min(scores) / 10 + 0.5) * 100, 1) if scores else 0.0

        if answer.strip() == NO_ANSWER_MSG:
            sources, chunks_data, confidence = [], [], 0.0

        return {
            "question": request.question, "answer": answer,
            "sources": sources, "chunks_data": chunks_data,
            "confidence_score": confidence, "answer_source": "notes",
            "session_id": request.session_id,
        }

    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


# ── Streaming /query-stream ────────────────────────────────────────────────────

@router.post("/query-stream")
async def query_notes_stream(request: QueryRequest):
    history = get_history(request.session_id)
    candidates = _hybrid_retrieve(
        request.question, history,
        subject=request.subject, semester=request.semester,
        department=request.department, filename=request.filename,
    )
    needs_general = not candidates

    docs   = [c["document"] for c in candidates] if candidates else []
    metas  = [c["metadata"] for c in candidates] if candidates else []
    scores = [c.get("rerank_score", 0.0) for c in candidates] if candidates else []

    def generate():
        full_answer = ""
        try:
            if needs_general:
                prefix = (
                    "I couldn't find that information in your uploaded notes, "
                    "but here's a general overview:\n\n"
                )
                full_answer += prefix
                yield _sse({"token": prefix})
                for tok in stream_general_overview(request.question):
                    full_answer += tok
                    yield _sse({"token": tok})
                sources, chunks_data, confidence = [], [], 0.0
                answer_source = "general"
            else:
                for tok in stream_answer(request.question, docs, history):
                    full_answer += tok
                    yield _sse({"token": tok})

                sources = list({m.get("source", "N/A") for m in metas})
                chunks_data = [
                    {
                        "source": m.get("source", "N/A"),
                        "page_number": m.get("page_number", 0),
                        "rerank_score": round(s, 4),
                        "text": d,
                    }
                    for d, m, s in zip(docs, metas, scores)
                ]
                confidence = round(max(0.0, min(scores) / 10 + 0.5) * 100, 1) if scores else 0.0
                answer_source = "notes"

                if full_answer.strip() == NO_ANSWER_MSG:
                    sources, chunks_data, confidence = [], [], 0.0

            add_message(request.session_id, "user", request.question)
            add_message(request.session_id, "assistant", full_answer)

            yield _sse({
                "done": True,
                "confidence_score": confidence,
                "sources": sources,
                "answer_source": answer_source,
                "chunks_data": chunks_data,
            })

        except Exception as exc:
            logger.exception("Stream generation failed")
            yield _sse({"error": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )