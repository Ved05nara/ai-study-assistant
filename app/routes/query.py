from fastapi import APIRouter
from pydantic import BaseModel

from app.services.embedding_service import model
from app.services.chroma_service import search_chunks
from app.services.chroma_service import collection

router = APIRouter()


@router.get("/stats")
def stats():

    return {
        "total_chunks": collection.count()
    }


class QueryRequest(BaseModel):
    question: str


@router.post("/query")
async def query_notes(request: QueryRequest):

    question_embedding = model.encode(
        request.question
    ).tolist()

    results = search_chunks(question_embedding)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    formatted_results = []

    for doc, meta, distance in zip(
        documents,
        metadatas,
        distances
    ):
        formatted_results.append(
            {
                "source": meta["source"],
                "distance": distance,
                "content": doc
            }
        )

    return {
        "question": request.question,
        "results": formatted_results
    }