import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.upload import router as upload_router
from app.routes.query import router as query_router
from app.routes.documents import router as documents_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the BM25 index from whatever is already in ChromaDB at boot."""
    from app.services.chroma_service import get_all_chunks
    from app.services.bm25_service import build_index

    docs, ids, metas = get_all_chunks()
    build_index(docs, ids, metas)
    yield


app = FastAPI(title="CampusGPT", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)


@app.get("/")
def root():
    return {"message": "CampusGPT v2.0 Running"}