import logging
import os

from fastapi import APIRouter, HTTPException, Query

from app.services.chroma_service import list_documents, delete_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = "uploads"


@router.get("")
def get_documents(
    subject: str | None = Query(default=None, description="Filter by subject"),
    semester: str | None = Query(default=None, description="Filter by semester"),
    department: str | None = Query(default=None, description="Filter by department"),
):
    """
    Return all indexed documents, with optional filters.
    Example: GET /documents?subject=mathematics&semester=sem3
    """
    docs = list_documents(subject=subject, semester=semester, department=department)
    return {"documents": docs, "total": len(docs)}


@router.delete("/{filename}")
def remove_document(filename: str):
    """
    Delete all chunks for the given filename from ChromaDB and remove from disk.
    """
    chunks_deleted = delete_document(filename)

    if chunks_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in the index.",
        )

    file_path = os.path.join(UPLOAD_DIR, filename)
    removed_from_disk = False
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            removed_from_disk = True
        except OSError as exc:
            logger.warning("Could not delete file '%s' from disk: %s", file_path, exc)

    logger.info("Deleted '%s': %d chunks removed", filename, chunks_deleted)

    return {
        "message": f"Document '{filename}' deleted successfully.",
        "chunks_deleted": chunks_deleted,
        "removed_from_disk": removed_from_disk,
    }