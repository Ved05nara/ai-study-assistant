import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="study_materials")


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
    upload_time: str = "",
    subject: str = "",
    semester: str = "",
    department: str = "",
    page_numbers: list[int] | None = None,
) -> None:
    """
    Store text chunks and their embeddings in ChromaDB with rich metadata.
    Deletes existing chunks for the same filename before inserting.
    """
    existing = collection.get(where={"source": filename})
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": filename,
            "chunk_index": i,
            "page_number": (page_numbers[i] if page_numbers and i < len(page_numbers) else 0),
            "upload_time": upload_time,
            "subject": subject.strip().lower(),
            "semester": semester.strip().lower(),
            "department": department.strip().lower(),
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def get_all_chunks() -> tuple[list[str], list[str], list[dict]]:
    """
    Return all chunks stored in ChromaDB as (documents, ids, metadatas).
    Used by BM25 to build its index.
    """
    try:
        data = collection.get(include=["documents", "metadatas"])
        return (
            data.get("documents") or [],
            data.get("ids") or [],
            data.get("metadatas") or [],
        )
    except Exception:
        return [], [], []


def _build_where(
    subject: str | None = None,
    semester: str | None = None,
    department: str | None = None,
    filename: str | None = None,
) -> dict | None:
    clauses = []
    if filename:
        clauses.append({"source": {"$eq": filename}})
    if subject:
        clauses.append({"subject": {"$eq": subject.strip().lower()}})
    if semester:
        clauses.append({"semester": {"$eq": semester.strip().lower()}})
    if department:
        clauses.append({"department": {"$eq": department.strip().lower()}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search_chunks(
    question_embedding: list[float],
    n_results: int = 10,
    subject: str | None = None,
    semester: str | None = None,
    department: str | None = None,
    filename: str | None = None,
) -> dict:
    """Vector search with optional metadata filters."""
    where = _build_where(subject, semester, department, filename)
    kwargs = dict(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    return collection.query(**kwargs)


def get_ids_for_filter(
    subject: str | None = None,
    semester: str | None = None,
    department: str | None = None,
    filename: str | None = None,
) -> set[str] | None:
    """
    Return the set of chunk IDs that match the given metadata filters.
    Returns None if no filters are active (meaning: allow all).
    Used to restrict BM25 results to the same filtered subset as vector search.
    """
    where = _build_where(subject, semester, department, filename)
    if not where:
        return None
    try:
        data = collection.get(where=where, include=[])
        return set(data.get("ids") or [])
    except Exception:
        return None


def list_documents(
    subject: str | None = None,
    semester: str | None = None,
    department: str | None = None,
) -> list[dict]:
    """Return unique documents with metadata, optionally filtered."""
    try:
        where = _build_where(subject=subject, semester=semester, department=department)
        kwargs = {"include": ["metadatas"]}
        if where:
            kwargs["where"] = where
        all_data = collection.get(**kwargs)
        metadatas = all_data.get("metadatas") or []
    except Exception:
        return []

    docs: dict[str, dict] = {}
    for meta in metadatas:
        src = meta.get("source", "Unknown")
        if src not in docs:
            docs[src] = {
                "filename": src,
                "chunk_count": 0,
                "upload_time": meta.get("upload_time", ""),
                "subject": meta.get("subject", ""),
                "semester": meta.get("semester", ""),
                "department": meta.get("department", ""),
            }
        docs[src]["chunk_count"] += 1

    return list(docs.values())


def delete_document(filename: str) -> int:
    """Delete all chunks for a filename. Returns count deleted."""
    existing = collection.get(where={"source": filename})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)