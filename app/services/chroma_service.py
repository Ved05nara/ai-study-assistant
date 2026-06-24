import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="study_materials"
)


def store_chunks(chunks, embeddings, filename):

    ids = [
        f"{filename}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": filename,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )


def search_chunks(question_embedding, n_results=3):

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    return results