from __future__ import annotations

from pathlib import Path

from support_agent.documents import DocumentChunk, chunk_documents


def build_chroma_index(
    sop_dir: Path,
    persist_dir: Path,
    collection_name: str = "support_sops",
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> int:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install optional capstone dependencies first: "
            "python -m pip install -r requirements-capstone.txt"
        ) from exc

    chunks = chunk_documents(sop_dir)
    model = SentenceTransformer(embedding_model_id)
    embeddings = model.encode([chunk.text for chunk in chunks], normalize_embeddings=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(collection_name)
    ids = [_chunk_id(chunk) for chunk in chunks]
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "document_id": chunk.document_id,
                "title": chunk.title,
                "source_path": chunk.source_path,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ],
    )
    return len(chunks)


def _chunk_id(chunk: DocumentChunk) -> str:
    return f"{chunk.document_id}:{chunk.chunk_index}"
