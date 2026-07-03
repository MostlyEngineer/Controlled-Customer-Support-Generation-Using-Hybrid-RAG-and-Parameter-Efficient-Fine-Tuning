from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from support_agent.documents import DocumentChunk
from support_agent.schema import RetrievedChunk


class TfidfRetriever:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Retriever requires at least one document chunk")
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(self, query: str, top_k: int = 3, category_hint: str | None = None) -> list[RetrievedChunk]:
        search_query = f"{query} {category_hint or ''}".strip()
        query_vector = self._vectorizer.transform([search_query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        if category_hint:
            hinted_document_id = category_hint.casefold().replace("_", "-")
            for idx, chunk in enumerate(self._chunks):
                if chunk.document_id.casefold().replace("_", "-") == hinted_document_id:
                    scores[idx] += 0.25
        ranked = scores.argsort()[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for idx in ranked:
            chunk = self._chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    source_path=chunk.source_path,
                    chunk_index=chunk.chunk_index,
                    score=float(scores[int(idx)]),
                    text=chunk.text,
                )
            )
        return results
