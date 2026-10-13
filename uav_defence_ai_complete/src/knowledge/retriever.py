from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    source: str
    text: str


class SimpleRetriever:
    """Lightweight TF-IDF retriever; avoids requiring a hosted vector DB."""

    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.vectorizer = None
        self.matrix = None
        if chunks:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    @classmethod
    def from_folder(cls, folder: str | Path) -> "SimpleRetriever":
        folder = Path(folder)
        chunks: list[DocumentChunk] = []
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                chunks.append(DocumentChunk(str(path), text))
        return cls(chunks)

    def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = scores.argsort()[::-1][:top_k]
        return [self.chunks[int(i)] for i in idx if scores[int(i)] > 0]
