"""Beginner-friendly retrievers and a shared Ollama answer generator."""
from __future__ import annotations
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")


class TfidfRetriever:
    """Retrieve pages by lexical similarity using TF-IDF and cosine similarity."""
    name = "tfidf"

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in documents])

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [
            {"page": self.documents[i]["page"], "text": self.documents[i]["text"], "score": float(scores[i])}
            for i in indices
        ]


class SemanticRetriever:
    """Retrieve pages by sentence-embedding similarity."""
    name = "semantic"

    def __init__(self, documents: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic retrieval needs sentence-transformers. Run: pip install -r requirements.txt"
            ) from exc
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [d["text"] for d in documents],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        query_embedding = self.model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ query_embedding
        indices = scores.argsort()[::-1][:top_k]
        return [
            {"page": self.documents[i]["page"], "text": self.documents[i]["text"], "score": float(scores[i])}
            for i in indices
        ]


def generate_answer(question: str, retrieved_documents: list[dict], model_name: str = DEFAULT_LLM_MODEL) -> str:
    """Use the same local LLM and prompt for any retrieval method."""
    if not retrieved_documents:
        context = "No passages were retrieved."
    else:
        blocks = []
        for doc in retrieved_documents:
            # Keep context manageable while retaining the evidence around the top-ranked page.
            excerpt = doc["text"][:3500]
            blocks.append(f"[PDF page {doc['page']}] {excerpt}")
        context = "\n\n".join(blocks)

    system_prompt = (
        "You are a careful question-answering assistant. Answer using only the CONTEXT provided. "
        "If the context does not state the answer, say: 'The document does not specify this information.' "
        "Do not use outside knowledge or guess. Keep the answer concise."
    )
    user_prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nAnswer only from the context."
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "").strip() or "[No answer returned by the local model.]"
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot reach Ollama at " + OLLAMA_URL + ". Start Ollama and run `ollama pull " + model_name + "`. "
            "For a retrieval-only test, use --skip-generation."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
