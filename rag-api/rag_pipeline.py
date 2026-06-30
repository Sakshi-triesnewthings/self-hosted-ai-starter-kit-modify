from ollama import Client as OllamaClient
from qdrant_client import QdrantClient

from config import settings


class RAGPipeline:
    def __init__(self):
        self.ollama = OllamaClient(host=settings.ollama_host)
        self.qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection = settings.qdrant_collection
        self.embedding_model = settings.embedding_model
        self.chat_model = settings.chat_model
        self.top_k = settings.top_k

    def check_ollama(self) -> bool:
        """Lightweight reachability probe for Ollama. Returns False on any error."""
        try:
            self.ollama.list()
            return True
        except Exception:
            return False

    def check_qdrant(self) -> bool:
        """Lightweight reachability probe for Qdrant. Returns False on any error."""
        try:
            self.qdrant.get_collections()
            return True
        except Exception:
            return False

    def embed_query(self, question: str) -> list[float]:
        response = self.ollama.embeddings(
            model=self.embedding_model,
            prompt=question,
        )
        return response["embedding"]

    def retrieve(self, vector: list[float], top_k: int) -> list[dict]:
        response = self.qdrant.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        
        chunks = []
        for hit in response.points:
            chunks.append({
                "text": hit.payload.get("text") or hit.payload.get("content") or "",
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k not in ("text", "content")},
            })
        return chunks

    def generate(self, question: str, chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[Source {i+1}]\n{chunk['text']}"
            for i, chunk in enumerate(chunks)
        )

        prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question.
        If the answer is not in the context, say "I don't have enough information to answer that."

        Context:
        {context}

        Question: {question}

        Answer:"""

        response = self.ollama.generate(
            model=self.chat_model,
            prompt=prompt,
            stream=False,
        )
        return response["response"]

    def query(self, question: str, top_k: int | None = None) -> dict:
        k = top_k if top_k is not None else self.top_k

        vector = self.embed_query(question)
        chunks = self.retrieve(vector, k)
        answer = self.generate(question, chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": chunks,
        }