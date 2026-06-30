"""Integration smoke test — exercises the pipeline against LIVE Ollama + Qdrant.

This is intentionally skipped when those services aren't reachable, so the
suite stays green offline / in CI. To actually run it, have Ollama and Qdrant
up (with the collection populated) and run `pytest`.
"""
import pytest

from rag_pipeline import RAGPipeline

try:
    _pipeline = RAGPipeline()
    _live = _pipeline.check_ollama() and _pipeline.check_qdrant()
except Exception:
    _pipeline = None
    _live = False

pytestmark = pytest.mark.skipif(
    not _live,
    reason="Ollama/Qdrant not reachable; skipping live integration test",
)


def test_retrieve_returns_well_formed_chunks():
    vec = _pipeline.embed_query("Which party won the West Bengal elections?")
    chunks = _pipeline.retrieve(vec, top_k=3)

    assert isinstance(chunks, list)
    for c in chunks:
        assert "text" in c
        assert "score" in c
        assert "metadata" in c


def test_query_end_to_end():
    result = _pipeline.query("Which party won the West Bengal elections?")
    assert set(result) == {"question", "answer", "sources"}
    assert isinstance(result["answer"], str) and result["answer"]
