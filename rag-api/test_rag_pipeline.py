"""Offline unit tests for RAGPipeline.

Ollama and Qdrant are mocked, so these run without any live services — fast,
deterministic, CI-friendly. They lock down the data-shaping logic (response ->
chunks mapping, prompt assembly, orchestration) that's easy to break silently.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import rag_pipeline
from rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline():
    """A RAGPipeline whose Ollama/Qdrant clients are MagicMocks."""
    with patch.object(rag_pipeline, "OllamaClient"), patch.object(
        rag_pipeline, "QdrantClient"
    ):
        return RAGPipeline()


def _hit(payload, score):
    return SimpleNamespace(payload=payload, score=score)


def test_embed_query_returns_vector_and_calls_ollama(pipeline):
    pipeline.ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

    vec = pipeline.embed_query("hello")

    assert vec == [0.1, 0.2, 0.3]
    pipeline.ollama.embeddings.assert_called_once_with(
        model=pipeline.embedding_model, prompt="hello"
    )


def test_retrieve_maps_payload_and_falls_back_to_content(pipeline):
    pipeline.qdrant.query_points.return_value = SimpleNamespace(
        points=[
            _hit({"text": "foo", "source": "a.pdf"}, 0.9),
            _hit({"content": "bar"}, 0.8),   # no "text" -> use "content"
            _hit({"page": 7}, 0.7),          # neither -> empty string
        ]
    )

    chunks = pipeline.retrieve([0.1, 0.2], top_k=3)

    assert chunks[0] == {"text": "foo", "score": 0.9, "metadata": {"source": "a.pdf"}}
    assert chunks[1]["text"] == "bar"          # content fallback
    assert chunks[1]["metadata"] == {}         # "content" stripped from metadata
    assert chunks[2]["text"] == ""             # graceful empty
    assert chunks[2]["metadata"] == {"page": 7}

    pipeline.qdrant.query_points.assert_called_once_with(
        collection_name=pipeline.collection,
        query=[0.1, 0.2],
        limit=3,
        with_payload=True,
    )


def test_generate_builds_grounded_prompt(pipeline):
    pipeline.ollama.generate.return_value = {"response": "the answer"}

    out = pipeline.generate("what is X?", [{"text": "ctx-one"}, {"text": "ctx-two"}])

    assert out == "the answer"
    kwargs = pipeline.ollama.generate.call_args.kwargs
    assert kwargs["model"] == pipeline.chat_model
    assert kwargs["stream"] is False
    assert "ctx-one" in kwargs["prompt"] and "ctx-two" in kwargs["prompt"]
    assert "what is X?" in kwargs["prompt"]


def test_query_orchestrates_full_path(pipeline):
    pipeline.ollama.embeddings.return_value = {"embedding": [0.5]}
    pipeline.qdrant.query_points.return_value = SimpleNamespace(
        points=[_hit({"text": "ctx"}, 0.5)]
    )
    pipeline.ollama.generate.return_value = {"response": "ans"}

    result = pipeline.query("the question")

    assert result["question"] == "the question"
    assert result["answer"] == "ans"
    assert result["sources"][0]["text"] == "ctx"
    # default top_k from settings is used when none is passed
    assert pipeline.qdrant.query_points.call_args.kwargs["limit"] == pipeline.top_k


def test_query_respects_top_k_override(pipeline):
    pipeline.ollama.embeddings.return_value = {"embedding": [0.5]}
    pipeline.qdrant.query_points.return_value = SimpleNamespace(points=[])
    pipeline.ollama.generate.return_value = {"response": "ans"}

    pipeline.query("q", top_k=2)

    assert pipeline.qdrant.query_points.call_args.kwargs["limit"] == 2


def test_check_ollama_true_and_false(pipeline):
    pipeline.ollama.list.return_value = {"models": []}
    assert pipeline.check_ollama() is True

    pipeline.ollama.list.side_effect = ConnectionError("down")
    assert pipeline.check_ollama() is False


def test_check_qdrant_true_and_false(pipeline):
    pipeline.qdrant.get_collections.return_value = object()
    assert pipeline.check_qdrant() is True

    pipeline.qdrant.get_collections.side_effect = ConnectionError("down")
    assert pipeline.check_qdrant() is False
