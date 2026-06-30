"""Endpoint tests for the FastAPI app.

The pipeline is replaced with a mock inside the lifespan, so no real Ollama or
Qdrant is needed. These verify the /health contract: 200 + "ok" when both
dependencies are reachable, 503 + "degraded" when either is down.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from main import app


def _client_with(ollama_ok: bool, qdrant_ok: bool) -> TestClient:
    """TestClient whose app builds a mock pipeline with the given health."""
    patcher = patch.object(main, "RAGPipeline")
    mock_cls = patcher.start()
    instance = mock_cls.return_value
    instance.check_ollama.return_value = ollama_ok
    instance.check_qdrant.return_value = qdrant_ok
    client = TestClient(app)
    client._patcher = patcher  # keep a handle so the test can stop it
    return client


def test_health_ok_when_both_reachable():
    client = _client_with(True, True)
    try:
        with client:
            r = client.get("/health")
    finally:
        client._patcher.stop()

    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "ollama_reachable": True, "qdrant_reachable": True}


def test_health_degraded_returns_503_when_qdrant_down():
    client = _client_with(True, False)
    try:
        with client:
            r = client.get("/health")
    finally:
        client._patcher.stop()

    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["ollama_reachable"] is True
    assert body["qdrant_reachable"] is False


def test_root_reports_config():
    client = _client_with(True, True)
    try:
        with client:
            r = client.get("/")
    finally:
        client._patcher.stop()

    assert r.status_code == 200
    assert r.json()["message"] == "RAG API is running"
