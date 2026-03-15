"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.app import app

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "gpu_available" in data


class TestModelsEndpoint:
    def test_models_returns_200(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 4


class TestExtractEndpoint:
    def test_extract_with_valid_text(self, client):
        response = client.post(
            "/extract",
            json={
                "text": "Angela Merkel visited Berlin to discuss economic policies.",
                "include_summary": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "entities" in data
        assert "classification" in data
        assert "processing_time_ms" in data

    def test_extract_returns_entities(self, client):
        response = client.post(
            "/extract",
            json={
                "text": "Microsoft CEO met with the German chancellor in Berlin.",
                "include_summary": False,
            },
        )
        data = response.json()
        assert len(data["entities"]) > 0
        entity = data["entities"][0]
        assert "text" in entity
        assert "label" in entity
        assert "confidence" in entity

    def test_extract_returns_classification(self, client):
        response = client.post(
            "/extract",
            json={
                "text": "The stock market experienced a significant downturn.",
                "include_summary": False,
            },
        )
        data = response.json()
        classification = data["classification"]
        assert classification["label"] in ["Political", "Economic", "Sports", "Technology"]
        assert 0 <= classification["confidence"] <= 1
        assert "all_scores" in classification
