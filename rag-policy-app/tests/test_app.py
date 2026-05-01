"""
Test suite for TechCorp Policy RAG Application
Tests: health endpoint, chat API validation, app import
"""

import pytest
import json
import os

# Set dummy env var before importing app
os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ── Health endpoint ─────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_returns_ok_status(client):
    resp = client.get("/health")
    data = resp.get_json()
    assert data["status"] == "ok"


def test_health_contains_service_name(client):
    resp = client.get("/health")
    data = resp.get_json()
    assert "service" in data


# ── Index page ──────────────────────────────────────────────────────────────────

def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_index_returns_html(client):
    resp = client.get("/")
    assert b"Policy Assistant" in resp.data


# ── Chat endpoint validation ─────────────────────────────────────────────────────

def test_chat_missing_question_returns_400(client):
    resp = client.post("/chat",
                       data=json.dumps({}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_chat_empty_question_returns_400(client):
    resp = client.post("/chat",
                       data=json.dumps({"question": "   "}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_chat_too_long_question_returns_400(client):
    resp = client.post("/chat",
                       data=json.dumps({"question": "x" * 1001}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_chat_no_json_body_returns_400(client):
    resp = client.post("/chat", data="not json", content_type="text/plain")
    assert resp.status_code == 400
