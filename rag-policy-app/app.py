"""
TechCorp Policy RAG Application
Flask-based web app with RAG pipeline using ChromaDB + HuggingFace embeddings + Groq LLM
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# RAG pipeline
from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Initialize RAG pipeline (singleton)
rag: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    global rag
    if rag is None:
        logger.info("Initializing RAG pipeline...")
        rag = RAGPipeline(
            policies_dir="policies",
            chroma_dir="chroma_db",
            collection_name="techcorp_policies",
        )
        rag.build_index()
        logger.info("RAG pipeline ready.")
    return rag


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Web chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Body: { "question": "..." }
    Returns: { "answer": "...", "citations": [...], "latency_ms": ... }
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question field is required"}), 400

    if len(question) > 1000:
        return jsonify({"error": "question must be under 1000 characters"}), 400

    try:
        start = time.time()
        pipeline = get_rag()
        result = pipeline.query(question)
        latency_ms = round((time.time() - start) * 1000, 2)

        return jsonify({
            "answer": result["answer"],
            "citations": result["citations"],
            "retrieved_chunks": result["retrieved_chunks"],
            "latency_ms": latency_ms,
        })

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error. Please try again."}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "TechCorp Policy RAG",
        "version": "1.0.0",
    })


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-warm the RAG pipeline on startup
    try:
        get_rag()
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}", exc_info=True)

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
