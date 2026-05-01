"""
RAG Pipeline for TechCorp Policy Assistant
- Ingestion: Parses markdown policy files
- Embedding: sentence-transformers (local, free)
- Vector Store: ChromaDB (local)
- LLM: Groq (free tier) with llama3
- Guardrails: out-of-scope detection, citation enforcement
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq

logger = logging.getLogger(__name__)

# ── Chunking config ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 400          # tokens (approx words)
CHUNK_OVERLAP = 80        # overlapping words between chunks
TOP_K = 5                 # number of chunks to retrieve
MAX_OUTPUT_TOKENS = 600   # LLM max output length

SYSTEM_PROMPT = """You are TechCorp's Policy Assistant. Your job is to answer employee questions 
ONLY using the provided policy document excerpts below.

Rules you MUST follow:
1. Only answer questions that are directly supported by the provided context.
2. If the answer is not in the provided context, respond EXACTLY with:
   "I can only answer questions about TechCorp's company policies. I couldn't find information about that in our policy documents."
3. Always cite the source document(s) using their Document ID (e.g., HR-001, FIN-001).
4. Keep answers concise and factual — maximum 200 words.
5. Never make up or infer information not present in the context.
6. Format citations as [Source: <Document ID> - <Document Title>] at the end of your answer.

Context (policy excerpts):
{context}
"""


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    1. Ingest & chunk policy markdown files
    2. Embed with sentence-transformers
    3. Store in ChromaDB
    4. Retrieve top-k chunks per query
    5. Generate answer via Groq LLM
    """

    def __init__(
        self,
        policies_dir: str = "policies",
        chroma_dir: str = "chroma_db",
        collection_name: str = "techcorp_policies",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.policies_dir = Path(policies_dir)
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name

        # Embedding model (free, runs locally)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Groq LLM client
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable not set. "
                "Get your free key at https://console.groq.com"
            )
        self.groq_client = Groq(api_key=groq_api_key)

    # ── Ingestion & Indexing ─────────────────────────────────────────────────────

    def _parse_doc_metadata(self, text: str, filename: str) -> dict:
        """Extract Document ID and title from markdown front matter."""
        doc_id_match = re.search(r"\*\*Document ID:\*\*\s*([\w-]+)", text)
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        return {
            "doc_id": doc_id_match.group(1) if doc_id_match else filename,
            "title": title_match.group(1).strip() if title_match else filename,
            "filename": filename,
        }

    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        """
        Word-window chunking with overlap.
        Splits on words to approximate token count.
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks

    def build_index(self, force_rebuild: bool = False):
        """
        Parse all policy files, chunk, embed, and store in ChromaDB.
        Skips rebuilding if collection already has documents (unless forced).
        """
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        existing_count = self.collection.count()
        if existing_count > 0 and not force_rebuild:
            logger.info(f"Collection already has {existing_count} chunks. Skipping rebuild.")
            return

        if force_rebuild:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        logger.info("Building index from policy documents...")
        all_chunks, all_embeddings, all_ids, all_metadatas = [], [], [], []

        policy_files = list(self.policies_dir.glob("*.md")) + list(self.policies_dir.glob("*.txt"))

        if not policy_files:
            raise FileNotFoundError(f"No policy files found in {self.policies_dir}")

        for fpath in policy_files:
            text = fpath.read_text(encoding="utf-8")
            meta = self._parse_doc_metadata(text, fpath.name)
            chunks = self._chunk_text(text)
            logger.info(f"  {fpath.name}: {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{meta['doc_id']}_chunk_{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "doc_id": meta["doc_id"],
                    "title": meta["title"],
                    "filename": meta["filename"],
                    "chunk_index": i,
                })

        # Batch embed
        logger.info(f"Embedding {len(all_chunks)} chunks...")
        all_embeddings = self.embedder.encode(
            all_chunks, batch_size=32, show_progress_bar=True
        ).tolist()

        # Batch upsert into ChromaDB
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            self.collection.upsert(
                ids=all_ids[i:i+batch_size],
                documents=all_chunks[i:i+batch_size],
                embeddings=all_embeddings[i:i+batch_size],
                metadatas=all_metadatas[i:i+batch_size],
            )

        logger.info(f"Index built: {len(all_chunks)} total chunks across {len(policy_files)} documents.")

    # ── Retrieval ────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """Embed query and retrieve top-k chunks from ChromaDB."""
        if not hasattr(self, "collection"):
            self.collection = self.chroma_client.get_or_create_collection(self.collection_name)

        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append({
                "text": doc,
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "filename": meta["filename"],
                "similarity": round(1 - dist, 4),  # cosine similarity
            })
        return retrieved

    # ── Generation ───────────────────────────────────────────────────────────────

    def _build_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into context string for LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Excerpt {i} — {chunk['doc_id']}: {chunk['title']}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def _extract_citations(self, chunks: list[dict]) -> list[dict]:
        """Deduplicate and format citations from retrieved chunks."""
        seen = set()
        citations = []
        for chunk in chunks:
            key = chunk["doc_id"]
            if key not in seen:
                seen.add(key)
                citations.append({
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "filename": chunk["filename"],
                })
        return citations

    def query(self, question: str) -> dict[str, Any]:
        """
        Full RAG query:
        1. Retrieve relevant chunks
        2. Build context
        3. Generate answer with Groq
        4. Return answer + citations
        """
        # Retrieve
        chunks = self.retrieve(question, k=TOP_K)

        if not chunks:
            return {
                "answer": "I can only answer questions about TechCorp's company policies. "
                          "I couldn't find any relevant policy information for your question.",
                "citations": [],
                "retrieved_chunks": [],
            }

        # Build context and prompt
        context = self._build_context(chunks)
        system_message = SYSTEM_PROMPT.format(context=context)

        # LLM call via Groq
        response = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.1,  # low temperature for factual accuracy
        )

        answer = response.choices[0].message.content.strip()
        citations = self._extract_citations(chunks)

        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": [
                {
                    "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "similarity": c["similarity"],
                }
                for c in chunks
            ],
        }
