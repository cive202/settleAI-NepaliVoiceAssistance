"""
rag/store.py — Ollama embeddings + persistent Chroma vector store.
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import (
    RAG_COLLECTION_NAME,
    RAG_EMBED_MODEL,
    RAG_OLLAMA_BASE_URL,
    RAG_PERSIST_DIR,
)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=RAG_EMBED_MODEL, base_url=RAG_OLLAMA_BASE_URL)


def get_vectorstore(embeddings: OllamaEmbeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=RAG_COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=RAG_PERSIST_DIR,
    )
