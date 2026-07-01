"""
rag/service.py — Ties scraping, the Chroma store, and the Ollama LLM
together into a single RAG ingest/query interface.
"""

import re

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import (
    RAG_CHAT_MODEL,
    RAG_CONFIDENT,
    RAG_LLM_TEMPERATURE,
    RAG_LOW_CONFIDENCE,
    RAG_MAX_CONTEXT_PAGES,
    RAG_OLLAMA_BASE_URL,
    RAG_RETRIEVER_K,
)
from .scraper import scrape, split
from .store import get_embeddings, get_vectorstore

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only the following context. If the answer "
                   "isn't in the context, say you don't know.\n\n{context}"),
        ("human", "{input}"),
    ]
)


class RAGService:
    def __init__(self):
        self._embeddings = get_embeddings()
        self._store = get_vectorstore(self._embeddings)
        self._llm = ChatOllama(
            model=RAG_CHAT_MODEL,
            base_url=RAG_OLLAMA_BASE_URL,
            temperature=RAG_LLM_TEMPERATURE,
        )
        self._retriever = self._store.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})
        combine_docs_chain = create_stuff_documents_chain(self._llm, _RAG_PROMPT)
        self._chain = create_retrieval_chain(self._retriever, combine_docs_chain)

    def ingest(self, url: str, max_depth: int) -> dict:
        """Scrape, chunk, embed, and store the pages found at `url`."""
        raw_docs = scrape(url, max_depth=max_depth)
        if not raw_docs:
            raise ValueError(f"No pages found at {url} (max_depth={max_depth})")
        chunks = split(raw_docs)
        ids = self._store.add_documents(chunks)
        return {
            "url": url,
            "max_depth": max_depth,
            "pages_scraped": len(raw_docs),
            "chunks_indexed": len(ids),
        }

    def query(self, question: str) -> dict:
        """Answer `question` using only previously ingested context."""
        result = self._chain.invoke({"input": question})
        sources = sorted({d.metadata.get("source", "") for d in result.get("context", [])})
        return {"answer": result["answer"], "sources": sources}

    def retrieve_context(self, question: str) -> tuple[str | None, list[str]]:
        """Return (joined chunk text, source urls) for the pages matching `question`.

        context is None specifically when the best match is too weak to be a real
        topic match AND not weak enough to be confidently out-of-scope either — the
        ambiguous middle band where the question was probably just garbled beyond
        recognition (see RAG_LOW_CONFIDENCE / RAG_CONFIDENT). Callers should treat
        that as "ask the user to repeat/rephrase", as opposed to an empty-but-real
        context, which means "understood, but genuinely not in the knowledge base".

        The ingested content is in English, while voice questions are often Nepali
        (and may be garbled, since they typically come through ASR). Embedding the
        raw Nepali query directly against English documents retrieves poorly, so the
        query is translated to English first, purely for retrieval purposes.

        Top-k similarity search picks individual chunks, but a chunk boundary can
        split apart something like a full staff list. So top-k is used only to
        rank which *pages* are relevant; the RAG_MAX_CONTEXT_PAGES highest-ranked
        pages are then pulled back in full (every chunk, reassembled in original
        order) — the LLM sees whole pages instead of a partial slice of one, while
        the page cap keeps the prompt from ballooning when many pages match.
        """
        search_query = self._translate_to_english(question)
        scored = self._store.similarity_search_with_relevance_scores(
            search_query, k=RAG_RETRIEVER_K
        )
        if not scored:
            return "", []

        best_score = max(score for _doc, score in scored)
        if RAG_LOW_CONFIDENCE <= best_score < RAG_CONFIDENT:
            return None, []

        sources: list[str] = []
        for doc, _score in scored:
            source = doc.metadata.get("source", "")
            if source and source not in sources:
                sources.append(source)
        sources = sources[:RAG_MAX_CONTEXT_PAGES]

        page_texts = []
        for source in sources:
            result = self._store.get(where={"source": source}, include=["documents", "metadatas"])
            ordered = sorted(
                zip(result["documents"], result["metadatas"]),
                key=lambda pair: pair[1].get("start_index", 0),
            )
            page_texts.append("\n".join(text for text, _ in ordered))

        return "\n\n".join(page_texts), sources

    def _translate_to_english(self, text: str) -> str:
        # Asking the local model to "translate" already-English text into English
        # is a no-op it should perform but sometimes doesn't (observed: it instead
        # translates clean English INTO Nepali, presumably misreading "may be
        # Nepali" in the instructions as "should be Nepali"). Skipping the call
        # entirely for non-Devanagari input sidesteps that, and is faster too.
        if not re.search(r"[ऀ-ॿ]", text):
            return text

        # A simple, blunt "you are a translation engine" framing held up far
        # better in testing than an XML-delimited instruction — the local model
        # would sometimes "clean up" garbled Nepali instead of translating it,
        # or transliterate instead of translating, especially with the more
        # hedged delimiter-based prompt. Still strip stray Devanagari as a
        # defensive fallback in case it slips through anyway.
        prompt = (
            "You are a translation engine. Your ONLY job is to output the ENGLISH "
            "translation of the Nepali text below. Never output Devanagari script "
            "in your answer. If a word is garbled, guess the closest English word. "
            "Output format: just the English sentence, nothing else, no Devanagari "
            f"characters at all.\n\nNepali text: {text}\n\nEnglish translation:"
        )
        raw = self._llm.invoke(prompt).content.strip()
        return re.sub(r"[ऀ-ॿ]", "", raw).strip()
