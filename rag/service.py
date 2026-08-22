"""
rag/service.py — Ties scraping, the Chroma store, and the Groq-hosted RAG
chat model together into a single RAG ingest/query interface.
"""

import uuid

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import (
    GROQ_API_KEY,
    LLM_BASE_URL,
    RAG_CHAT_MODEL,
    RAG_CONFIDENT,
    RAG_CONTEXT_TOKEN_BUDGET,
    RAG_LLM_TEMPERATURE,
    RAG_LLM_TIMEOUT_S,
    RAG_LOW_CONFIDENCE,
    RAG_MAX_CONTEXT_PAGES,
    RAG_RETRIEVER_K,
)
from perf import timed

from .scraper import scrape, split
from .store import get_embeddings, get_vectorstore

_CHARS_PER_TOKEN = 4  # rough heuristic for English/mixed-script scraped web text


def _approx_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


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
        self._llm = ChatOpenAI(
            model=RAG_CHAT_MODEL,
            base_url=LLM_BASE_URL,
            api_key=GROQ_API_KEY,
            temperature=RAG_LLM_TEMPERATURE,
            timeout=RAG_LLM_TIMEOUT_S,
        )
        self._retriever = self._store.as_retriever(search_kwargs={"k": RAG_RETRIEVER_K})
        combine_docs_chain = create_stuff_documents_chain(self._llm, _RAG_PROMPT)
        self._chain = create_retrieval_chain(self._retriever, combine_docs_chain)

    def ingest(self, url: str, max_depth: int) -> dict:
        """Scrape, chunk, embed, and store the pages found at `url`."""
        with timed("rag.scrape"):
            raw_docs = scrape(url, max_depth=max_depth)
        if not raw_docs:
            raise ValueError(f"No pages found at {url} (max_depth={max_depth})")
        with timed("rag.split"):
            chunks = split(raw_docs)
        with timed("rag.embed_and_store"):
            ids = self._store.add_documents(chunks)
        return {
            "url": url,
            "max_depth": max_depth,
            "pages_scraped": len(raw_docs),
            "chunks_indexed": len(ids),
        }

    def add_qa(self, question: str, answer: str) -> dict:
        """Store a hand-written question/answer pair as its own retrievable page.

        Bypasses scraping entirely — useful for facts that aren't on the
        source website, or to correct/supplement what scraping found. The
        pair gets a synthetic "source" so retrieve_context() (which groups
        chunks by source and reassembles the full page) treats it exactly
        like a scraped page.
        """
        return self.add_qa_batch([(question, answer)])[0]

    def add_qa_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Like add_qa, but embeds and stores every pair in one round trip.

        Ollama's embedding endpoint is called once per add_documents() call,
        not once per document, so batching a bulk FAQ import through this
        instead of looping add_qa() turns N embedding requests into 1.
        """
        docs = []
        sources = []
        for question, answer in pairs:
            source = f"manual-qa:{uuid.uuid4().hex}"
            sources.append(source)
            docs.append(
                Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={"source": source, "start_index": 0},
                )
            )
        ids = self._store.add_documents(docs) if docs else []
        return [
            {"question": q, "answer": a, "source": s, "id": i}
            for (q, a), s, i in zip(pairs, sources, ids)
        ]

    def add_texts(self, items: list[tuple[str, str]]) -> list[dict]:
        """Store arbitrary text as its own retrievable page, keyed by an explicit source.

        Unlike add_qa_batch (which wraps pairs in a "Q: ... A: ..." template),
        text is stored verbatim — for content that already reads naturally on
        its own, e.g. Slack messages (see slack_bot/listener.py).
        """
        docs = [
            Document(page_content=text, metadata={"source": source, "start_index": 0})
            for text, source in items
        ]
        ids = self._store.add_documents(docs) if docs else []
        return [{"text": t, "source": s, "id": i} for (t, s), i in zip(items, ids)]

    def embed_query(self, text: str) -> list[float]:
        """Embed `text` directly, with no translation step.

        Used for the answer-cache similarity check (api.py), not for document
        retrieval — that path needs the translation in retrieve_context() to
        match well against the English doc store, but comparing a Nepali
        question against previously-cached Nepali questions doesn't.
        """
        return self._embeddings.embed_query(text)

    def query(self, question: str) -> dict:
        """Answer `question` using only previously ingested context."""
        with timed("rag.chain_invoke"):
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
        (and may be garbled, since they typically come through ASR). Retrieval used
        to translate the query to English first — that was needed for the old
        Ollama nomic-embed-text embeddings (English-only), but bge-m3 is
        multilingual and embeds the raw Nepali query directly at comparable or
        better relevance scores (verified against RAG_LOW_CONFIDENCE/RAG_CONFIDENT
        across clear/garbled/off-topic test queries — same threshold band either
        way), for one less Groq round trip per turn.

        Top-k similarity search picks individual chunks, but a chunk boundary can
        split apart something like a full staff list. So top-k is used only to
        rank which *pages* are relevant; the RAG_MAX_CONTEXT_PAGES highest-ranked
        pages are then pulled back in full (every chunk, reassembled in original
        order) — the LLM sees whole pages instead of a partial slice of one, while
        the page cap keeps the prompt from ballooning when many pages match.
        """
        with timed("rag.similarity_search"):
            scored = self._store.similarity_search_with_relevance_scores(
                question, k=RAG_RETRIEVER_K
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

        with timed("rag.fetch_pages"):
            result = self._store.get(
                where={"source": {"$in": sources}}, include=["documents", "metadatas"]
            )
            chunks_by_source: dict[str, list[tuple[str, dict]]] = {s: [] for s in sources}
            for text, metadata in zip(result["documents"], result["metadatas"]):
                chunks_by_source[metadata.get("source", "")].append((text, metadata))

            page_texts = []
            budget = RAG_CONTEXT_TOKEN_BUDGET
            used_sources = []
            for source in sources:
                ordered = sorted(
                    chunks_by_source[source], key=lambda pair: pair[1].get("start_index", 0)
                )
                page_text = "\n".join(text for text, _ in ordered)
                page_tokens = _approx_tokens(page_text)
                if page_tokens > budget:
                    if not page_texts:
                        # Even the single highest-ranked page alone exceeds the
                        # budget — truncate it rather than return no context at all.
                        page_text = page_text[: budget * _CHARS_PER_TOKEN]
                        page_texts.append(page_text)
                        used_sources.append(source)
                    break
                page_texts.append(page_text)
                used_sources.append(source)
                budget -= page_tokens

        return "\n\n".join(page_texts), used_sources
