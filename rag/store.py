"""
rag/store.py — RunPod-hosted Infinity (BAAI/bge-m3) embeddings + persistent
Chroma vector store.

The embedding worker is a RunPod Serverless queue-based endpoint
(runpod-workers/worker-infinity-embedding), not a plain HTTP server — every
call is a RunPod job, with vectors coming back nested in
output.data[].embedding.

Every call goes through /run (submit) + poll /status, not the blocking
/runsync — queue-dispatch wait (the gap between a job being queued and a
worker picking it up, separate from GPU processing, which is near-instant
once started) was observed exceeding /runsync's timeout even against a warm
worker. Since a timed-out /runsync doesn't hand back the job id it already
created server-side, that used to mean submitting a duplicate job on every
slow-dispatch case. /run always returns a job id in ~1-3s regardless of
dispatch latency, so there's nothing to fall back from.
"""

import time
from functools import lru_cache

import requests
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from config import (
    INFINITY_API_KEY,
    INFINITY_BASE_URL,
    RAG_COLLECTION_NAME,
    RAG_EMBED_MODEL,
    RAG_EMBED_POLL_TIMEOUT_S,
    RAG_EMBED_TIMEOUT_S,
    RAG_PERSIST_DIR,
)

_DONE_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
_POLL_INTERVAL_S = 0.5


class RunPodInfinityEmbeddings(Embeddings):
    def __init__(self, model: str = RAG_EMBED_MODEL):
        self._model = model
        self._base_url = INFINITY_BASE_URL
        # A RAGService (and this embedder) lives for the whole process, not
        # per-request — a shared Session reuses one pooled HTTPS connection
        # to RunPod instead of paying a fresh TLS handshake every call.
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {INFINITY_API_KEY}",
                "Content-Type": "application/json",
            }
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Always submit via /run and poll, rather than trying /runsync
        # first: RunPod's own logs (see tts.py's RunPodIndicParlerTTS) showed
        # queue-dispatch wait regularly exceeding 10s even against a warm,
        # idle worker — GPU processing itself is near-instant once a worker
        # picks the job up. A /runsync timeout doesn't hand back the job id
        # it already created server-side, so falling back to /run on timeout
        # ended up submitting a duplicate job on every slow-dispatch case,
        # which then queues up and makes dispatch even slower afterward.
        # /run always returns a job id in ~1-3s regardless of dispatch
        # latency, so there's nothing to fall back from.
        payload = {"input": {"model": self._model, "input": texts}}
        resp = self._session.post(f"{self._base_url}/run", json=payload, timeout=RAG_EMBED_TIMEOUT_S)
        resp.raise_for_status()
        body = self._await_completion(resp.json())
        if body["status"] == "FAILED":
            raise RuntimeError(f"RunPod embedding job failed: {body.get('error')}")
        return [item["embedding"] for item in body["output"]["data"]]

    def _await_completion(self, body: dict) -> dict:
        deadline = time.monotonic() + RAG_EMBED_POLL_TIMEOUT_S
        while body.get("status") not in _DONE_STATUSES:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"RunPod embedding job {body.get('id')} did not finish within "
                    f"{RAG_EMBED_POLL_TIMEOUT_S}s (last status: {body.get('status')})"
                )
            time.sleep(_POLL_INTERVAL_S)
            try:
                resp = self._session.get(
                    f"{self._base_url}/status/{body['id']}",
                    timeout=RAG_EMBED_TIMEOUT_S,
                )
                resp.raise_for_status()
                body = resp.json()
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
                # A transient 5xx from RunPod's own status endpoint (observed
                # in practice, see tts.py) shouldn't kill the whole request —
                # the job itself is still running server-side regardless;
                # just retry the poll. A 4xx (e.g. job genuinely gone) is a
                # real error.
                if isinstance(e, requests.exceptions.HTTPError) and (
                    e.response is None or e.response.status_code < 500
                ):
                    raise
        return body

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        # Repeated identical queries (retries, common FAQ phrasing, the
        # answer-cache similarity check re-embedding the same question a
        # user asks twice) shouldn't re-hit the RunPod endpoint — same
        # reasoning as tts.py's gTTS cache. self is part of the cache key,
        # but there's one long-lived instance per process, so this is
        # effectively a per-process cache.
        return self._cached_embed_query(text)

    @lru_cache(maxsize=256)
    def _cached_embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embeddings() -> RunPodInfinityEmbeddings:
    return RunPodInfinityEmbeddings()


def get_vectorstore(embeddings: RunPodInfinityEmbeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=RAG_COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=RAG_PERSIST_DIR,
    )
