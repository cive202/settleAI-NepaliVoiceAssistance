"""
api.py — FastAPI HTTP server wrapping the SettleAI voice pipeline.

Endpoints:
  POST /api/process     audio file → newline-delimited JSON stream (see below)
  POST /api/text        text body  → newline-delimited JSON stream (see below)
  POST /api/reset       clear conversation history
  POST /api/rag/ingest  url → scrape, chunk, embed, store in Chroma
  POST /api/rag/qa      question + answer → store as a hand-written page in Chroma
  POST /api/rag/query   question → answer grounded in ingested context
  GET  /api/health      readiness check

/api/process and /api/text stream the reply as it's generated instead of
waiting for the full LLM response before synthesizing speech: the LLM output
is split into sentences as it streams in, each sentence is synthesized to
audio as soon as it's complete, and the client can start playback of the
first sentence while later ones are still being generated/synthesized. The
response body is `application/x-ndjson` — one JSON object per line:
  {"type": "transcript", "text": "..."}                    (process only, once)
  {"type": "sentence", "text": "...", "audio": "<b64 mp3>"} (one per sentence)
  {"type": "done", "assistant_text": "..."}                 (once, at the end)

They reply directly (no retrieval) to short greetings/small talk; any other
message is answered strictly from content retrieved via the Chroma vector
store ingested through /api/rag/ingest, declining if the answer isn't in the
ingested knowledge base.

When SLACK_BOT_TOKEN / SLACK_APP_TOKEN / SLACK_CHANNELS are set (config.py),
a SlackListener (slack_bot/listener.py) is started in lifespan() and streams
messages from the configured channels into that same vector store over a
Socket Mode connection, so Slack activity becomes answerable content without
a manual ingest call.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)
_log_formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
_file_handler = RotatingFileHandler("logs/app.log", maxBytes=5_000_000, backupCount=3)
_file_handler.setFormatter(_log_formatter)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[_file_handler, _console_handler])

log = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from asr import get_asr
from asr.base import ASRBackend
from config import (
    BARGE_IN_CONSECUTIVE_FRAMES,
    GROQ_API_KEY,
    LLM_MODEL,
    RAG_MAX_DEPTH,
    SAMPLE_RATE,
    SLACK_ENABLED,
    SYSTEM_PROMPT,
    TTS_MAX_CONCURRENT,
    VAD_THRESHOLD,
)
from llm import LLM
from perf import timed
from rag import RAGService
from tts import TTS
from vad import VAD

_asr: ASRBackend | None = None
_llm: LLM | None = None
_tts: TTS | None = None
_rag: RAGService | None = None
_vad: VAD | None = None
_slack_listener = None
_slack_task: asyncio.Task | None = None


async def _warm_up_runpod_endpoints() -> None:
    """Send one throwaway request to each RunPod endpoint so their workers
    are already booted by the time a real user shows up. Failures here are
    logged, not raised — a slow first real request is a fine fallback."""
    assert _rag and _tts
    try:
        await asyncio.gather(
            asyncio.to_thread(_rag.embed_query, "warm up"),
            asyncio.to_thread(_tts.synthesize, "नमस्ते"),
        )
        print("RunPod endpoints warmed up.")
    except Exception as e:
        print(f"RunPod warm-up failed (non-fatal, first real request will be slower): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _asr, _llm, _tts, _rag, _vad, _slack_listener, _slack_task
    print("Loading models…")
    _asr = get_asr(type="nvidia")
    _llm = LLM(api_key=GROQ_API_KEY or "", model=LLM_MODEL, system_prompt=SYSTEM_PROMPT)
    _tts = TTS()
    _rag = RAGService()
    _vad = VAD(threshold=VAD_THRESHOLD)
    print("All models ready.")

    # Fire warm-up calls at the RunPod embedding/TTS endpoints in the
    # background — even with FlashBoot, a worker that has never run yet
    # still pays a real cold boot the first time. Doing that here means the
    # first real user request doesn't. Fired as a background task (not
    # awaited) so server startup/readiness isn't delayed waiting for it.
    asyncio.create_task(_warm_up_runpod_endpoints())

    if SLACK_ENABLED:
        from slack_bot import SlackListener

        _slack_listener = SlackListener(_rag, on_ingest=_ANSWER_CACHE.clear)
        _slack_task = asyncio.create_task(_slack_listener.start())
        print("Slack listener starting…")
    else:
        print("Slack listener disabled (SLACK_BOT_TOKEN/SLACK_APP_TOKEN/SLACK_CHANNELS not set).")

    yield

    if _slack_listener:
        await _slack_listener.stop()
    if _slack_task:
        _slack_task.cancel()


app = FastAPI(title="SettleAI Voice API", lifespan=lifespan)

# Local dev origins are always allowed; deployed frontend origins come from
# CORS_ORIGINS (comma-separated) so the image doesn't need rebuilding when
# the frontend URL changes. Note allow_credentials=True means "*" is not a
# usable wildcard here — browsers reject it — so the real origin must be
# listed explicitly.
_CORS_ORIGINS = ["http://localhost:3000", "http://localhost:3001"] + [
    o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_GREETING_PHRASES = {
    "hi",
    "hii",
    "hello",
    "hey",
    "heya",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "how are you",
    "whats up",
    "what's up",
    "sup",
    "namaste",
    "namaskar",
    "नमस्ते",
    "नमस्कार",
    "के छ",
    "कस्तो छ",
    "हजुर",
}


def _normalize_question(text: str) -> str:
    normalized = re.sub(r"[^\w\sऀ-ॿ]", "", text.strip().lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _is_greeting(text: str) -> bool:
    """True only when the whole message is a short greeting/small-talk phrase."""
    return _normalize_question(text) in _GREETING_PHRASES


# Caches full RAG-grounded answers (text + synthesized sentence audio) so a
# repeated FAQ-style question ("what programs does KEC offer") skips
# retrieval, LLM generation, and TTS synthesis entirely. Matched by raw-text
# embedding similarity rather than exact string match — Whisper transcribes
# the "same" spoken question slightly differently every time, so exact match
# would rarely hit for voice input. Does NOT account for prior conversation
# turns folded into retrieval (see _resolve_reply_kwargs), so a follow-up
# that's semantically close to an earlier *unrelated-context* question could
# replay the wrong answer. Acceptable for this demo's mostly-single-turn FAQ
# usage; revisit if that becomes a problem.
_ANSWER_CACHE: dict[str, dict] = {}
_ANSWER_CACHE_MAX = 200
_CACHE_SIMILARITY_THRESHOLD = 0.92  # tune against real demo questions before showtime


def _find_cached_answer(user_text: str) -> dict | None:
    """Return the cached entry for the closest previously-cached question, if
    close enough by embedding cosine similarity, else None.

    Embeds `user_text` as-is (bge-m3 is multilingual, so no translation step
    is needed for either this or document retrieval — see
    RAGService.retrieve_context) — cheap enough to run on every turn.
    """
    if not _ANSWER_CACHE or _is_greeting(user_text):
        return None
    assert _rag
    with timed("cache.similarity_check"):
        query_vec = np.array(_rag.embed_query(user_text))
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None
        best_entry, best_sim = None, 0.0
        for entry in _ANSWER_CACHE.values():
            sim = float(np.dot(query_vec, entry["embedding"])) / (query_norm * entry["embedding_norm"])
            if sim > best_sim:
                best_entry, best_sim = entry, sim
    if best_entry is not None and best_sim >= _CACHE_SIMILARITY_THRESHOLD:
        log.info("cache hit (sim=%.3f): %r", best_sim, user_text)
        return best_entry
    return None


def _cache_answer(user_text: str, assistant_text: str, sentences: list[dict]) -> None:
    assert _rag
    if len(_ANSWER_CACHE) >= _ANSWER_CACHE_MAX:
        _ANSWER_CACHE.pop(next(iter(_ANSWER_CACHE)))  # evict oldest (dicts preserve insertion order)
    embedding = np.array(_rag.embed_query(user_text))
    _ANSWER_CACHE[user_text] = {
        "assistant_text": assistant_text,
        "sentences": sentences,
        "embedding": embedding,
        "embedding_norm": float(np.linalg.norm(embedding)),
    }


def _resolve_reply_kwargs(user_text: str) -> dict:
    """Decide how to answer user_text; returns kwargs for LLM.get_response_stream.

    {} for a plain greeting reply, {"clarify": True} when the question couldn't
    be confidently understood, or {"context": ...} once RAG retrieval succeeds.
    """
    assert _llm and _rag
    if _is_greeting(user_text):
        return {}

    # Follow-ups ("since when does it exist?") carry no topic on their own —
    # fold in the prior turn so retrieval has something to match against.
    prior_turn = _llm.last_user_message()
    retrieval_query = f"{prior_turn} {user_text}" if prior_turn else user_text

    with timed("rag.retrieve_context"):
        context, _sources = _rag.retrieve_context(retrieval_query)
    if context is None:
        return {"clarify": True}
    return {"context": context}


_SENTENCE_BOUNDARY = re.compile(r"[।॥.!?]")
# Clause-level (adds comma/semicolon/colon) — used only for the very first
# chunk of a reply, so TTS can start on less text instead of waiting for a
# full sentence. Later chunks use _SENTENCE_BOUNDARY for better prosody.
_CLAUSE_BOUNDARY = re.compile(r"[।॥.!?,;:]")

# A "." after any of these isn't a sentence end. Answers here are dense with
# titles ("Er. Binod Bhandari", "Dr. Lila Raj Koirala") — treating those
# periods as boundaries split names across two TTS calls, which both doubled
# synthesis cost and broke pronunciation. A faculty-list answer fragmented
# into 13 chunks instead of ~2 this way.
_ABBREVIATIONS = (
    "er dr mr mrs ms prof asst assoc sn st jr sr no vs etc bsc msc phd "
    "एर डा डा० श्री प्रा"
).split()
_ABBREV_RE = re.compile(
    r"(?:^|[\s(\[])(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")$",
    re.IGNORECASE,
)
# "45.5" / "१२.५" — a digit on both sides of the period is a decimal, not a stop.
_DECIMAL_RE = re.compile(r"[\d०-९]$")


def _is_real_boundary(buffer: str, match: re.Match[str]) -> bool:
    """False for periods that only look like sentence ends: abbreviations
    ("Er."), single-letter initials ("B."), and decimal points ("45.5")."""
    if match.group() != ".":
        return True  # ।, ॥, !, ? are unambiguous
    before = buffer[: match.start()]
    if _ABBREV_RE.search(before):
        return False
    if re.search(r"(?:^|[\s(\[])[A-Za-z]$", before):
        return False  # initial, e.g. the "B." in "Binod B. Bhandari"
    after = buffer[match.end() : match.end() + 1]
    return not (_DECIMAL_RE.search(before) and after.isdigit())


def _pop_chunk(buffer: str, boundary: re.Pattern[str]) -> tuple[str | None, str]:
    """Split the first complete chunk off `buffer` at `boundary`, or (None, buffer) if none yet."""
    pos = 0
    while True:
        match = boundary.search(buffer, pos)
        if not match:
            return None, buffer
        if _is_real_boundary(buffer, match):
            break
        pos = match.end()
    chunk, rest = buffer[: match.end()].strip(), buffer[match.end() :]
    if not chunk:
        return _pop_chunk(rest, boundary)  # stray boundary char with nothing before it
    return chunk, rest


def _pop_sentence(buffer: str) -> tuple[str | None, str]:
    """Split the first complete sentence off `buffer`, or (None, buffer) if none yet."""
    return _pop_chunk(buffer, _SENTENCE_BOUNDARY)


def _pop_first_chunk(buffer: str) -> tuple[str | None, str]:
    """Split the first complete *clause* off `buffer` — a shorter unit than a
    full sentence, so the first TTS call has less text to synthesize."""
    return _pop_chunk(buffer, _CLAUSE_BOUNDARY)


def _ndjson_line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


# Sent on every streaming response. Deployed behind RunPod's HTTP proxy
# (nginx + Cloudflare), a streamed NDJSON body arrived as HTTP 200 with the
# right content-type and *zero bytes* — the proxy buffered the whole
# response and dropped it, while plain JSON endpoints on the same pod were
# fine. X-Accel-Buffering: no is the nginx-family opt-out; the Cache-Control
# / Connection pair keeps intermediaries from buffering or coalescing too.
_STREAM_HEADERS = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
}


# Blank NDJSON lines, sent to keep the connection producing bytes while the
# LLM/TTS work happens. Deployed behind RunPod's proxy, a streaming response
# that sent headers and then went quiet for 10-30s had its connection killed
# after ~2.5s with the body discarded (HTTP 200, zero bytes, curl exit 92) —
# plain-JSON endpoints were unaffected because they send headers and body
# together. NDJSON tolerates blank lines and the frontend parser already
# skips them (`if (line)` after trim in lib/api.ts), so these are invisible
# to clients and need no frontend change.
_PRODUCER_DONE = object()  # sentinel: producer thread finished
_KEEPALIVE = b"\n"
# 1.0s, not 2.0s: the observed proxy cutoff was ~2.5s of silence, and a 2.0s
# interval measured a 2.40s worst-case gap (the interval only bounds the TTS
# wait — LLM generation before the first submit adds to it). 1.0s keeps the
# worst case near ~1.4s, a real margin rather than a coin flip.
_KEEPALIVE_INTERVAL_S = 1.0


def _stream_cached_reply(user_text: str, cached: dict, transcript: str | None = None):
    """Sync generator: replays a cached answer's sentences/audio verbatim —
    no RAG retrieval, LLM generation, or TTS synthesis involved."""
    assert _llm
    yield _KEEPALIVE  # prime the connection before any slow work
    if transcript is not None:
        yield _ndjson_line({"type": "transcript", "text": transcript})
    with timed("cache.hit"):
        for sentence in cached["sentences"]:
            yield _ndjson_line({"type": "sentence", **sentence})
        _llm.append_turn(user_text, cached["assistant_text"])
    yield _ndjson_line({"type": "done", "assistant_text": cached["assistant_text"]})


def _stream_reply(user_text: str, reply_kwargs: dict, transcript: str | None = None):
    """Sync generator: streams the LLM reply sentence-by-sentence as NDJSON bytes.

    Runs in a worker thread (via StreamingResponse's iterate_in_threadpool),
    so the blocking LLM/TTS calls inside don't block the event loop.

    TTS calls for up to TTS_MAX_CONCURRENT sentences run concurrently via a
    thread pool, overlapping with LLM token generation for later sentences —
    RunPod's own logs showed a single TTS call regularly taking 4-17s (mostly
    queue-dispatch wait, not GPU time), so synthesizing strictly one sentence
    at a time left most of that time idle instead of overlapped. Concurrency
    is capped to match the TTS endpoint's max worker count — submitting more
    in-flight jobs than there are workers to run them just makes them queue
    instead of actually running in parallel. Sentence order is still
    preserved: results are drained oldest-first regardless of which finishes
    first.
    """
    assert _llm and _tts

    def produce(out: queue.Queue) -> None:
        """Run the whole LLM+TTS pipeline, pushing finished NDJSON lines onto
        `out`. Runs in its own thread so the generator below can emit
        keepalives on a fixed schedule no matter which stage is slow."""
        buffer = ""
        assistant_parts: list[str] = []
        sentences_for_cache: list[dict] = []
        first_chunk_sent = False
        pending: list[tuple[str, Future]] = []

        def drain_oldest() -> None:
            text, fut = pending.pop(0)
            with timed("tts.synthesize"):
                audio = fut.result()
            entry = {"text": text, "audio": base64.b64encode(audio).decode()}
            sentences_for_cache.append(entry)
            out.put(_ndjson_line({"type": "sentence", **entry}))

        try:
            if transcript is not None:
                out.put(_ndjson_line({"type": "transcript", "text": transcript}))

            with ThreadPoolExecutor(max_workers=TTS_MAX_CONCURRENT) as executor:
                for delta in _llm.get_response_stream(user_text, **reply_kwargs):
                    buffer += delta
                    assistant_parts.append(delta)
                    while True:
                        pop = _pop_first_chunk if not first_chunk_sent else _pop_sentence
                        sentence, buffer = pop(buffer)
                        if sentence is None:
                            break
                        first_chunk_sent = True
                        pending.append((sentence, executor.submit(_tts.synthesize, sentence)))
                        if len(pending) >= TTS_MAX_CONCURRENT:
                            drain_oldest()

                tail = buffer.strip()
                if tail:
                    pending.append((tail, executor.submit(_tts.synthesize, tail)))

                while pending:
                    drain_oldest()

            assistant_text = "".join(assistant_parts).strip()
            out.put(_ndjson_line({"type": "done", "assistant_text": assistant_text}))

            if "context" in reply_kwargs and assistant_text:
                _cache_answer(user_text, assistant_text, sentences_for_cache)
        except BaseException as exc:  # surfaced to the consumer below
            out.put(exc)
        finally:
            out.put(_PRODUCER_DONE)

    out: queue.Queue = queue.Queue()
    threading.Thread(target=produce, args=(out,), daemon=True).start()

    yield _KEEPALIVE  # prime the connection before any slow work
    while True:
        try:
            item = out.get(timeout=_KEEPALIVE_INTERVAL_S)
        except queue.Empty:
            yield _KEEPALIVE  # bounded silence regardless of which stage is slow
            continue
        if item is _PRODUCER_DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _load_audio(raw: bytes) -> np.ndarray:
    """Return float32 mono array at 16 kHz from any soundfile-compatible bytes."""
    audio, sr = sf.read(io.BytesIO(raw))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


class TextBody(BaseModel):
    text: str


class IngestBody(BaseModel):
    url: str
    max_depth: int | None = None


class QueryBody(BaseModel):
    question: str


class QABody(BaseModel):
    question: str
    answer: str


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": _asr is not None}


@app.websocket("/ws/barge-in")
async def barge_in_ws(ws: WebSocket):
    """Scores a live stream of 512-sample (32ms @ 16kHz) float32 PCM chunks
    with Silero VAD while the client plays back TTS audio. Sends
    {"type": "barge_in"} the moment BARGE_IN_CONSECUTIVE_FRAMES consecutive
    chunks score as speech, so the client can stop playback and start
    recording the user's interruption."""
    assert _vad
    await ws.accept()
    _vad.reset_states()
    start = time.perf_counter()
    consecutive = 0
    try:
        while True:
            data = await ws.receive_bytes()
            # .copy(): frombuffer's view is read-only, which torch.from_numpy
            # (inside VAD.is_speech) warns about / doesn't support.
            chunk = np.frombuffer(data, dtype=np.float32).copy()
            is_speech = await asyncio.to_thread(_vad.is_speech, chunk)
            consecutive = consecutive + 1 if is_speech else 0
            if consecutive >= BARGE_IN_CONSECUTIVE_FRAMES:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logging.getLogger("perf").info("%-28s %8.1f ms", "barge_in.trigger", elapsed_ms)
                await ws.send_json({"type": "barge_in"})
                consecutive = 0
    except WebSocketDisconnect:
        pass


@app.post("/api/process")
async def process_audio(audio: UploadFile = File(...)):
    raw = await audio.read()
    try:
        with timed("load_audio"):
            audio_np = await asyncio.to_thread(_load_audio, raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}")

    assert _asr and _llm and _tts and _rag
    try:
        with timed("asr.transcribe"):
            text = await asyncio.to_thread(_asr.transcribe, audio_np)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASR error: {exc}")
    if not text:
        raise HTTPException(status_code=422, detail="Nothing transcribed — please try again")

    cached = await asyncio.to_thread(_find_cached_answer, text)
    if cached is not None:
        return StreamingResponse(
            _stream_cached_reply(text, cached, transcript=text),
            media_type="application/x-ndjson",
            headers=_STREAM_HEADERS,
        )

    reply_kwargs = await asyncio.to_thread(_resolve_reply_kwargs, text)
    return StreamingResponse(
        _stream_reply(text, reply_kwargs, transcript=text),
        media_type="application/x-ndjson",
        headers=_STREAM_HEADERS,
    )


@app.post("/api/text")
async def process_text(body: TextBody):
    assert _llm and _tts and _rag
    cached = await asyncio.to_thread(_find_cached_answer, body.text)
    if cached is not None:
        return StreamingResponse(
            _stream_cached_reply(body.text, cached),
            media_type="application/x-ndjson",
            headers=_STREAM_HEADERS,
        )

    reply_kwargs = await asyncio.to_thread(_resolve_reply_kwargs, body.text)
    return StreamingResponse(
        _stream_reply(body.text, reply_kwargs),
        media_type="application/x-ndjson",
        headers=_STREAM_HEADERS,
    )


@app.post("/api/reset")
def reset():
    assert _llm
    _llm.reset_history()
    return {"status": "ok"}


@app.get("/api/cache/debug")
def cache_debug():
    """Lists cached questions so you can confirm demo warmup actually landed."""
    return {
        "count": len(_ANSWER_CACHE),
        "similarity_threshold": _CACHE_SIMILARITY_THRESHOLD,
        "questions": [
            {"question": question, "answer": entry["assistant_text"]}
            for question, entry in _ANSWER_CACHE.items()
        ],
    }


@app.post("/api/cache/clear")
def cache_clear():
    _ANSWER_CACHE.clear()
    return {"status": "ok"}


@app.post("/api/rag/ingest")
async def rag_ingest(body: IngestBody):
    assert _rag
    try:
        with timed("rag_ingest.total"):
            result = await asyncio.to_thread(
                _rag.ingest, body.url, body.max_depth or RAG_MAX_DEPTH
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest error: {exc}")
    _ANSWER_CACHE.clear()
    return result


@app.post("/api/rag/query")
async def rag_query(body: QueryBody):
    assert _rag
    try:
        result = await asyncio.to_thread(_rag.query, body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query error: {exc}")
    return result


@app.post("/api/rag/qa")
async def rag_add_qa(body: QABody):
    assert _rag
    try:
        result = await asyncio.to_thread(_rag.add_qa, body.question, body.answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Add QA error: {exc}")
    _ANSWER_CACHE.clear()
    return result
