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
    VAD_THRESHOLD,
)
from llm import LLM
from perf import timed
from rag import RAGService
from tts import GTTSEngine
from vad import VAD

_asr: ASRBackend | None = None
_llm: LLM | None = None
_tts: GTTSEngine | None = None
_rag: RAGService | None = None
_vad: VAD | None = None
_slack_listener = None
_slack_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _asr, _llm, _tts, _rag, _vad, _slack_listener, _slack_task
    print("Loading models…")
    _asr = get_asr(type="local")
    _llm = LLM(api_key=GROQ_API_KEY or "", model=LLM_MODEL, system_prompt=SYSTEM_PROMPT)
    _tts = GTTSEngine(lang="ne")
    _rag = RAGService()
    _vad = VAD(threshold=VAD_THRESHOLD)
    print("All models ready.")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
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

    Embeds `user_text` as-is (no translation) — cheap enough to run on every
    turn, and comparing Nepali-to-Nepali sidesteps the translation-quality
    issues that make raw-embedding matches unreliable against the English doc
    store (see RAGService._translate_to_english).
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


def _pop_sentence(buffer: str) -> tuple[str | None, str]:
    """Split the first complete sentence off `buffer`, or (None, buffer) if none yet."""
    match = _SENTENCE_BOUNDARY.search(buffer)
    if not match:
        return None, buffer
    sentence, rest = buffer[: match.end()].strip(), buffer[match.end() :]
    if not sentence:
        return _pop_sentence(rest)  # stray boundary char with nothing before it
    return sentence, rest


def _ndjson_line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _stream_cached_reply(user_text: str, cached: dict, transcript: str | None = None):
    """Sync generator: replays a cached answer's sentences/audio verbatim —
    no RAG retrieval, LLM generation, or TTS synthesis involved."""
    assert _llm
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
    """
    assert _llm and _tts
    if transcript is not None:
        yield _ndjson_line({"type": "transcript", "text": transcript})

    buffer = ""
    assistant_parts: list[str] = []
    sentences_for_cache: list[dict] = []
    for delta in _llm.get_response_stream(user_text, **reply_kwargs):
        buffer += delta
        assistant_parts.append(delta)
        while True:
            sentence, buffer = _pop_sentence(buffer)
            if sentence is None:
                break
            with timed("tts.synthesize"):
                audio = _tts.synthesize(sentence)
            entry = {"text": sentence, "audio": base64.b64encode(audio).decode()}
            sentences_for_cache.append(entry)
            yield _ndjson_line({"type": "sentence", **entry})

    tail = buffer.strip()
    if tail:
        with timed("tts.synthesize"):
            audio = _tts.synthesize(tail)
        entry = {"text": tail, "audio": base64.b64encode(audio).decode()}
        sentences_for_cache.append(entry)
        yield _ndjson_line({"type": "sentence", **entry})

    assistant_text = "".join(assistant_parts).strip()
    yield _ndjson_line({"type": "done", "assistant_text": assistant_text})

    if "context" in reply_kwargs and assistant_text:
        _cache_answer(user_text, assistant_text, sentences_for_cache)


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
        )

    reply_kwargs = await asyncio.to_thread(_resolve_reply_kwargs, text)
    return StreamingResponse(
        _stream_reply(text, reply_kwargs, transcript=text),
        media_type="application/x-ndjson",
    )


@app.post("/api/text")
async def process_text(body: TextBody):
    assert _llm and _tts and _rag
    cached = await asyncio.to_thread(_find_cached_answer, body.text)
    if cached is not None:
        return StreamingResponse(
            _stream_cached_reply(body.text, cached),
            media_type="application/x-ndjson",
        )

    reply_kwargs = await asyncio.to_thread(_resolve_reply_kwargs, body.text)
    return StreamingResponse(
        _stream_reply(body.text, reply_kwargs),
        media_type="application/x-ndjson",
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
