"""
api.py — FastAPI HTTP server wrapping the SettleAI voice pipeline.

Endpoints:
  POST /api/process     audio file → {user_text, assistant_text, tts_audio}
  POST /api/text        text body  → {user_text, assistant_text, tts_audio}
  POST /api/reset       clear conversation history
  POST /api/rag/ingest  url → scrape, chunk, embed, store in Chroma
  POST /api/rag/qa      question + answer → store as a hand-written page in Chroma
  POST /api/rag/query   question → answer grounded in ingested context
  GET  /api/health      readiness check

/api/process and /api/text reply directly (no retrieval) to short greetings/
small talk; any other message is answered strictly from content retrieved
via the Chroma vector store ingested through /api/rag/ingest, declining if
the answer isn't in the ingested knowledge base.
"""

import asyncio
import base64
import io
import logging
import os
import re
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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from asr import get_asr
from asr.base import ASRBackend
from config import (
    API_KEY,
    LLM_MODEL,
    RAG_MAX_DEPTH,
    SAMPLE_RATE,
    SYSTEM_PROMPT,
)
from llm import LLM
from perf import timed
from rag import RAGService
from tts import GTTSEngine

_asr: ASRBackend | None = None
_llm: LLM | None = None
_tts: GTTSEngine | None = None
_rag: RAGService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _asr, _llm, _tts, _rag
    print("Loading models…")
    _asr = get_asr(type="nvidia")
    _llm = LLM(api_key=API_KEY or "", model=LLM_MODEL, system_prompt=SYSTEM_PROMPT)
    _tts = GTTSEngine(lang="ne")
    _rag = RAGService()
    print("All models ready.")
    yield


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


def _is_greeting(text: str) -> bool:
    """True only when the whole message is a short greeting/small-talk phrase."""
    normalized = re.sub(r"[^\w\sऀ-ॿ]", "", text.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in _GREETING_PHRASES


def _generate_reply(user_text: str) -> str:
    assert _llm and _rag
    if _is_greeting(user_text):
        with timed("llm.greeting"):
            return _llm.get_response(user_text)

    # Follow-ups ("since when does it exist?") carry no topic on their own —
    # fold in the prior turn so retrieval has something to match against.
    prior_turn = _llm.last_user_message()
    retrieval_query = f"{prior_turn} {user_text}" if prior_turn else user_text

    with timed("rag.retrieve_context"):
        context, _sources = _rag.retrieve_context(retrieval_query)
    if context is None:
        with timed("llm.clarify"):
            return _llm.get_response(user_text, clarify=True)
    with timed("llm.answer"):
        return _llm.get_response(user_text, context=context)


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


@app.post("/api/process")
async def process_audio(audio: UploadFile = File(...)):
    with timed("process_audio.total"):
        raw = await audio.read()
        try:
            with timed("load_audio"):
                audio_np = _load_audio(raw)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not decode audio: {exc}"
            )

        assert _asr and _llm and _tts and _rag
        try:
            with timed("asr.transcribe"):
                text = _asr.transcribe(audio_np)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ASR error: {exc}")
        if not text:
            raise HTTPException(
                status_code=422, detail="Nothing transcribed — please try again"
            )

        reply = _generate_reply(text)
        with timed("tts.synthesize"):
            tts_bytes = _tts.synthesize(reply)

    return {
        "user_text": text,
        "assistant_text": reply,
        "tts_audio": base64.b64encode(tts_bytes).decode(),
    }


@app.post("/api/text")
async def process_text(body: TextBody):
    assert _llm and _tts and _rag
    with timed("process_text.total"):
        reply = _generate_reply(body.text)
        with timed("tts.synthesize"):
            tts_bytes = _tts.synthesize(reply)
    return {
        "user_text": body.text,
        "assistant_text": reply,
        "tts_audio": base64.b64encode(tts_bytes).decode(),
    }


@app.post("/api/reset")
def reset():
    assert _llm
    _llm.reset_history()
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
    return result
