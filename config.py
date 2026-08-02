"""
config.py — All constants in one place.
Change behaviour of the whole agent by editing only this file.
"""

import os
from dotenv import load_dotenv

load_dotenv(".env_local")

# ── API ──────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-8b-instant"
LLM_BASE_URL = "https://api.groq.com/openai/v1"

# ── ASR ──────────────────────────────────────
WHISPER_MODEL = "kiranpantha/whisper-large-v3-turbo-nepali"  # HF repo id, used by the local (transformers) ASR fallback
SAMPLE_RATE = 16000  # Hz — Whisper + Silero both require 16kHz
ASR_KEY = os.getenv("ASR_KEY")
# Hosted whisper-large-v3 is only reachable via NVIDIA's gRPC NVCF endpoint
# (no OpenAI-compatible REST route exists). Re-verify this id on the
# "View Code" tab of https://build.nvidia.com/openai/whisper-large-v3/api
# if calls start failing.
ASR_NVCF_URI = "grpc.nvcf.nvidia.com:443"
ASR_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
ASR_TIMEOUT_S = 15  # bound worst-case latency if the gRPC endpoint stalls

# ── VAD ──────────────────────────────────────
CHUNK_MS = 32  # ms per VAD chunk (must be 32ms for 16kHz Silero)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 512
VAD_THRESHOLD = 0.5  # 0.0–1.0; raise if false triggers in noisy room
SILENCE_AFTER_SPEECH = 1.2  # seconds of silence to mark end of turn
MIN_SPEECH_DURATION = 0.4  # seconds; shorter = ignored (coughs, noise)
MAX_SPEECH_DURATION = 30  # seconds; safety cap per turn
PRE_SPEECH_PADDING_MS = 300  # ms of audio kept before speech starts
POST_SPEECH_PADDING_MS = 400  # ms of silence appended after speech ends
BARGE_IN_CONSECUTIVE_FRAMES = 3  # ~96ms of continuous speech required before
# firing barge-in over /ws/barge-in (debounces single-frame flukes)

# ── LLM ──────────────────────────────────────
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 768
LLM_MAX_HISTORY_TURNS = 8  # user+assistant pairs kept, beyond which older turns are dropped
LLM_TIMEOUT_S = 15  # bound worst-case latency if the Groq endpoint stalls

SYSTEM_PROMPT = (
    "तपाईं SettleAI नामक एक सहायक हुनुहुन्छ। सधैं नेपालीमा छोटो र स्पष्ट जवाफ दिनुहोस्। "
    "तपाईंको जवाफ आवाजमा बोलिने भएकोले मार्कडाउन (तालिका, बोल्ड, बुलेट चिन्ह) प्रयोग नगर्नुहोस् "
    "— सामान्य बोलिने वाक्यहरूमा मात्र जवाफ दिनुहोस्।"
)
# "You are an assistant called SettleAI. Always reply in Nepali, briefly and clearly.
#  Your reply is spoken aloud, so don't use markdown (tables, bold, bullets) —
#  answer only in plain spoken sentences."

# ── RAG ──────────────────────────────────────
RAG_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RAG_OLLAMA_TIMEOUT_S = 60  # bound worst-case latency if the local Ollama daemon stalls;
# raised from 20s after full retrieval+generation was observed to exceed it on
# this machine's CPU for llama3.1 (retrieval alone was instant, generation was the slow part)
RAG_EMBED_MODEL = "nomic-embed-text"  # run: ollama pull nomic-embed-text
RAG_CHAT_MODEL = "llama3.1"  # run: ollama pull llama3.1
RAG_PERSIST_DIR = "chroma_db"  # on-disk Chroma persistence directory
RAG_COLLECTION_NAME = "rag_docs"
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 150
RAG_MAX_DEPTH = 2  # default recursive crawl depth
RAG_MAX_PAGES = 60  # safety cap on pages per ingest (JS rendering is slow)
RAG_RETRIEVER_K = 12  # top-k chunks retrieved per query
RAG_MAX_CONTEXT_PAGES = 5  # cap on distinct pages fully expanded into context
RAG_CONTEXT_TOKEN_BUDGET = 3000  # approx-token cap on assembled context text, so
# system prompt + history + context + response stay under Groq's free-tier
# 6000 TPM limit even when RAG_MAX_CONTEXT_PAGES pages would otherwise blow past it

# Best-match relevance score (0-1) from similarity_search_with_relevance_scores
# decides how to handle a query:
#   score < RAG_LOW_CONFIDENCE  -> confidently out-of-scope, decline normally
#   RAG_LOW_CONFIDENCE..RAG_CONFIDENT -> likely garbled/unclear, ask to repeat
#   score >= RAG_CONFIDENT      -> proceed with full retrieval as normal
# Calibrated against observed scores: a clear out-of-scope question ("capital
# of France") scored ~0.18-0.22; a heavily garbled but on-topic ASR query
# scored ~0.37-0.38; clean on-topic queries scored 0.51+.
RAG_LOW_CONFIDENCE = 0.25
RAG_CONFIDENT = 0.45
RAG_LLM_TEMPERATURE = 0.0  # factual RAG answers

# ── TTS ──────────────────────────────────────
TTS_TIMEOUT_S = 15  # bound worst-case latency if Google's TTS endpoint stalls

# ── Slack ────────────────────────────────────
# Live-ingests messages from named Slack channels into the RAG store via
# Socket Mode, so the voice agent can answer questions grounded in recent
# Slack traffic without a manual /api/rag/ingest call. See slack_bot/listener.py.
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")  # xoxb-... , needs channels:history + channels:read
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")  # xapp-... , needs connections:write (Socket Mode)
# Comma-separated channel names ("#support,#general") or IDs ("C0123,C0456")
SLACK_CHANNELS = [c.strip() for c in os.getenv("SLACK_CHANNELS", "").split(",") if c.strip()]
SLACK_ENABLED = bool(SLACK_BOT_TOKEN and SLACK_APP_TOKEN and SLACK_CHANNELS)
