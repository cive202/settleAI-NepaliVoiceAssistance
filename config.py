"""
config.py — All constants in one place.
Change behaviour of the whole agent by editing only this file.
"""

import os
from dotenv import load_dotenv

load_dotenv(".env_local")

# ── API ──────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "openai/gpt-oss-20b"
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
SILENCE_AFTER_SPEECH = 0.9  # seconds of silence to mark end of turn — lowered
# from 1.2s for latency, but conservatively: this needs validation against
# real speech (not something testable headlessly), and cutting a user off
# mid-thought is worse UX than the ~300ms this saves. Try 0.7-0.8s only
# after confirming 0.9s doesn't clip natural pauses in actual use.
MIN_SPEECH_DURATION = 0.4  # seconds; shorter = ignored (coughs, noise)
MAX_SPEECH_DURATION = 30  # seconds; safety cap per turn
PRE_SPEECH_PADDING_MS = 300  # ms of audio kept before speech starts
POST_SPEECH_PADDING_MS = 400  # ms of silence appended after speech ends
BARGE_IN_CONSECUTIVE_FRAMES = 3  # ~96ms of continuous speech required before
# firing barge-in over /ws/barge-in (debounces single-frame flukes)

# ── LLM ──────────────────────────────────────
# 0.3, not the old 0.7 — this is the path that actually serves users
# (llm.py), and at 0.7 the same question produced different *facts* across
# runs: "KEC को बारेमा" once answered correctly with "काठमाडौं इन्जिनियरिङ
# कलेज" and once invented "केटुहानी इन्जिनियरिङ् कलेज". For a
# retrieval-grounded FAQ assistant, stability matters more than phrasing
# variety. (RAG_LLM_TEMPERATURE below was already 0.0 for the same reason.)
LLM_TEMPERATURE = 0.3
# Lowered from 768 — a 24-sentence, 42s-to-generate answer was observed for a
# question that should've taken 2-3 sentences. 500 was tried first but
# measured truncating a real 11-person faculty list answer mid-name (spoken
# sentence phrasing costs more tokens per name than a markdown list would);
# 700 leaves headroom for that case while still cutting off runaway
# rambling well before the old cap. Every sentence here also costs a TTS
# call downstream, so token count directly multiplies voice latency.
LLM_MAX_TOKENS = 700
LLM_MAX_HISTORY_TURNS = 8  # user+assistant pairs kept, beyond which older turns are dropped
LLM_TIMEOUT_S = 15  # bound worst-case latency if the Groq endpoint stalls

SYSTEM_PROMPT = (
    "तपाईं SettleAI नामक एक सहायक हुनुहुन्छ। सधैं नेपालीमा छोटो र स्पष्ट जवाफ दिनुहोस् — "
    "सामान्यतया २–३ वाक्यमा। धेरै व्यक्ति वा वस्तुहरूको सूची दिनुपर्ने प्रश्नमा मात्र लामो जवाफ दिनुहोस्। "
    "तपाईंको जवाफ मेसिनले बोलेर सुनाउने भएकोले यी नियम अनिवार्य छन्: "
    "मार्कडाउन, बुलेट चिन्ह, नम्बरिङ, तालिका वा नयाँ लाइन कहिल्यै प्रयोग नगर्नुहोस्; "
    "सूची दिनुपर्दा पनि एउटै बग्ने वाक्यमा कमाले छुट्याएर लेख्नुहोस्; "
    "हरेक वाक्य पूर्णविराम (।) मा टुङ्ग्याउनुहोस्। "
    "सामान्य बोलिने वाक्यहरूमा मात्र जवाफ दिनुहोस्।"
)
# "You are an assistant called SettleAI. Always give short and clear answers
#  in Nepali — generally in 2-3 sentences. Only give a longer answer for
#  questions that need a list of several people or items.
#  Your reply is READ ALOUD BY A MACHINE, so these rules are mandatory:
#  never use markdown, bullets, numbering, tables or line breaks; write even
#  a list as one flowing sentence separated by commas; end every sentence
#  with a Nepali full stop (।).
#  Answer only in plain spoken sentences."
#
# The bullet/line-break ban isn't cosmetic: a bulleted list has no sentence
# terminators, so api.py's chunker can't split it — the whole list becomes a
# single huge TTS call — and the markup itself gets read out loud.

# ── RAG ──────────────────────────────────────
# Embeddings are served by a RunPod Serverless queue-based endpoint running
# runpod-workers/worker-infinity-embedding (BAAI/bge-m3, L4 GPU, scale-to-zero).
# It's a job queue, not a plain REST API — see rag/store.py's
# RunPodInfinityEmbeddings, which posts to {INFINITY_BASE_URL}/runsync.
# INFINITY_BASE_URL is the endpoint's job-API base, e.g.
#   https://api.runpod.ai/v2/<endpoint-id>   (no trailing /runsync)
RAG_EMBED_MODEL = "BAAI/bge-m3"
INFINITY_BASE_URL = os.getenv("INFINITY_BASE_URL")
INFINITY_API_KEY = os.getenv("INFINITY_API_KEY") or os.getenv("RUNPOD_API_KEY")
RAG_EMBED_TIMEOUT_S = 60  # per-HTTP-call timeout (the /run submit call, or a status poll)
RAG_EMBED_POLL_TIMEOUT_S = 120  # total time allowed (queue dispatch + GPU
# processing) for a job to finish while we poll /status — see
# RunPodInfinityEmbeddings; dispatch wait alone has been observed up to ~15s
# even against a warm worker, so this needs real headroom
# RAG chat model runs on Groq (same GROQ_API_KEY/LLM_BASE_URL/model as the main
# LLM) instead of a local Ollama daemon — avoids needing a GPU pod just for
# RAG chat. llama-3.1-8b-instant was retired from Groq's catalog; reusing
# LLM_MODEL keeps this in sync with whatever Groq model is actually live.
RAG_CHAT_MODEL = LLM_MODEL
RAG_LLM_TIMEOUT_S = 15  # bound worst-case latency if the Groq endpoint stalls
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
# Recalibrated for BAAI/bge-m3 (the old 0.25/0.45 pair was tuned against
# Ollama nomic-embed-text, whose score distribution was different). Measured
# over 14 on-topic and 10 off-topic Nepali questions against this store:
#   on-topic  0.358 - 0.563
#   off-topic 0.023 - 0.344   (top end: "how do I get a bank loan")
#   garbled   0.256 - 0.427
# The bands nearly touch, so no threshold separates them perfectly — but the
# old 0.45 sat *inside* the on-topic range and wrongly sent 6 of 14 valid
# questions ("where is KEC", "what are the fees") to the clarify path.
# CONFIDENT=0.35 answers all 14 while still leaving every off-topic question
# below it. The thin margin is tolerable because a question that sneaks
# through lands on irrelevant context, and the prompt already tells the model
# to decline when the context doesn't cover the question — whereas a false
# "please rephrase" is a dead end for the user with no recovery.
RAG_LOW_CONFIDENCE = 0.30
RAG_CONFIDENT = 0.35
RAG_LLM_TEMPERATURE = 0.0  # factual RAG answers

# ── TTS ──────────────────────────────────────
TTS_TIMEOUT_S = 15  # bound worst-case latency if Google's TTS endpoint stalls

# indic-parler-tts on a RunPod Serverless queue-based endpoint (tts_worker/),
# same job-API shape as the RAG embeddings endpoint — see tts.py's
# RunPodIndicParlerTTS (the active TTS engine, per TTS.__init__).
TTS_MODEL = "ai4bharat/indic-parler-tts"
TTS_BASE_URL = os.getenv("TTS_BASE_URL")
TTS_RUNPOD_API_KEY = os.getenv("TTS_KEY") or os.getenv("RUNPOD_API_KEY")
# Max sentences api.py will have synthesizing at once (see _stream_reply).
# Should match the TTS_MODEL endpoint's max worker count on RunPod — more
# in-flight jobs than workers just makes them queue instead of running in
# parallel.
TTS_MAX_CONCURRENT = 2
# "Amrita" is the pre-trained speaker identity this description elicits —
# must match tts_worker/handler.py's DEFAULT_DESCRIPTION so a caller that
# omits its own description gets the same voice from both places.
TTS_VOICE_DESCRIPTION = (
    "Amrita speaks with a clear, moderate pace in Nepali. The recording is "
    "of very high quality, with the speaker's voice sounding clear and very "
    "close up."
)
TTS_POLL_TIMEOUT_S = 120  # total time allowed (queue dispatch + GPU processing)
# for a job to finish while we poll /status — dispatch wait alone has been
# observed up to ~15s even against a warm worker, so this needs real headroom

# ── Slack ────────────────────────────────────
# Live-ingests messages from named Slack channels into the RAG store via
# Socket Mode, so the voice agent can answer questions grounded in recent
# Slack traffic without a manual /api/rag/ingest call. See slack_bot/listener.py.
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")  # xoxb-... , needs channels:history + channels:read
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")  # xapp-... , needs connections:write (Socket Mode)
# Comma-separated channel names ("#support,#general") or IDs ("C0123,C0456")
SLACK_CHANNELS = [c.strip() for c in os.getenv("SLACK_CHANNELS", "").split(",") if c.strip()]
SLACK_ENABLED = bool(SLACK_BOT_TOKEN and SLACK_APP_TOKEN and SLACK_CHANNELS)
