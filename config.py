"""
config.py — All constants in one place.
Change behaviour of the whole agent by editing only this file.
"""

import os
from dotenv import load_dotenv

load_dotenv(".env_local")

# ── API ──────────────────────────────────────
API_KEY = os.getenv("API_KEY")
LLM_MODEL = "openai/gpt-oss-120b"
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── ASR ──────────────────────────────────────
WHISPER_MODEL = "base"  # tiny | base | small | medium | large
SAMPLE_RATE = 16000  # Hz — Whisper + Silero both require 16kHz
ASR_KEY = os.getenv("ASR_KEY")
# Hosted whisper-large-v3 is only reachable via NVIDIA's gRPC NVCF endpoint
# (no OpenAI-compatible REST route exists). Re-verify this id on the
# "View Code" tab of https://build.nvidia.com/openai/whisper-large-v3/api
# if calls start failing.
ASR_NVCF_URI = "grpc.nvcf.nvidia.com:443"
ASR_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"

# ── VAD ──────────────────────────────────────
CHUNK_MS = 32  # ms per VAD chunk (must be 32ms for 16kHz Silero)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 512
VAD_THRESHOLD = 0.5  # 0.0–1.0; raise if false triggers in noisy room
SILENCE_AFTER_SPEECH = 1.2  # seconds of silence to mark end of turn
MIN_SPEECH_DURATION = 0.4  # seconds; shorter = ignored (coughs, noise)
MAX_SPEECH_DURATION = 30  # seconds; safety cap per turn
PRE_SPEECH_PADDING_MS = 300  # ms of audio kept before speech starts
POST_SPEECH_PADDING_MS = 400  # ms of silence appended after speech ends

# ── LLM ──────────────────────────────────────
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512

SYSTEM_PROMPT = "तपाईं SettleAI नामक एक सहायक हुनुहुन्छ। सधैं नेपालीमा छोटो र स्पष्ट जवाफ दिनुहोस्।"
# "You are an assistant called SettleAI. Always reply in Nepali, briefly and clearly."
