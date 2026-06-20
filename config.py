# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000  # Hz — required by Whisper & Silero VAD
CHUNK_MS = 32  # ms per audio chunk fed to VAD (must be 32ms for 16kHz)
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # = 512 samples

# VAD tuning
VAD_THRESHOLD = 0.5  # speech confidence threshold (0.0–1.0); raise if noisy room
SILENCE_AFTER_SPEECH = 1.2  # seconds of silence before we consider the turn done
MIN_SPEECH_DURATION = 0.4  # seconds — ignore blips shorter than this
MAX_SPEECH_DURATION = 30  # seconds — safety cap (prevents infinite recording)

# Pre/post padding — keeps a small buffer so first/last word isn't clipped
PRE_SPEECH_PADDING_MS = 300  # ms of audio kept before VAD triggers
POST_SPEECH_PADDING_MS = 400  # ms of audio kept after VAD stops

WHISPER_MODEL = "base"
LLM_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = "तपाईं एक सहायक हुनुहुन्छ। सधैं नेपालीमा छोटो र स्पष्ट जवाफ दिनुहोस्।"
# Translation: "You are an assistant. Always reply in Nepali, briefly and clearly."
