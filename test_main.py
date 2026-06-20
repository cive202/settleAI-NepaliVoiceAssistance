import os
import tempfile
import time
import collections
import whisper
import sounddevice as sd
import numpy as np
import torch
import pygame
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv

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

# ──────────────────────────────────────────────
# INIT
# ──────────────────────────────────────────────

load_dotenv(".env_local")
api_key = os.getenv("API_KEY")
if not api_key:
    raise EnvironmentError("API_KEY not found in .env_local")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

print("Whisper is loading")
asr_model = whisper.load_model(WHISPER_MODEL)
print("Whisper is ready")

print("Silero VAD is loading")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    verbose=False,
)
vad_model.eval()
print("Silero VAD is ready\n")

pygame.mixer.init()

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]


# ──────────────────────────────────────────────
# VAD HELPER
# ──────────────────────────────────────────────
def vad_probability(chunk_np: np.ndarray) -> float:
    """
    Returns speech probability (0.0–1.0) for a 512-sample chunk.
    Silero VAD expects a float32 tensor of shape (1, N).
    """
    tensor = torch.from_numpy(chunk_np).float().unsqueeze(0)
    with torch.no_grad():
        prob = vad_model(tensor, SAMPLE_RATE).item()
    return prob


# ──────────────────────────────────────────────
# STEP 1 — RECORD WITH VAD
# ──────────────────────────────────────────────
def record_with_vad() -> np.ndarray | None:
    """
    Listens to the mic continuously using Silero VAD.

    States:
      WAITING  → listening for speech to start
      SPEAKING → speech detected, recording
      SILENCE  → speech ended, waiting to confirm end-of-turn

    Returns a numpy float32 array of the captured speech,
    or None if nothing meaningful was captured.
    """

    # Ring buffer for pre-speech padding (keeps last N chunks before VAD fires)
    pre_padding_chunks = int((PRE_SPEECH_PADDING_MS / 1000) / (CHUNK_MS / 1000))
    ring_buffer = collections.deque(maxlen=pre_padding_chunks)

    recorded_chunks = []  # chunks collected during speech
    state = "WAITING"
    silence_start = None
    speech_start = None

    print("Please Speak")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
    ) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk_np = chunk.flatten()
            prob = vad_probability(chunk_np)
            is_speech = prob >= VAD_THRESHOLD

            if state == "WAITING":
                ring_buffer.append(chunk_np)
                if is_speech:
                    state = "SPEAKING"
                    speech_start = time.time()
                    # Prepend ring buffer so we don't clip the first syllable
                    recorded_chunks = list(ring_buffer)
                    recorded_chunks.append(chunk_np)
                    print("Recording")  # "Recording..."

            elif state == "SPEAKING":
                recorded_chunks.append(chunk_np)

                # Safety cap
                if time.time() - speech_start > MAX_SPEECH_DURATION:
                    print("Max duration reached")
                    break

                if not is_speech:
                    state = "SILENCE"
                    silence_start = time.time()

            elif state == "SILENCE":
                recorded_chunks.append(chunk_np)

                if is_speech:
                    # User started speaking again — stay in SPEAKING
                    state = "SPEAKING"
                    silence_start = None
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    # Enough silence — end of turn
                    break

    if not recorded_chunks:
        return None

    audio_np = np.concatenate(recorded_chunks)

    # Check minimum speech duration
    speech_duration = len(audio_np) / SAMPLE_RATE
    if speech_duration < MIN_SPEECH_DURATION:
        print("Specch to short ")  # "Speech too short."
        return None

    # Append post-speech padding (silence tail to help Whisper)
    post_padding_samples = int(SAMPLE_RATE * POST_SPEECH_PADDING_MS / 1000)
    audio_np = np.concatenate(
        [audio_np, np.zeros(post_padding_samples, dtype=np.float32)]
    )

    print(f"{speech_duration:.1f}s is recorded।")  # "X.Xs recorded."
    return audio_np


# ──────────────────────────────────────────────
# STEP 2 — ASR
# ──────────────────────────────────────────────
def transcribe_nepali(audio_np: np.ndarray) -> str:
    print(" transcribing")
    result = asr_model.transcribe(audio_np, language="ne")
    return result["text"].strip()


# ──────────────────────────────────────────────
# STEP 3 — LLM
# ──────────────────────────────────────────────
def get_response(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})
    print("getting response")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=conversation_history,
        temperature=0.7,
        top_p=1,
        max_tokens=512,
        stream=False,
    )

    assistant_text = response.choices[0].message.content.strip()

    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    if reasoning:
        print(f"\nReasoning:\n{reasoning}\n")

    conversation_history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


# ──────────────────────────────────────────────
# STEP 4 — TTS
# ──────────────────────────────────────────────
def speak_nepali(text: str) -> None:
    print("Settle_AI is talking")
    tts = gTTS(text=text, lang="ne")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    try:
        tts.save(tmp_path)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    finally:
        pygame.mixer.music.unload()
        os.unlink(tmp_path)


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def run_agent() -> None:
    # "Start speaking — auto-sends when you stop"

    turn = 1
    while True:
        print(f"─── टर्न {turn} ───")

        # 1. Record (VAD-controlled)
        audio = record_with_vad()
        if audio is None:
            print("sound is not clear\n")
            continue

        # 2. Transcribe
        user_text = transcribe_nepali(audio)
        if not user_text:
            print(" try again\n")
            continue
        print(f"You: {user_text}")

        # 3. LLM
        response_text = get_response(user_text)
        print(f"Settle_AI:{response_text}\n")

        # 4. Speak
        speak_nepali(response_text)

        turn += 1


if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n\nLanta la")
        pygame.mixer.quit()
