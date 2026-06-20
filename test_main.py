"""
Nepali Voice Agent — Phase 2 (VAD Pipeline)
============================================
Pipeline: Mic → Silero VAD → Whisper ASR → LLM (NVIDIA) → gTTS → Playback

"""

import os
import tempfile
import whisper
import numpy as np
import pygame
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv

import config
from vad import VAD

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

print(" Whisper loading....")
asr_model = whisper.load_model(config.WHISPER_MODEL)
print("Whisper loaded")

vad = VAD(threshold=config.VAD_THRESHOLD)

pygame.mixer.init()

conversation_history = [{"role": "system", "content": config.SYSTEM_PROMPT}]


# ──────────────────────────────────────────────
# STEP 2 — ASR
# ──────────────────────────────────────────────
def transcribe_nepali(audio_np: np.ndarray) -> str:
    print(".....")
    result = asr_model.transcribe(audio_np, language="ne")
    return result["text"].strip()


# ──────────────────────────────────────────────
# STEP 3 — LLM
# ──────────────────────────────────────────────
def get_response(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})
    print("getting response...")

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=conversation_history,
        temperature=0.7,
        top_p=1,
        max_tokens=512,
        stream=False,
    )

    assistant_text = response.choices[0].message.content.strip()

    reasoning = getattr(response.choices[0].message, "reasoning_content", None)
    if reasoning:
        print(f"\n Reasoning:\n{reasoning}\n")

    conversation_history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


# ──────────────────────────────────────────────
# STEP 4 — TTS
# ──────────────────────────────────────────────
def speak_nepali(text: str) -> None:
    print("PLEASE SPEAK")
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
    print("=" * 55)
    print("Running_Agent")

    turn = 1
    while True:
        print(f"─── step {turn} ───")

        # 1. Record (VAD-controlled)
        audio = vad.record()
        if audio is None:
            print("Not audiable\n")
            continue

        # 2. Transcribe
        user_text = transcribe_nepali(audio)
        if not user_text:
            print("Please Speak again\n")
            continue
        print(f"You : {user_text}")

        # 3. LLM
        response_text = get_response(user_text)
        print(f"SettleAI: {response_text}\n")

        # 4. Speak
        speak_nepali(response_text)

        turn += 1


if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n\nLA hai ta SETTLE ")
        pygame.mixer.quit()
