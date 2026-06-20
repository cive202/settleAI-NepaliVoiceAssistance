"""
Nepali Voice Agent — Phase 2 (VAD Pipeline)
============================================
Pipeline: Mic → Silero VAD → Whisper ASR → LLM (NVIDIA) → gTTS → Playback

"""

import os
import tempfile
import pygame
from gtts import gTTS

import config
from vad import VAD
from asr import ASR
from llm import LLM

vad = VAD(threshold=config.VAD_THRESHOLD)
asr = ASR(model_name=config.WHISPER_MODEL)
llm = LLM(
    api_key=config.API_KEY, model=config.LLM_MODEL, system_prompt=config.SYSTEM_PROMPT
)
pygame.mixer.init()

conversation_history = [{"role": "system", "content": config.SYSTEM_PROMPT}]


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
        user_text = asr.transcribe_nepali(audio)
        if not user_text:
            print("Please Speak again\n")
            continue
        print(f"You : {user_text}")

        # 3. LLM
        response_text = llm.get_response(user_text)
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
