"""
main.py — Entry point for SettleAI Nepali Voice Agent.

Only responsible for:
  1. Initialising all components once
  2. Running the main loop
  3. Shutting down cleanly
"""

import pygame

from config import WHISPER_MODEL, VAD_THRESHOLD, LLM_MODEL, API_KEY, SYSTEM_PROMPT
from asr import ASR
from vad import VAD
from llm import LLM
from tts import TTS


def main() -> None:
    # ── 1. INIT ──────────────────────────────
    pygame.mixer.init()

    vad = VAD(threshold=VAD_THRESHOLD)
    asr = ASR(model_name=WHISPER_MODEL)
    llm = LLM(api_key=API_KEY, model=LLM_MODEL, system_prompt=SYSTEM_PROMPT)
    tts = TTS()

    print("\n" + "=" * 50)
    print("Jay Settle Panthi ")
    print("=" * 50 + "\n")

    # ── 2. LOOP ──────────────────────────────
    turn = 1
    while True:
        print(f"─── Turn {turn} ───")

        audio = vad.record()
        if audio is None:
            continue

        text = asr.transcribe(audio)
        if not text:
            print("Nothing transcribed, try again.\n")
            continue

        print(f"You     : {text}")

        reply = llm.get_response(text)
        print(f"SettleAI: {reply}\n")

        tts.speak(reply)
        turn += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nLA hai ta SETTLE Guys")
        pygame.mixer.quit()
