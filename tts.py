"""
tts.py — Text-to-Speech with a swap-ready interface.

Responsibility:
  - Define a TTSEngine base class (easy to swap engines)
  - GTTSEngine: current implementation using gTTS
  - Audio playback via pygame

To swap to your custom Nepali model later:
  1. Create a new class that extends TTSEngine
  2. Implement the speak() method
  3. Change one line in main.py: tts = YourCustomTTS()
  That's it — nothing else changes.
"""

import io
import os
import tempfile
from abc import ABC, abstractmethod

import pygame
from gtts import gTTS


class TTSEngine(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        pass

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return MP3 bytes without playing."""
        pass


# ── gTTS implementation (current) ────────────────────────────────────────────
class GTTSEngine(TTSEngine):
    def __init__(self, lang: str = "ne"):
        self.lang = lang

    def synthesize(self, text: str) -> bytes:
        buf = io.BytesIO()
        gTTS(text=text, lang=self.lang).write_to_fp(buf)
        return buf.getvalue()

    def speak(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(self.synthesize(text))
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            pygame.mixer.music.unload()
            os.unlink(tmp_path)


# ── Stub for your custom Nepali model (fill in when ready) ───────────────────
class CustomNepaliTTS(TTSEngine):
    def __init__(self, model_path: str):
        raise NotImplementedError("Plug in your trained model here.")

    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    def speak(self, text: str) -> None:
        raise NotImplementedError


# ── TTS — what main.py imports ───────────────────────────────────────────────
# Change GTTSEngine() to CustomNepaliTTS("path/to/model") when ready
class TTS:
    def __init__(self):
        self._engine: TTSEngine = GTTSEngine()

    def speak(self, text: str) -> None:
        print("Speaking...")
        self._engine.speak(text)

    def synthesize(self, text: str) -> bytes:
        return self._engine.synthesize(text)

    def swap_engine(self, engine: TTSEngine) -> None:
        self._engine = engine
