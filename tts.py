"""
tts.py — Text-to-Speech with a swap-ready interface.

Responsibility:
  - Define a TTSEngine base class (easy to swap engines)
  - IndicParlerTTS: current default, ai4bharat/indic-parler-tts (Nepali + 20
    other Indic languages, voice/tone steered via a text description)
  - VitsNepaliTTS: smaller/faster local Nepali VITS model, kept as a fallback
  - GTTSEngine: network fallback implementation using gTTS
  - Audio playback via pygame

To swap engines: change one line in TTS.__init__, or call TTS.swap_engine().
"""

import io
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from functools import lru_cache

import pygame
from gtts import gTTS

from config import (
    TTS_MODEL,
    TTS_PARLER_CHARS_PER_SEC,
    TTS_PARLER_DURATION_MARGIN,
    TTS_PARLER_FRAME_RATE_HZ,
    TTS_PARLER_MIN_DURATION_S,
    TTS_SPEAKING_RATE,
    TTS_TIMEOUT_S,
    TTS_VOICE_DESCRIPTION,
    VITS_TTS_MODEL,
)
from perf import timed

log = logging.getLogger(__name__)


class TTSEngine(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        pass

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return audio bytes without playing."""
        pass

    def _play(self, data: bytes, suffix: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            tmp_path = f.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            pygame.mixer.music.unload()
            os.unlink(tmp_path)


# ── Parler-TTS implementation (current default) ──────────────────────────────
# ai4bharat/indic-parler-tts: an autoregressive Parler-TTS Mini fine-tune.
# Requires the `parler-tts` package (installed from GitHub — see
# requirements_local.txt) and its pinned transformers==4.46.1. Roughly 25x
# slower than VITS on CPU per config.py's TTS section notes; use VitsNepaliTTS
# instead if per-sentence latency matters more than voice-description control.
@lru_cache(maxsize=2)
def _load_parler(model_name: str):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading local Parler-TTS model: %s (device=%s)", model_name, device)
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    return model, tokenizer, description_tokenizer, device


def _max_new_tokens_for(text: str) -> int:
    """Bound Parler's decode length so a sample that doesn't hit EOS
    promptly can't run away toward generation_config.max_length (2610
    frames, ~30s of audio) for what should be a short sentence."""
    est_s = max(len(text) / TTS_PARLER_CHARS_PER_SEC, TTS_PARLER_MIN_DURATION_S)
    return min(int(est_s * TTS_PARLER_DURATION_MARGIN * TTS_PARLER_FRAME_RATE_HZ), 2610)


# Repeated replies (greetings especially) shouldn't re-run inference.
@lru_cache(maxsize=128)
def _synthesize_parler(model_name: str, text: str, description: str) -> bytes:
    import soundfile as sf
    import torch

    model, tokenizer, description_tokenizer, device = _load_parler(model_name)
    description_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_ids.input_ids,
            attention_mask=description_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
            max_new_tokens=_max_new_tokens_for(text),
        )
    audio = generation.cpu().numpy().squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, model.config.sampling_rate, format="WAV")
    return buf.getvalue()


class IndicParlerTTS(TTSEngine):
    def __init__(self, model_name: str = TTS_MODEL, description: str = TTS_VOICE_DESCRIPTION):
        self.model_name = model_name
        self.description = description
        _load_parler(model_name)  # warm on construction, not on first request

    def synthesize(self, text: str) -> bytes:
        with timed("tts.parler_synthesize"):
            return _synthesize_parler(self.model_name, text, self.description)

    def speak(self, text: str) -> None:
        self._play(self.synthesize(text), suffix=".wav")


# ── VITS implementation (fast fallback) ──────────────────────────────────────
# atul10/nepali_male_v1: a VITS model fine-tuned for Nepali, served locally via
# transformers so no network round-trip is needed per utterance.
@lru_cache(maxsize=2)
def _load_vits(model_name: str):
    import torch
    from transformers import AutoTokenizer, VitsModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading local VITS TTS model: %s (device=%s)", model_name, device)
    model = VitsModel.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, device


# Repeated replies (greetings especially) shouldn't re-run inference.
@lru_cache(maxsize=128)
def _synthesize_vits(model_name: str, text: str, speaking_rate: float) -> bytes:
    import soundfile as sf
    import torch

    model, tokenizer, device = _load_vits(model_name)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        waveform = model(**inputs, speaking_rate=speaking_rate).waveform[0]
    buf = io.BytesIO()
    sf.write(buf, waveform.cpu().numpy(), model.config.sampling_rate, format="WAV")
    return buf.getvalue()


class VitsNepaliTTS(TTSEngine):
    def __init__(self, model_name: str = VITS_TTS_MODEL, speaking_rate: float = TTS_SPEAKING_RATE):
        self.model_name = model_name
        self.speaking_rate = speaking_rate
        _load_vits(model_name)  # warm on construction, not on first request

    def synthesize(self, text: str) -> bytes:
        with timed("tts.vits_synthesize"):
            return _synthesize_vits(self.model_name, text, self.speaking_rate)

    def speak(self, text: str) -> None:
        self._play(self.synthesize(text), suffix=".wav")


# ── gTTS implementation (network fallback) ───────────────────────────────────
@lru_cache(maxsize=128)
def _synthesize_gtts(text: str, lang: str) -> bytes:
    buf = io.BytesIO()
    gTTS(text=text, lang=lang, timeout=TTS_TIMEOUT_S).write_to_fp(buf)
    return buf.getvalue()


class GTTSEngine(TTSEngine):
    def __init__(self, lang: str = "ne"):
        self.lang = lang

    def synthesize(self, text: str) -> bytes:
        with timed("tts.gtts_synthesize"):
            return _synthesize_gtts(text, self.lang)

    def speak(self, text: str) -> None:
        self._play(self.synthesize(text), suffix=".mp3")


# ── TTS — what main.py imports ───────────────────────────────────────────────
class TTS:
    def __init__(self):
        self._engine: TTSEngine = IndicParlerTTS()

    def speak(self, text: str) -> None:
        print("Speaking...")
        self._engine.speak(text)

    def synthesize(self, text: str) -> bytes:
        return self._engine.synthesize(text)

    def swap_engine(self, engine: TTSEngine) -> None:
        self._engine = engine
