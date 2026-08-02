import time
import collections
from typing import Optional
import torch
import numpy as np

from config import (
    SAMPLE_RATE,
    CHUNK_MS,
    CHUNK_SAMPLES,
    VAD_THRESHOLD,
    SILENCE_AFTER_SPEECH,
    MIN_SPEECH_DURATION,
    MAX_SPEECH_DURATION,
    PRE_SPEECH_PADDING_MS,
    POST_SPEECH_PADDING_MS,
)


class VAD:
    def __init__(self, threshold: float = VAD_THRESHOLD) -> None:
        self.threshold = threshold
        self._model = self._load_model()

    def _load_model(self):
        """Loads the VAD model"""
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                verbose=False,
            )
            model.eval()
            print("VAD model is ready")
        except Exception as e:
            raise RuntimeError(f"Failed to load VAD{e}")
        return model

    def _score(self, chunk_np: np.ndarray) -> float:
        """
        Scores a single 32ms audio chunk.
        Returns float 0.0 (silence) → 1.0 (speech).
        """
        tensor = torch.from_numpy(chunk_np).float().unsqueeze(0)
        with torch.no_grad():
            return self._model(tensor, SAMPLE_RATE).item()

    def is_speech(self, chunk_np: np.ndarray) -> bool:
        return self._score(chunk_np) >= self.threshold

    def reset_states(self) -> None:
        """Clears the model's internal RNN state. Call at the start of each
        new audio stream/connection — Silero carries hidden state across
        calls, so reusing the model without resetting bleeds one stream's
        context into the next."""
        self._model.reset_states()

    def _run_state_machine(self, stream) -> list:
        """
        Reads chunks from the mic stream and runs the
        WAITING->SPEAKING->SILENCE state machine.
        Returns a list of recorded numpy chunks.
        """

        pre_padding_chunks = int((PRE_SPEECH_PADDING_MS / 1000) / (CHUNK_MS / 1000))
        ring_buffer = collections.deque(maxlen=pre_padding_chunks)

        recorded_chunks: list[np.ndarray] = []  # chunks collected during speech
        WAITING = "WAITING"
        SPEAKING = "SPEAKING"
        SILENCE = "SILENCE"
        state = WAITING
        speech_start: Optional[float] = None
        silence_start: Optional[float] = None

        print("You can Speak")

        while True:
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk_np = chunk.flatten()
            speaking = self.is_speech(chunk_np)

            if state == WAITING:
                ring_buffer.append(chunk_np)
                if speaking:
                    state = SPEAKING
                    speech_start = time.time()
                    # Prepend ring buffer so we don't clip the first syllable
                    recorded_chunks = list(ring_buffer)
                    recorded_chunks.append(chunk_np)
                    print("Recording....")  # "Recording..."

            elif state == SPEAKING:
                recorded_chunks.append(chunk_np)

                # Safety cap

                if time.time() - speech_start > MAX_SPEECH_DURATION:
                    print("Max duration reached")  # "Max duration reached."
                    break

                if not speaking:
                    state = SILENCE
                    silence_start = time.time()

            elif state == SILENCE:
                recorded_chunks.append(chunk_np)

                if speaking:
                    # User started speaking again — stay in SPEAKING
                    state = SPEAKING
                    silence_start = None
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    # Enough silence — end of turn
                    break
        return recorded_chunks

    def _postprocess(self, recorded_chunks: list) -> Optional[np.ndarray]:
        """
        Validating length and appends silence padding
        Returns numpy array or None if audio is too short
        """
        if not recorded_chunks:
            return None

        audio_np = np.concatenate(recorded_chunks)
        speech_duration = len(audio_np) / SAMPLE_RATE
        if speech_duration < MIN_SPEECH_DURATION:
            print("  Too short — ignored.")
            return None

        # silence tail so Whisper doesn't hard-clip the last word
        post_pad = np.zeros(
            int(SAMPLE_RATE * POST_SPEECH_PADDING_MS / 1000), dtype=np.float32
        )
        print(f" {speech_duration:.1f}s recorded.")
        return np.concatenate([audio_np, post_pad])

    def record(self) -> Optional[np.ndarray]:
        """
        Listens to the mic and returns a float32 numpy array
        of the captured utterance, or None if nothing detected.

        CLI-only path (records from the local machine's mic via
        sounddevice) — imported lazily so importing VAD elsewhere (e.g.
        api.py, which scores browser audio over a WebSocket instead) doesn't
        require sounddevice/portaudio to be installed.
        """
        import sounddevice as sd

        print("Speak")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            chunks = self._run_state_machine(stream)

        return self._postprocess(chunks)
