"""
asr_worker/handler.py — RunPod Serverless worker for Whisper ASR.

Loads the model once at worker start, then serves jobs of the form:
    {"input": {"audio_base64": "<b64 wav>", "language": "ne" (optional)}}
returning:
    {"text": "<transcript>"}

Exists because the API pod can't host a Whisper checkpoint at all: its image
ships CPU-only torch (see the --extra-index-url in the root Dockerfile) and
no transformers, so asr/whisper_local.py is unreachable there. Giving ASR its
own GPU endpoint also keeps it off the same GPU as indic-parler-tts.

ASR_MODEL_ID is env-swappable — same pattern as tts_worker's TTS_MODEL_ID —
so the endpoint can be repointed at e.g. openai/whisper-large-v3 from RunPod's
UI without rebuilding the image.
"""

import base64
import io
import os

import numpy as np
import runpod
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.getenv("ASR_MODEL_ID", "kiranpantha/whisper-large-v3-turbo-nepali")
LANGUAGE = os.getenv("ASR_LANGUAGE", "ne")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
SAMPLE_RATE = 16000  # Whisper's required input rate

config = AutoConfig.from_pretrained(MODEL_ID)
# kiranpantha/whisper-large-v3-turbo-nepali ships a config.json copied from
# full whisper-large-v3 (decoder_layers=32) instead of the turbo base it is
# actually fine-tuned from (decoder_layers=4, per openai/whisper-large-v3-turbo).
# Left uncorrected, transformers stacks 28 randomly-initialised decoder layers
# on top of the real 4-layer checkpoint and generation comes out empty. Same
# correction as asr/whisper_local.py — keep the two in sync.
if "large-v3-turbo" in MODEL_ID.lower() and getattr(config, "decoder_layers", None) != 4:
    print(f"Correcting decoder_layers {config.decoder_layers} -> 4 for {MODEL_ID}")
    config.decoder_layers = 4

# That same repo ships model weights only — tokenizer.json, vocab.json et al.
# all 404. AutoProcessor.from_pretrained silently falls back to an empty
# tokenizer in that case, so generation produces ids that decode to "". Load
# the processor from the base model it was fine-tuned from instead.
PROCESSOR_ID = os.getenv("ASR_PROCESSOR_ID") or (
    "openai/whisper-large-v3-turbo" if "large-v3-turbo" in MODEL_ID.lower() else MODEL_ID
)

# Whisper's standard anti-hallucination guards. NVIDIA's hosted endpoint was
# degenerating into repetition loops ("केकेसेसेसेसेसे…", "लाप लाप लाप…") and
# Riva's RecognitionConfig exposes none of these knobs — only language_code,
# max_alternatives and enable_automatic_punctuation — so there was no way to
# rein it in. Running our own handler is what makes them reachable.
GENERATE_KWARGS = {
    "language": LANGUAGE,
    "task": "transcribe",
    # Direct block on the observed failure mode: a runaway loop is by
    # definition a short n-gram repeating without end.
    "no_repeat_ngram_size": 4,
    # Re-decode at rising temperature when a candidate trips either threshold
    # below. These are OpenAI's reference defaults for Whisper.
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,  # gzip ratio above this ⇒ repetitive
    "logprob_threshold": -1.0,  # mean logprob below this ⇒ low confidence
    "no_speech_threshold": 0.6,  # silence ⇒ emit nothing rather than invent
    # Don't feed the previous window's text back in as a prompt: that is the
    # main mechanism by which a loop, once started, sustains itself.
    "condition_on_prev_tokens": False,
}

print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE}), processor from {PROCESSOR_ID}...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, config=config).to(DEVICE, dtype=DTYPE)
processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=DTYPE,
    device=DEVICE,
)
print("Model ready.")


def handler(job):
    inp = job["input"]
    raw = base64.b64decode(inp["audio_base64"])
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # The browser recorder already sends 16kHz mono (frontend/lib/audio.ts
        # forces a 16kHz AudioContext), so this is a fallback for other callers.
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    generate_kwargs = dict(GENERATE_KWARGS)
    if inp.get("language"):
        generate_kwargs["language"] = inp["language"]

    result = pipe(
        {"array": audio.astype(np.float32), "sampling_rate": SAMPLE_RATE},
        generate_kwargs=generate_kwargs,
    )
    return {"text": result["text"].strip()}


runpod.serverless.start({"handler": handler})
