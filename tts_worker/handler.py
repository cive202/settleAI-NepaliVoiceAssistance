"""
tts_worker/handler.py — RunPod Serverless worker for ai4bharat/indic-parler-tts.

Loads the model once at worker start, then serves jobs of the form:
    {"input": {"text": "...", "description": "..." (optional)}}
returning:
    {"audio_base64": "<b64 wav>", "sample_rate": <int>}

DEFAULT_DESCRIPTION picks the "Amrita" speaker (see config.py's
TTS_VOICE_DESCRIPTION in the main app, which should match this) — callers
normally omit "description" and get that voice.
"""

import base64
import io
import os

import runpod
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_ID = os.getenv("TTS_MODEL_ID", "ai4bharat/indic-parler-tts")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# bfloat16 halves memory bandwidth vs the default F32 load — this is what
# ai4bharat's own HF Space demo uses on GPU.
#
# Do NOT pass attn_implementation="sdpa" here, despite what parler-tts's
# INFERENCE.md suggests: this model's text encoder is a T5, and transformers
# 4.46 raises "T5EncoderModel does not support an attention implementation
# through torch.nn.functional.scaled_dot_product_attention yet" at load time,
# which kills the worker on startup (every worker unhealthy, exit code 1).
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

DEFAULT_DESCRIPTION = (
    "Amrita speaks with a clear, moderate pace in Nepali. The recording is "
    "of very high quality, with the speaker's voice sounding clear and very "
    "close up."
)

print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE})...")
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE, dtype=DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
print("Model ready.")


def handler(job):
    inp = job["input"]
    text = inp["text"]
    description = inp.get("description") or DEFAULT_DESCRIPTION

    description_ids = description_tokenizer(description, return_tensors="pt").to(DEVICE)
    prompt_ids = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generation = model.generate(
            input_ids=description_ids.input_ids,
            attention_mask=description_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
    audio = generation.to(torch.float32).cpu().numpy().squeeze()

    buf = io.BytesIO()
    sf.write(buf, audio, model.config.sampling_rate, format="WAV")
    return {
        "audio_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "sample_rate": model.config.sampling_rate,
    }


runpod.serverless.start({"handler": handler})
