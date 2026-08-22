# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /worker

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    "git+https://github.com/huggingface/parler-tts.git"

# Bake the model into the image so cold starts don't depend on a live HF
# Hub download (~3.6GB, F32) — same reasoning as the Silero VAD pre-warm in
# the backend Dockerfile. ai4bharat/indic-parler-tts is a gated HF repo, so
# this needs an authenticated download; the token is mounted as a BuildKit
# secret (build-time only, never persisted in an image layer or history).
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) python -c "\
from parler_tts import ParlerTTSForConditionalGeneration; \
from transformers import AutoTokenizer; \
m = ParlerTTSForConditionalGeneration.from_pretrained('ai4bharat/indic-parler-tts'); \
AutoTokenizer.from_pretrained('ai4bharat/indic-parler-tts'); \
AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)"

# Weights are already baked in above — skip any live Hub network/auth check
# at container start so runtime doesn't need HF_TOKEN at all.
ENV HF_HUB_OFFLINE=1

COPY handler.py .

CMD ["python", "-u", "handler.py"]
