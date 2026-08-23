# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /worker

# libsndfile1: soundfile, used to decode the WAV the API sends.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# transformers is pinned rather than floating: handler.py passes Whisper's
# temperature-fallback kwargs (compression_ratio_threshold, logprob_threshold,
# condition_on_prev_tokens), which were verified against this exact version.
RUN pip install --no-cache-dir \
    runpod \
    "transformers==4.46.1" \
    accelerate \
    soundfile \
    librosa

# Bake the default model into the image so cold starts don't wait on a live HF
# Hub download — same reasoning as tts_worker's Dockerfile. Both the weights
# and the base model's processor are fetched, since the Nepali fine-tune ships
# no tokenizer of its own (see handler.py's PROCESSOR_ID comment).
#
# Deliberately NOT setting HF_HUB_OFFLINE=1 (unlike tts_worker): ASR_MODEL_ID
# is meant to be swappable from RunPod's UI, and going offline would break any
# value other than the one baked here. A swapped-in model just pays a one-off
# download on its first cold start.
ARG ASR_MODEL_ID=kiranpantha/whisper-large-v3-turbo-nepali
ARG ASR_PROCESSOR_ID=openai/whisper-large-v3-turbo
RUN python -c "\
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor; \
c = AutoConfig.from_pretrained('${ASR_MODEL_ID}'); \
c.decoder_layers = 4 if 'large-v3-turbo' in '${ASR_MODEL_ID}'.lower() else c.decoder_layers; \
AutoModelForSpeechSeq2Seq.from_pretrained('${ASR_MODEL_ID}', config=c); \
AutoProcessor.from_pretrained('${ASR_PROCESSOR_ID}')"

# Built with the repo root as context (RunPod's GitHub builder always uses
# the repo root, not the Dockerfile's own directory — see git history for
# the build failure this caused), so this path is relative to root.
COPY asr_worker/handler.py .

CMD ["python", "-u", "handler.py"]
