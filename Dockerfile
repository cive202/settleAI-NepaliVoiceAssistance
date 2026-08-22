FROM python:3.12-slim

# libsndfile1: soundfile. ffmpeg: broader audio decode coverage for
# soundfile/librosa. libsdl2-*: pygame.mixer (tts.py). git: torch.hub.load
# for Silero VAD (vad.py) during the pre-warm step below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libsdl2-2.0-0 \
    libsdl2-mixer-2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchaudio only back Silero VAD (a tiny model) — pull the CPU-only
# wheel (~200MB) instead of PyPI's default CUDA-bundled one (~2GB+); every
# other package still resolves from PyPI as usual.
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# playwright the pip package is installed (rag/scraper.py imports it at
# module load), but its Chromium binary is deliberately NOT downloaded here
# to keep the image lean. /api/rag/ingest (URL scraping) will fail until you
# run `playwright install --with-deps chromium` in the image; re-ingesting
# via `python -m scripts.ingest_faq` or add_qa/add_texts doesn't need it.

# Bake in the Silero VAD model so container startup doesn't depend on a live
# GitHub fetch. trust_repo=True here avoids torch.hub's interactive trust
# prompt (which would otherwise also hang vad.py's own call at runtime, since
# it doesn't pass trust_repo — but a cached repo skips that check entirely).
RUN python -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, verbose=False, trust_repo=True)"

COPY api.py config.py llm.py perf.py tts.py vad.py ./
COPY asr/ ./asr/
COPY rag/ ./rag/
COPY slack_bot/ ./slack_bot/
COPY scripts/ ./scripts/
COPY faq.json electrical.json ./
COPY chroma_db/ ./chroma_db/

# No microphone/speaker in a container — silence pygame.mixer's audio device
# probe (tts.py only uses it for synthesize(), never speak(), in the API path).
ENV SDL_AUDIODRIVER=dummy

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
