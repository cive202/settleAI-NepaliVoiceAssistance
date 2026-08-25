/**
 * The offline demo reply.
 *
 * The GPU backends (ASR/TTS/embeddings) are scale-to-zero RunPod serverless
 * endpoints, so a visitor arriving cold has no live pipeline to talk to. Rather
 * than showing them an error, the landing page and /chat fall back to this
 * single pre-rendered reply — the audio is a static asset, so it needs no
 * backend at all.
 *
 * Regenerate the MP3 with `python -m scripts.make_demo_clip` from the repo root
 * (keep DEMO_TEXT_NE in sync with DEMO_TEXT there).
 */
export const DEMO_AUDIO_SRC = "/demo/settleai-nepali-demo.mp3";

export const DEMO_TEXT_NE =
  "नमस्ते, यो सेटल-एआई हो। यो एउटा नेपाली भ्वाइस असिस्टेन्ट हो। अहिले म उपलब्ध छैन।";

export const DEMO_TEXT_EN =
  "Hello, this is SettleAI. This is a Nepali voice assistant. Right now I am not available.";
