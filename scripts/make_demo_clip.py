"""
scripts/make_demo_clip.py — Regenerate the demo audio clip served by the frontend.

The deployed site's GPU backends are scale-to-zero, so a visitor with no warm
worker would otherwise get silence. This bakes one reply into a static MP3 that
frontend/lib/demo.ts plays with no backend at all.

Uses the project's own TTS stack (tts.py), so the clip is genuine
indic-parler-tts output — the same voice a live turn produces — rather than a
stand-in. Falls back to gTTS only if the RunPod endpoint is unreachable, and
says loudly which engine actually produced the file.

Run from the project root: python -m scripts.make_demo_clip
"""

import sys
from pathlib import Path

from tts import GTTSEngine, RunPodIndicParlerTTS

# Keep in sync with DEMO_TEXT_NE in frontend/lib/demo.ts.
DEMO_TEXT = (
    "नमस्ते, यो सेटल-एआई हो। यो एउटा नेपाली भ्वाइस असिस्टेन्ट हो। अहिले म उपलब्ध छैन।"
)
OUT_PATH = Path("frontend/public/demo/settleai-nepali-demo.mp3")


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Text: {DEMO_TEXT}")

    # The endpoint is scale-to-zero; a cold start can take a couple of minutes
    # before the first token of audio comes back. TTS_POLL_TIMEOUT_S covers it.
    print("\n[1/2] Trying indic-parler-tts on RunPod (cold start may take ~1-2 min)...")
    try:
        audio = RunPodIndicParlerTTS().synthesize(DEMO_TEXT)
        engine = "indic-parler-tts (RunPod)"
    except Exception as e:
        print(f"      FAILED: {type(e).__name__}: {e}")
        print("\n[2/2] Falling back to gTTS...")
        try:
            audio = GTTSEngine(lang="ne").synthesize(DEMO_TEXT)
            engine = "gTTS (FALLBACK — not the real model voice)"
        except Exception as e2:
            print(f"      FAILED: {type(e2).__name__}: {e2}")
            return 1

    OUT_PATH.write_bytes(audio)
    print(f"\nWrote {OUT_PATH} ({len(audio):,} bytes)")
    print(f"Engine: {engine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
