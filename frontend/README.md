# SettleAI — frontend

Next.js (App Router) client for the Nepali voice assistant. See the
[root README](../README.md) for the full system.

```bash
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

`NEXT_PUBLIC_API_URL` is inlined at **build** time, not read at runtime — changing it
requires a rebuild.

## Routes

| Route | |
|---|---|
| `/` | Landing page, with an offline demo clip that needs no backend |
| `/chat` | Voice chat — mic capture, streamed audio playback, barge-in |
| `/kec-demo` | Mock college site showing `<AskWidget/>` embedded in a third-party page |

With no backend reachable, `/` and `/chat` fall back to a pre-rendered Nepali reply
(`lib/demo.ts`) instead of erroring.

## Layout

- `lib/audio.ts` — mic capture and WAV encoding
- `lib/audioQueue.ts` — gapless playback of streamed MP3 chunks
- `lib/bargeIn.ts` — streams mic frames to `/ws/barge-in` during playback
- `lib/api.ts` — NDJSON stream reader and health probe
- `lib/demo.ts` — the offline demo reply
