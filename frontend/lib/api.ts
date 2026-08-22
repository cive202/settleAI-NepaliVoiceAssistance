export interface TranscriptEvent {
  type: "transcript";
  text: string;
}
export interface SentenceEvent {
  type: "sentence";
  text: string;
  audio: string; // base64-encoded MP3
}
export interface DoneEvent {
  type: "done";
  assistant_text: string;
}
export type StreamEvent = TranscriptEvent | SentenceEvent | DoneEvent;

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const WS_API = API.replace(/^http/, "ws");

/**
 * Parses a string that may contain multiple back-to-back JSON objects with
 * no separator between them (e.g. "{...}{...}{...}") by tracking brace
 * depth rather than assuming a single JSON value. Used as a fallback for
 * the trailing buffer content in readNdjson, where the RunPod proxy has
 * been observed to occasionally deliver multiple NDJSON lines glued
 * together without the newline separator that normally splits them.
 */
function parseConcatenatedJson(text: string): StreamEvent[] {
  const events: StreamEvent[] = [];
  let depth = 0;
  let start = -1;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (c === "}") {
      depth--;
      if (depth === 0 && start >= 0) {
        events.push(JSON.parse(text.slice(start, i + 1)) as StreamEvent);
        start = -1;
      }
    }
  }
  return events;
}

async function* readNdjson(res: Response): AsyncGenerator<StreamEvent> {
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Request failed");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newlineIdx: number;
    while ((newlineIdx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newlineIdx).trim();
      buffer = buffer.slice(newlineIdx + 1);
      if (line) {
        try {
          yield JSON.parse(line) as StreamEvent;
        } catch (e) {
          console.error("Bad NDJSON line:", line);
          throw e;
        }
      }
    }
  }
  const tail = buffer.trim();
  if (tail) {
    try {
      for (const evt of parseConcatenatedJson(tail)) yield evt;
    } catch (e) {
      console.error("Bad NDJSON tail:", tail);
      throw e;
    }
  }
}

export async function* streamProcessAudio(
  blob: Blob,
  signal?: AbortSignal
): AsyncGenerator<StreamEvent> {
  const form = new FormData();
  form.append("audio", blob, "recording.wav");
  const res = await fetch(`${API}/api/process`, { method: "POST", body: form, signal });
  yield* readNdjson(res);
}

export async function* streamText(text: string): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${API}/api/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  yield* readNdjson(res);
}

export async function resetConversation(): Promise<void> {
  await fetch(`${API}/api/reset`, { method: "POST" });
}
