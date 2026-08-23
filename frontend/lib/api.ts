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

/** Parses one NDJSON record, logging the offending text before rethrowing so
 * a malformed line is identifiable from the browser console. */
function parseEvent(line: string): StreamEvent {
  try {
    return JSON.parse(line) as StreamEvent;
  } catch (e) {
    console.error("Bad NDJSON line:", line);
    throw e;
  }
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
      // Blank lines are the server's keepalives (_KEEPALIVE in api.py), sent
      // to stop RunPod's proxy killing a connection that goes quiet while the
      // LLM/TTS work runs. They carry no payload, so skip them.
      if (line) yield parseEvent(line);
    }
  }
  buffer += decoder.decode(); // flush any trailing multi-byte character
  const tail = buffer.trim();
  if (tail) yield parseEvent(tail);
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
