export interface ProcessResult {
  user_text: string;
  assistant_text: string;
  tts_audio: string; // base64-encoded MP3
}

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function processAudio(blob: Blob): Promise<ProcessResult> {
  const form = new FormData();
  form.append("audio", blob, "recording.wav");
  const res = await fetch(`${API}/api/process`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Processing failed");
  }
  return res.json();
}

export async function sendText(text: string): Promise<ProcessResult> {
  const res = await fetch(`${API}/api/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("Text processing failed");
  return res.json();
}

export async function resetConversation(): Promise<void> {
  await fetch(`${API}/api/reset`, { method: "POST" });
}
