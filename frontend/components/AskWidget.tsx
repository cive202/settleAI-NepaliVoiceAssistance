"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, MicOff, Send, Volume2, X, Loader2, Sparkles } from "lucide-react";
import { FormattedText } from "./FormattedText";
import { AudioRecorder } from "@/lib/audio";
import { AudioQueue } from "@/lib/audioQueue";
import { streamProcessAudio, streamText, type StreamEvent } from "@/lib/api";

type Status = "idle" | "recording" | "processing" | "speaking" | "error";

interface Turn {
  id: string;
  question: string;
  answer: string;
  audioChunks: string[]; // base64 wav, one per sentence
}

const STATUS_LABEL: Record<Status, string> = {
  idle: "Ask a question, type or speak",
  recording: "Listening…",
  processing: "Thinking…",
  speaking: "Speaking…",
  error: "Something went wrong",
};

/**
 * Small embeddable Q&A widget: a nav-bar trigger that opens a floating panel
 * where a visitor can type or speak a question and get a formatted answer
 * plus spoken (TTS) playback — meant to be dropped into any site's navbar.
 */
export function AskWidget() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<Status>("idle");
  const [inputText, setInputText] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef(new AudioRecorder());
  const audioQueueRef = useRef(new AudioQueue());
  const scrollRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns, status]);

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return;
    const onPointer = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const playChunks = useCallback((chunks: string[]) => {
    const queue = audioQueueRef.current;
    queue.stop();
    queue.onDrained = () => setStatus("idle");
    for (const chunk of chunks) queue.push(chunk);
    queue.finish();
    setStatus("speaking");
  }, []);

  /** Consumes a reply stream, creating/updating a Turn as sentences arrive
   * and queuing each sentence's audio for immediate playback. */
  const runQuery = useCallback(async (knownQuestion: string | null, stream: AsyncGenerator<StreamEvent>) => {
    const id = `${Date.now()}`;
    let question = knownQuestion ?? "";
    let answer = "";
    const chunks: string[] = [];
    let turnCreated = false;

    const ensureTurn = () => {
      if (turnCreated) return;
      turnCreated = true;
      setTurns((prev) => [...prev, { id, question, answer: "", audioChunks: [] }]);
    };
    if (knownQuestion !== null) ensureTurn();

    const queue = audioQueueRef.current;
    queue.onDrained = () => setStatus("idle");

    try {
      for await (const evt of stream) {
        if (evt.type === "transcript") {
          question = evt.text;
          ensureTurn();
          setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, question } : t)));
        } else if (evt.type === "sentence") {
          answer = answer ? `${answer} ${evt.text}` : evt.text;
          chunks.push(evt.audio);
          const textSoFar = answer;
          const chunksSoFar = [...chunks];
          setTurns((prev) =>
            prev.map((t) => (t.id === id ? { ...t, answer: textSoFar, audioChunks: chunksSoFar } : t))
          );
          setStatus("speaking");
          queue.push(evt.audio);
        } else if (evt.type === "done") {
          queue.finish();
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
      setStatus("error");
    }
  }, []);

  const submitText = useCallback(async () => {
    const text = inputText.trim();
    if (!text) return;
    setInputText("");
    setError(null);
    setStatus("processing");
    await runQuery(text, streamText(text));
  }, [inputText, runQuery]);

  const toggleMic = useCallback(async () => {
    if (status === "recording") {
      setStatus("processing");
      try {
        const blob = await recorderRef.current.stop();
        await runQuery(null, streamProcessAudio(blob));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not process audio");
        setStatus("error");
      }
      return;
    }
    setError(null);
    try {
      await recorderRef.current.start();
      setStatus("recording");
    } catch {
      setError("Microphone access denied");
      setStatus("error");
    }
  }, [status, runQuery]);

  const busy = status === "processing" || status === "recording";

  return (
    <div className="relative text-white">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label="Ask KEC Sahayak"
        className={[
          "flex items-center gap-2 px-3.5 py-2 rounded-full text-sm font-medium transition-all",
          "bg-cyan-500 hover:bg-cyan-400 text-[#080d1a] shadow-[0_0_18px_rgba(6,182,212,0.45)]",
        ].join(" ")}
      >
        <Sparkles className="w-4 h-4" />
        <span className="hidden sm:inline">Ask KEC</span>
      </button>

      {open && (
        <div
          ref={panelRef}
          className="absolute right-0 top-[calc(100%+10px)] w-[340px] sm:w-[380px] max-h-[70vh] flex flex-col rounded-2xl border border-white/10 bg-[#0b1120] text-white shadow-[0_12px_48px_rgba(0,0,0,0.55)] overflow-hidden z-50"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-white/[0.03]">
            <div className="flex items-center gap-2.5">
              <div className="relative w-8 h-8 rounded-full bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                {status !== "idle" && status !== "error" && (
                  <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse ring-2 ring-[#0b1120]" />
                )}
              </div>
              <div>
                <p className="text-sm font-semibold leading-tight">KEC Sahayak</p>
                <p className="text-[11px] text-white/40 leading-tight">{STATUS_LABEL[status]}</p>
              </div>
            </div>
            <button
              onClick={() => setOpen(false)}
              aria-label="Close"
              className="text-white/40 hover:text-white transition-colors p-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Body */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-4 min-h-[160px]">
            {turns.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center gap-1.5 py-6 select-none">
                <p className="text-2xl opacity-30">नमस्ते</p>
                <p className="text-xs text-white/30 max-w-[220px]">
                  Ask about admissions, courses, or anything on the KEC site — by voice or text.
                </p>
              </div>
            )}
            {turns.map((t) => (
              <div key={t.id} className="space-y-2">
                <div className="flex justify-end">
                  <div className="max-w-[85%] px-3 py-2 rounded-xl rounded-br-sm bg-white/10 text-xs leading-relaxed">
                    {t.question}
                  </div>
                </div>
                <div className="flex justify-start">
                  <div className="max-w-[92%] px-3 py-2.5 rounded-xl rounded-bl-sm bg-cyan-500/10 border border-cyan-500/20 text-xs">
                    <FormattedText text={t.answer} />
                    <button
                      onClick={() => playChunks(t.audioChunks)}
                      className="mt-2 inline-flex items-center gap-1 text-[11px] text-cyan-400/80 hover:text-cyan-300 transition-colors"
                    >
                      <Volume2 className="w-3 h-3" />
                      Replay
                    </button>
                  </div>
                </div>
              </div>
            ))}
            {status === "processing" && (
              <div className="flex items-center gap-2 text-white/30 text-xs">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Thinking…
              </div>
            )}
            {error && <p className="text-red-400 text-[11px]">{error}</p>}
          </div>

          {/* Input row */}
          <div className="flex items-center gap-2 px-3 py-3 border-t border-white/10 bg-white/[0.02]">
            <input
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submitText()}
              placeholder="Type a question…KEC"
              disabled={busy}
              className="flex-1 bg-white/5 border border-white/10 rounded-full px-3.5 py-2 text-xs placeholder:text-white/25 focus:outline-none focus:border-cyan-500/50 disabled:opacity-50"
            />
            <button
              onClick={toggleMic}
              disabled={status === "processing"}
              aria-label={status === "recording" ? "Stop recording" : "Ask by voice"}
              className={[
                "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-colors",
                status === "recording"
                  ? "bg-red-500 shadow-[0_0_14px_rgba(239,68,68,0.6)]"
                  : "bg-white/10 hover:bg-white/15",
                status === "processing" && "opacity-50 cursor-not-allowed",
              ].join(" ")}
            >
              {status === "recording" ? (
                <MicOff className="w-3.5 h-3.5" />
              ) : (
                <Mic className="w-3.5 h-3.5" />
              )}
            </button>
            <button
              onClick={submitText}
              disabled={busy || !inputText.trim()}
              aria-label="Send"
              className="flex-shrink-0 w-8 h-8 rounded-full bg-cyan-500 hover:bg-cyan-400 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
            >
              <Send className="w-3.5 h-3.5 text-[#080d1a]" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
