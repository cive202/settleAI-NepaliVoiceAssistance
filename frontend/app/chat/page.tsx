"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import { RotateCcw } from "lucide-react";
import { MicButton, type AppState } from "@/components/MicButton";
import { ChatBubble, type Message } from "@/components/ChatBubble";
import { AudioRecorder } from "@/lib/audio";
import { processAudio, resetConversation } from "@/lib/api";

const STATUS: Record<AppState, string> = {
  idle: "Tap to speak",
  recording: "Recording — tap to stop",
  processing: "Processing…",
  speaking: "Speaking…",
};

export default function ChatPage() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [volume, setVolume] = useState(0);
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef(new AudioRecorder());
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleMic = useCallback(async () => {
    if (appState === "idle") {
      setError(null);
      try {
        await recorderRef.current.start(setVolume);
        setAppState("recording");
      } catch {
        setError("Microphone access denied — check browser permissions");
      }
      return;
    }

    if (appState === "recording") {
      setAppState("processing");
      try {
        const blob = await recorderRef.current.stop();
        const result = await processAudio(blob);
        const ts = Date.now();
        setMessages((prev) => [
          ...prev,
          { id: `${ts}-u`, role: "user", text: result.user_text },
          { id: `${ts}-a`, role: "assistant", text: result.assistant_text },
        ]);
        setAppState("speaking");
        const audio = new Audio(`data:audio/mp3;base64,${result.tts_audio}`);
        audio.onended = () => {
          setAppState("idle");
          setVolume(0);
        };
        audio.onerror = () => setAppState("idle");
        audio.play().catch(() => setAppState("idle"));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong");
        setAppState("idle");
      }
    }
  }, [appState]);

  const handleReset = useCallback(async () => {
    setMessages([]);
    setError(null);
    try {
      await resetConversation();
    } catch {
      // silently ignore — local state is already cleared
    }
  }, []);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 flex items-center justify-between px-5 py-4 border-b border-white/10">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-full bg-cyan-500 shadow-[0_0_12px_rgba(6,182,212,0.6)]" />
          <span className="font-semibold tracking-tight">SettleAI</span>
        </Link>
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 text-xs text-white/40 hover:text-white/80 transition-colors px-3 py-1.5 rounded-full hover:bg-white/5"
        >
          <RotateCcw className="w-3 h-3" />
          Reset
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-5">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-2 select-none pointer-events-none">
            <p className="text-4xl opacity-20">नमस्ते</p>
            <p className="text-sm text-white/20">
              Tap the mic and speak in Nepali
            </p>
          </div>
        )}
        {messages.map((m) => (
          <ChatBubble key={m.id} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Controls */}
      <div className="flex-shrink-0 flex flex-col items-center gap-3 px-6 py-6 border-t border-white/10">
        {error && <p className="text-red-400 text-xs text-center">{error}</p>}
        <p className="text-white/30 text-xs tracking-wide">{STATUS[appState]}</p>
        <MicButton state={appState} volume={volume} onClick={handleMic} />
      </div>
    </div>
  );
}
