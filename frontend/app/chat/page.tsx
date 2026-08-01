"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import { RotateCcw } from "lucide-react";
import { MicButton, type AppState } from "@/components/MicButton";
import { ChatBubble, type Message } from "@/components/ChatBubble";
import { AudioRecorder } from "@/lib/audio";
import { AudioQueue } from "@/lib/audioQueue";
import { BargeInDetector } from "@/lib/bargeIn";
import { streamProcessAudio, resetConversation } from "@/lib/api";

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
  const audioQueueRef = useRef(new AudioQueue());
  const bargeInRef = useRef(new BargeInDetector());
  const abortControllerRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleBargeIn = useCallback(async (preRoll: Float32Array[]) => {
    bargeInRef.current.stop();
    audioQueueRef.current.stop();
    abortControllerRef.current?.abort();
    setError(null);
    try {
      await recorderRef.current.start(setVolume, preRoll);
      setAppState("recording");
    } catch {
      setError("Microphone access denied — check browser permissions");
      setAppState("idle");
    }
  }, []);

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
        const ts = Date.now();
        const assistantId = `${ts}-a`;
        let assistantText = "";
        let bargeInStarted = false;

        const queue = audioQueueRef.current;
        queue.onDrained = () => {
          bargeInRef.current.stop();
          setAppState("idle");
          setVolume(0);
        };

        const controller = new AbortController();
        abortControllerRef.current = controller;

        for await (const evt of streamProcessAudio(blob, controller.signal)) {
          if (evt.type === "transcript") {
            setMessages((prev) => [
              ...prev,
              { id: `${ts}-u`, role: "user", text: evt.text },
              { id: assistantId, role: "assistant", text: "" },
            ]);
          } else if (evt.type === "sentence") {
            assistantText = assistantText ? `${assistantText} ${evt.text}` : evt.text;
            const textSoFar = assistantText;
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, text: textSoFar } : m))
            );
            setAppState("speaking");
            if (!bargeInStarted) {
              bargeInStarted = true;
              bargeInRef.current.start(handleBargeIn);
            }
            queue.push(evt.audio);
          } else if (evt.type === "done") {
            queue.finish();
          }
        }
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") {
          // Cancelled deliberately by a barge-in — handleBargeIn already
          // moved state to "recording", nothing else to do here.
        } else {
          setError(e instanceof Error ? e.message : "Something went wrong");
          setAppState("idle");
        }
      }
    }
  }, [appState, handleBargeIn]);

  const handleReset = useCallback(async () => {
    audioQueueRef.current.stop();
    bargeInRef.current.stop();
    abortControllerRef.current?.abort();
    setMessages([]);
    setError(null);
    setAppState("idle");
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
