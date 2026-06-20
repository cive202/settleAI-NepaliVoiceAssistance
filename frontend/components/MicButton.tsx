"use client";

import { Mic, MicOff, Loader2 } from "lucide-react";

export type AppState = "idle" | "recording" | "processing" | "speaking";

interface Props {
  state: AppState;
  volume: number; // 0–1 RMS
  onClick: () => void;
}

export function MicButton({ state, volume, onClick }: Props) {
  const disabled = state === "processing" || state === "speaking";
  const isRecording = state === "recording";
  const ringScale = isRecording ? 1 + volume * 4 : 1;

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={isRecording ? "Stop recording" : "Start recording"}
      className={[
        "relative w-20 h-20 rounded-full flex items-center justify-center",
        "transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40",
        disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer",
        isRecording
          ? "bg-red-500 shadow-[0_0_32px_rgba(239,68,68,0.6)]"
          : "bg-cyan-500 hover:bg-cyan-400 shadow-[0_0_28px_rgba(6,182,212,0.5)] hover:shadow-[0_0_44px_rgba(6,182,212,0.7)]",
      ].join(" ")}
    >
      {/* Volume-reactive pulse ring — only visible while recording */}
      {isRecording && (
        <span
          className="absolute inset-0 rounded-full bg-red-400/40 pointer-events-none"
          style={{
            transform: `scale(${ringScale})`,
            transition: "transform 0.08s ease-out",
          }}
        />
      )}

      {state === "processing" ? (
        <Loader2 className="w-8 h-8 text-white animate-spin" />
      ) : isRecording ? (
        <MicOff className="w-8 h-8 text-white" />
      ) : (
        <Mic className="w-8 h-8 text-white" />
      )}
    </button>
  );
}
