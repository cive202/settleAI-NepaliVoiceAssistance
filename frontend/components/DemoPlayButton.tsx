"use client";

import { useEffect, useRef, useState } from "react";
import { Volume2, Square } from "lucide-react";
import { DEMO_AUDIO_SRC, DEMO_TEXT_NE, DEMO_TEXT_EN } from "@/lib/demo";

/**
 * Plays the pre-rendered Nepali reply (lib/demo.ts) straight from the landing
 * page, so a first-time visitor hears the assistant speak without a mic
 * permission prompt, an account, or a live GPU backend behind the site.
 */
export function DemoPlayButton() {
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Stop playback if the user navigates away mid-clip.
  useEffect(() => {
    const ref = audioRef;
    return () => ref.current?.pause();
  }, []);

  function toggle() {
    const existing = audioRef.current;
    if (existing) {
      existing.pause();
      audioRef.current = null;
      setPlaying(false);
      return;
    }
    // Everything is configured before the element is handed to the ref, so the
    // instance the cleanup effect sees is never mutated afterwards.
    const audio = new Audio(DEMO_AUDIO_SRC);
    const reset = () => {
      audioRef.current = null;
      setPlaying(false);
    };
    audio.onended = reset;
    audio.onerror = reset;
    audioRef.current = audio;
    audio.play().then(() => setPlaying(true), reset);
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <button
        onClick={toggle}
        aria-label={playing ? "Stop the demo clip" : "Play the Nepali demo clip"}
        className="group inline-flex items-center gap-2 px-7 py-4 rounded-full border border-cyan-500/40 bg-cyan-500/5 hover:bg-cyan-500/15 hover:border-cyan-500/70 text-base font-semibold text-cyan-300 transition-all duration-200 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/50"
      >
        {playing ? (
          <Square className="w-4 h-4 fill-current" />
        ) : (
          <Volume2 className="w-4 h-4" />
        )}
        {playing ? "Stop" : "Hear a demo"}
      </button>

      {/* Revealed while the clip plays so the listener can follow along. */}
      <div
        aria-live="polite"
        className={[
          "max-w-md text-center transition-opacity duration-300",
          playing ? "opacity-100" : "opacity-0",
        ].join(" ")}
      >
        <p className="text-white/80 text-sm leading-relaxed">{DEMO_TEXT_NE}</p>
        <p className="text-white/30 text-xs mt-1.5 italic">{DEMO_TEXT_EN}</p>
      </div>
    </div>
  );
}
