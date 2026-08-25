"use client";

import { useEffect, useRef, useState } from "react";
import { Volume2, Square } from "lucide-react";
import { DEMO_AUDIO_SRC, DEMO_TEXT_NE, DEMO_TEXT_EN } from "@/lib/demo";

/**
 * Plays the pre-rendered Nepali reply (lib/demo.ts) straight from the landing
 * page, so a first-time visitor hears the assistant speak without a mic
 * permission prompt, an account, or a live GPU backend behind the site.
 *
 * Renders the whole call-to-action block, not just its own button: `children`
 * (the primary CTA) sits beside it in a button row, and the transcript is
 * centred underneath *both*. Keeping the transcript out of that row matters —
 * it is always in the layout so playback causes no shift, and nesting it next
 * to the primary CTA would drag the pair off-centre and misalign the buttons.
 */
export function DemoPlayButton({ children }: { children?: React.ReactNode }) {
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
    <div className="flex flex-col items-center gap-5">
      <div className="flex flex-col sm:flex-row items-center gap-3">
        {children}
        <button
          onClick={toggle}
          aria-label={
            playing ? "Stop the demo clip" : "Play the Nepali demo clip"
          }
          className="group inline-flex items-center gap-2 px-7 py-4 rounded-full border border-cyan-300 bg-white hover:bg-cyan-50 text-base font-semibold text-cyan-700 transition-all duration-200 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/50 shadow-sm hover:shadow-md"
        >
          {playing ? (
            <Square className="w-4 h-4 fill-current" />
          ) : (
            <Volume2 className="w-4 h-4" />
          )}
          {playing ? "Stop" : "Hear a demo"}
        </button>
      </div>

      {/* Revealed while the clip plays so the listener can follow along. */}
      <div
        aria-live="polite"
        className={[
          "max-w-md text-center transition-opacity duration-300",
          playing ? "opacity-100" : "opacity-0",
        ].join(" ")}
      >
        <p className="text-slate-700 text-sm leading-relaxed">{DEMO_TEXT_NE}</p>
        <p className="text-slate-400 text-xs mt-1.5 italic">{DEMO_TEXT_EN}</p>
      </div>
    </div>
  );
}
