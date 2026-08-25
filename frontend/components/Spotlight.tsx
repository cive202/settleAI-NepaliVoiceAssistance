"use client";

import { createPortal } from "react-dom";
import { ArrowUp } from "lucide-react";

export interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

export function rectOf(el: Element | null | undefined): Rect | null {
  if (!el) return null;
  const r = el.getBoundingClientRect();
  if (r.width === 0 && r.height === 0) return null;
  return { top: r.top, left: r.left, width: r.width, height: r.height };
}

export function unionRect(a: Rect | null, b: Rect | null): Rect | null {
  if (!a) return b;
  if (!b) return a;
  const top = Math.min(a.top, b.top);
  const left = Math.min(a.left, b.left);
  const right = Math.max(a.left + a.width, b.left + b.width);
  const bottom = Math.max(a.top + a.height, b.top + b.height);
  return { top, left, width: right - left, height: bottom - top };
}

interface Props {
  /** Stays bright; everything outside it is dimmed. */
  hole: Rect | null;
  /** What the bouncing arrow points at. */
  target: Rect | null;
  label: string;
  onSkip?: () => void;
}

/**
 * Dims the whole viewport except one rectangle, and floats a bouncing arrow
 * beneath a second one.
 *
 * The dimming is a single huge spread box-shadow around the hole rather than a
 * full-screen backdrop, so the highlighted element stays fully interactive with
 * no click-through hacks. Rendered into <body> because the host navbar uses
 * backdrop-blur, which makes it the containing block for position:fixed
 * descendants — a fixed overlay nested inside it would be trapped in the header.
 */
export function Spotlight({ hole, target, label, onSkip }: Props) {
  // hole/target are only ever set from a layout measurement effect, so both are
  // null during SSR and the first client render — no hydration mismatch here.
  if (typeof document === "undefined" || !hole || !target) return null;

  const PAD = 8;

  return createPortal(
    <div className="pointer-events-none">
      <div
        className="fixed z-[100] rounded-2xl transition-all duration-300 ease-out"
        style={{
          top: hole.top - PAD,
          left: hole.left - PAD,
          width: hole.width + PAD * 2,
          height: hole.height + PAD * 2,
          boxShadow:
            "0 0 0 2px rgba(34,211,238,0.85), 0 0 24px 4px rgba(34,211,238,0.35), 0 0 0 9999px rgba(2,6,23,0.78)",
        }}
      />

      <div
        className="fixed z-[101] flex flex-col items-center gap-1 transition-all duration-300 ease-out"
        style={{
          top: target.top + target.height + 14,
          left: target.left + target.width / 2,
          transform: "translateX(-50%)",
        }}
      >
        <ArrowUp className="w-6 h-6 text-cyan-300 animate-bounce drop-shadow-[0_0_8px_rgba(34,211,238,0.85)]" />
        <span className="whitespace-nowrap rounded-full bg-cyan-400 px-3 py-1 text-xs font-semibold text-slate-900 shadow-[0_4px_18px_rgba(34,211,238,0.45)]">
          {label}
        </span>
        {onSkip && (
          <button
            onClick={onSkip}
            className="pointer-events-auto mt-0.5 text-[10px] text-white/45 hover:text-white/80 underline underline-offset-2"
          >
            skip
          </button>
        )}
      </div>
    </div>,
    document.body
  );
}
