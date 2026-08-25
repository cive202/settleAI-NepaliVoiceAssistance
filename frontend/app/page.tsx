import Link from "next/link";
import {
  Mic,
  ArrowRight,
  ChevronDown,
  Headphones,
  FileSearch,
  AudioLines,
  Code2,
  Check,
} from "lucide-react";
import { DemoPlayButton } from "@/components/DemoPlayButton";

/**
 * Shared by both "Start Talking" buttons. The transparent border is load-bearing:
 * the "Hear a demo" button beside it has a real 1px border, and without a
 * matching one this button would sit 2px shorter than its neighbour.
 */
const PRIMARY_CTA =
  "group inline-flex items-center gap-2 px-8 py-4 rounded-full border border-transparent " +
  "bg-cyan-500 hover:bg-cyan-400 text-white text-base font-semibold transition-all duration-200 " +
  "shadow-[0_8px_30px_rgba(6,182,212,0.3)] hover:shadow-[0_8px_40px_rgba(6,182,212,0.45)]";

const OFFERINGS = [
  {
    icon: Headphones,
    title: "24/7 Nepali voice support",
    desc: "Customers speak naturally in Nepali and hear a natural Nepali reply — no queue, no hold music, no business hours.",
  },
  {
    icon: FileSearch,
    title: "Answers from your own documents",
    desc: "Grounded in your FAQs, policies, and site content. When it doesn't know, it says so instead of inventing an answer.",
  },
  {
    icon: AudioLines,
    title: "Interrupt it like a person",
    desc: "Talk over the assistant and it stops mid-sentence and listens — the same barge-in you'd get from a real support agent.",
  },
  {
    icon: Code2,
    title: "Drop-in widget, one line",
    desc: "Embed it in your existing site's navbar and it just works.",
    link: { href: "/kec-demo", label: "See it live →" },
  },
];

const METRICS = [
  { value: "3.7s", label: "Spoken reply (p50)" },
  { value: "~16×", label: "Faster after our TTS deploy" },
  { value: "1ms", label: "Cached repeat answer" },
  { value: "397", label: "Turns measured" },
];

const ROADMAP = [
  {
    status: "done" as const,
    caption: "Shipped",
    title: "Fine-tuned Nepali ASR",
    desc: "Speech recognition retrained on Nepali, served from a self-hosted GPU worker.",
  },
  {
    status: "done" as const,
    caption: "Shipped",
    title: "Fine-tuned Nepali TTS",
    desc: "indic-parler-tts fine-tuned for natural Nepali speech on a dedicated endpoint.",
  },
  {
    status: "done" as const,
    caption: "Shipped",
    title: "Real-time voice pipeline",
    desc: "VAD endpointing, retrieval grounding, streaming, and barge-in — end to end.",
  },
  {
    status: "active" as const,
    caption: "In progress",
    title: "Agentic system",
    desc: "Tools and multi-step tasks: look up an order, check a status, escalate.",
  },
  {
    status: "explore" as const,
    caption: "Exploring",
    title: "Telephony / IVR + WhatsApp",
    desc: "Answer the phone line, and the channels customers already message on.",
  },
  {
    status: "explore" as const,
    caption: "Someday",
    title: "…and more to explore",
    desc: null,
  },
];

/**
 * Wave-timeline geometry, in the SVG's own pixel space. The wave is a sine of
 * period 260 and amplitude 30 around a 200 baseline, so peaks land every 260px
 * starting at 130 — one milestone per peak, with the label stem dropping onto it.
 * The canvas is a fixed pixel width and scrolls horizontally on narrow screens,
 * which keeps the HTML nodes and the SVG curve in the same coordinate system.
 */
const WAVE = {
  width: 1690,
  height: 270,
  canvasHeight: 520,
  svgTop: 250,
  peakY: 170,
  nodeX: [130, 390, 650, 910, 1170, 1430],
  /** Where "shipped/in progress" ends and the dotted, unexplored tail begins. */
  splitX: 910,
} as const;

/** Solid leg: start of the wave through the in-progress milestone. */
const WAVE_SOLID =
  "M 0,230 C 43,230 87,170 130,170 C 173,170 217,230 260,230 " +
  "C 303,230 347,170 390,170 C 433,170 477,230 520,230 " +
  "C 563,230 607,170 650,170 C 693,170 737,230 780,230 " +
  "C 823,230 867,170 910,170";

/** Dotted tail: everything still to explore, fading out to the right. */
const WAVE_TAIL =
  "M 910,170 C 953,170 997,230 1040,230 C 1083,230 1127,170 1170,170 " +
  "C 1213,170 1257,230 1300,230 C 1343,230 1387,170 1430,170 " +
  "C 1473,170 1517,230 1560,230 C 1603,230 1647,170 1690,170";

export default function LandingPage() {
  return (
    <div className="min-h-screen flex flex-col bg-white text-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-40 flex items-center justify-between px-6 py-4 bg-white/80 backdrop-blur border-b border-slate-200">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-full bg-cyan-500 shadow-[0_0_14px_rgba(6,182,212,0.35)]" />
          <span className="font-semibold tracking-tight">SettleAI</span>
        </div>
        <div className="flex items-center gap-5">
          <Link
            href="#offer"
            className="hidden sm:inline text-sm text-slate-500 hover:text-slate-900 transition-colors"
          >
            What we offer
          </Link>
          <Link
            href="#roadmap"
            className="hidden sm:inline text-sm text-slate-500 hover:text-slate-900 transition-colors"
          >
            Roadmap
          </Link>
          <Link
            href="/kec-demo"
            className="text-sm text-slate-500 hover:text-slate-900 transition-colors"
          >
            Widget Demo
          </Link>
          <Link
            href="/chat"
            className="text-sm text-slate-500 hover:text-slate-900 transition-colors"
          >
            Open App →
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section
        id="top"
        className="flex flex-col items-center justify-center px-6 text-center gap-8 py-20"
      >
        <div className="w-24 h-24 rounded-full bg-cyan-50 border border-cyan-200 flex items-center justify-center">
          <Mic className="w-10 h-10 text-cyan-600" />
        </div>

        <div className="space-y-4 max-w-2xl">
          <p className="text-cyan-600 text-xs font-semibold tracking-[0.25em] uppercase">
            B2C AI Customer Care · In Nepali
          </p>
          <h1 className="text-5xl md:text-6xl font-bold leading-tight tracking-tight font-[family-name:var(--font-devanagari)]">
            नेपाली भाषामा
            <br />
            <span className="text-cyan-600">कुरा गर्नुहोस्</span>
          </h1>
          <p className="text-slate-500 text-lg leading-relaxed">
            Your customers call, ask in their own language, and get an answer
            in seconds
            <br className="hidden md:block" />
            — spoken back in natural Nepali, around the clock.
          </p>
        </div>

        <DemoPlayButton>
          <Link href="/chat" className={PRIMARY_CTA}>
            Start Talking
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </Link>
        </DemoPlayButton>

        <Link
          href="#offer"
          aria-label="Scroll to learn more"
          className="mt-4 text-slate-300 hover:text-cyan-500 transition-colors animate-bounce"
        >
          <ChevronDown className="w-6 h-6" />
        </Link>
      </section>

      {/* What we offer */}
      <section id="offer" className="bg-slate-50 border-y border-slate-200 px-6 py-20">
        <div className="max-w-5xl mx-auto">
          <div className="text-center space-y-3 mb-12">
            <p className="text-cyan-600 text-xs font-semibold tracking-[0.25em] uppercase">
              What we offer
            </p>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
              AI customer care your customers actually want to use
            </h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {OFFERINGS.map((item) => (
              <OfferCard key={item.title} {...item} />
            ))}
          </div>
        </div>
      </section>

      {/* Numbers */}
      <section className="px-6 py-16">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            {METRICS.map((m) => (
              <div key={m.label}>
                <div className="text-3xl md:text-4xl font-bold text-cyan-600 tracking-tight">
                  {m.value}
                </div>
                <div className="text-slate-500 text-xs mt-1.5">{m.label}</div>
              </div>
            ))}
          </div>
          <p className="text-slate-400 text-xs text-center mt-8">
            Measured against warm workers over 397 real turns. Our GPU
            endpoints scale to zero, so the first request after an idle
            period pays a cold start.
          </p>
        </div>
      </section>

      {/* Roadmap */}
      <section id="roadmap" className="bg-slate-50 border-y border-slate-200 py-20">
        <div className="max-w-2xl mx-auto px-6 text-center space-y-3">
          <p className="text-cyan-600 text-xs font-semibold tracking-[0.25em] uppercase">
            Roadmap
          </p>
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
            What&apos;s shipped, what&apos;s next
          </h2>
          <p className="text-slate-500 text-sm">
            The solid stretch is live today. The dotted tail is where we&apos;re
            headed — not promised yet.
          </p>
        </div>

        <div className="overflow-x-auto mt-6">
          <ol
            className="relative mx-auto"
            style={{ width: WAVE.width, height: WAVE.canvasHeight }}
          >
            <WaveCurve />
            {ROADMAP.map((item, i) => (
              <RoadmapNode key={item.title} {...item} x={WAVE.nodeX[i]} />
            ))}
          </ol>
        </div>

        <div className="max-w-2xl mx-auto px-6 mt-6 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs text-slate-500">
          <span className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-500" />
            Shipped
          </span>
          <span className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full border-2 border-cyan-500 bg-white" />
            In progress
          </span>
          <span className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full border-2 border-dashed border-slate-400 bg-white" />
            Yet to explore
          </span>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-20 text-center">
        <h2 className="text-2xl md:text-3xl font-bold tracking-tight mb-6">
          Hear it speak Nepali for yourself
        </h2>
        <DemoPlayButton>
          <Link href="/chat" className={PRIMARY_CTA}>
            Start Talking
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </Link>
        </DemoPlayButton>
      </section>

      <footer className="text-center py-6 text-slate-400 text-xs border-t border-slate-200">
        Built with Next.js · FastAPI · Silero VAD · Whisper · Groq · indic-parler-tts
      </footer>
    </div>
  );
}

function OfferCard({
  icon: Icon,
  title,
  desc,
  link,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  desc: string;
  link?: { href: string; label: string };
}) {
  return (
    <div className="p-6 rounded-2xl border border-slate-200 bg-white shadow-sm hover:shadow-md transition-shadow text-left">
      <div className="w-10 h-10 rounded-xl bg-cyan-50 text-cyan-600 flex items-center justify-center mb-4">
        <Icon className="w-5 h-5" />
      </div>
      <h3 className="font-semibold text-base mb-1.5">{title}</h3>
      <p className="text-slate-500 text-sm leading-relaxed">{desc}</p>
      {link && (
        <Link
          href={link.href}
          className="inline-block mt-3 text-sm font-medium text-cyan-600 hover:text-cyan-700 transition-colors"
        >
          {link.label}
        </Link>
      )}
    </div>
  );
}

/**
 * The wave the milestones ride on: a soft echo curve for depth, a cyan gradient
 * for the shipped stretch (with a faded area fill under it), and a dotted,
 * fading tail for everything still unexplored.
 */
function WaveCurve() {
  return (
    <svg
      aria-hidden="true"
      width={WAVE.width}
      height={WAVE.height}
      viewBox={`0 0 ${WAVE.width} ${WAVE.height}`}
      className="absolute left-0"
      style={{ top: WAVE.svgTop }}
    >
      <defs>
        <linearGradient
          id="waveSolid"
          gradientUnits="userSpaceOnUse"
          x1="0"
          x2={WAVE.splitX}
        >
          <stop offset="0%" stopColor="#67e8f9" />
          <stop offset="55%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#0891b2" />
        </linearGradient>
        <linearGradient
          id="waveTail"
          gradientUnits="userSpaceOnUse"
          x1={WAVE.splitX}
          x2={WAVE.width}
        >
          <stop offset="0%" stopColor="#0891b2" stopOpacity="0.85" />
          <stop offset="45%" stopColor="#94a3b8" stopOpacity="0.65" />
          <stop offset="100%" stopColor="#94a3b8" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="waveArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.16" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* echo wave behind, for depth */}
      <path
        d={`${WAVE_SOLID} ${WAVE_TAIL.replace("M", "L")}`}
        fill="none"
        stroke="#cbd5e1"
        strokeWidth="7"
        strokeLinecap="round"
        opacity="0.4"
        transform="translate(0,16)"
      />

      {/* tinted area under the shipped stretch */}
      <path
        d={`${WAVE_SOLID} L ${WAVE.splitX},${WAVE.height} L 0,${WAVE.height} Z`}
        fill="url(#waveArea)"
      />

      <path
        d={WAVE_SOLID}
        fill="none"
        stroke="url(#waveSolid)"
        strokeWidth="5"
        strokeLinecap="round"
      />
      <path
        d={WAVE_TAIL}
        fill="none"
        stroke="url(#waveTail)"
        strokeWidth="5"
        strokeLinecap="round"
        strokeDasharray="1 14"
      />
    </svg>
  );
}

function RoadmapNode({
  status,
  caption,
  title,
  desc,
  x,
}: {
  status: "done" | "active" | "explore";
  caption: string;
  title: string;
  desc: string | null;
  x: number;
}) {
  const muted = status === "explore";

  return (
    // Offset by 4px so the bullet and stem centre on the wave peak at `x`.
    <li
      className="absolute top-0 w-[230px]"
      style={{ left: x - 4, height: WAVE.svgTop + WAVE.peakY }}
    >
      <div className="flex h-full flex-col items-start justify-end">
        <div className="relative pl-4 pb-3">
          <span
            className={[
              "absolute left-0 top-[5px] w-2 h-2 rounded-full",
              status === "done" ? "bg-cyan-500" : "",
              status === "active" ? "bg-cyan-600" : "",
              muted ? "bg-slate-300" : "",
            ].join(" ")}
          />
          <p
            className={[
              "text-[10px] font-semibold uppercase tracking-[0.18em] mb-1",
              muted ? "text-slate-400" : "text-cyan-600",
            ].join(" ")}
          >
            {caption}
          </p>
          <h3
            className={[
              "font-semibold text-base leading-snug",
              muted ? "text-slate-400" : "text-slate-900",
            ].join(" ")}
          >
            {title}
          </h3>
          {desc && (
            <p
              className={[
                "text-xs leading-relaxed mt-1.5 pr-6",
                muted ? "text-slate-400" : "text-slate-500",
              ].join(" ")}
            >
              {desc}
            </p>
          )}
        </div>

        {/* stem down to the wave */}
        <span
          className={[
            "ml-[3px] w-0.5 h-14 rounded-full",
            muted
              ? "bg-gradient-to-b from-slate-200 to-slate-300"
              : "bg-gradient-to-b from-cyan-200 to-cyan-400",
          ].join(" ")}
        />
      </div>

      {/* the node itself, sitting on the wave */}
      <span
        className="absolute -translate-x-1/2 -translate-y-1/2"
        style={{ left: 4, top: WAVE.svgTop + WAVE.peakY }}
      >
        <span
          className={[
            "relative flex items-center justify-center rounded-full",
            status === "done"
              ? "w-[18px] h-[18px] bg-cyan-500 ring-4 ring-cyan-500/15"
              : "",
            status === "active"
              ? "w-5 h-5 bg-white border-[3px] border-cyan-500 ring-4 ring-cyan-500/20"
              : "",
            muted
              ? "w-4 h-4 bg-white border-2 border-dashed border-slate-400"
              : "",
          ].join(" ")}
        >
          {status === "done" && (
            <Check className="w-2.5 h-2.5 text-white" strokeWidth={3.5} />
          )}
          {status === "active" && (
            <span className="absolute -inset-1 rounded-full border-2 border-cyan-500/60 animate-ping" />
          )}
        </span>
      </span>
    </li>
  );
}
