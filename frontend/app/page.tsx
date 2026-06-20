import Link from "next/link";
import { Mic, Brain, Volume2, ArrowRight } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-5">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-full bg-cyan-500 shadow-[0_0_14px_rgba(6,182,212,0.6)]" />
          <span className="font-semibold tracking-tight">SettleAI</span>
        </div>
        <Link
          href="/chat"
          className="text-sm text-white/50 hover:text-white transition-colors"
        >
          Open App →
        </Link>
      </header>

      {/* Hero */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 text-center gap-8 py-16">
        {/* Glowing mic icon */}
        <div className="relative">
          <div className="absolute inset-0 rounded-full bg-cyan-500/20 blur-3xl scale-[2]" />
          <div className="relative w-24 h-24 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center shadow-[0_0_48px_rgba(6,182,212,0.25)]">
            <Mic className="w-10 h-10 text-cyan-400" />
          </div>
        </div>

        <div className="space-y-4 max-w-2xl">
          <p className="text-cyan-400 text-xs font-semibold tracking-[0.25em] uppercase">
            Nepali Voice Model · Powered by SettleAI
          </p>
          <h1 className="text-5xl md:text-6xl font-bold leading-tight tracking-tight">
            नेपाली भाषामा
            <br />
            <span className="text-cyan-400">कुरा गर्नुहोस्</span>
          </h1>
          <p className="text-white/50 text-lg leading-relaxed">
            Speak in Nepali and get intelligent responses in real time.
            <br className="hidden md:block" />
            Whisper ASR · Large Language Model · Natural TTS.
          </p>
        </div>

        <Link
          href="/chat"
          className="group inline-flex items-center gap-2 px-8 py-4 bg-cyan-500 hover:bg-cyan-400 rounded-full text-base font-semibold transition-all duration-200 shadow-[0_0_32px_rgba(6,182,212,0.4)] hover:shadow-[0_0_52px_rgba(6,182,212,0.65)]"
        >
          Start Talking
          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </Link>

        {/* Feature cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl w-full mt-4">
          <FeatureCard
            icon={<Mic className="w-5 h-5" />}
            title="Whisper ASR"
            desc="State-of-the-art Nepali speech recognition by OpenAI Whisper"
          />
          <FeatureCard
            icon={<Brain className="w-5 h-5" />}
            title="Smart LLM"
            desc="Multi-turn Nepali conversations powered by a large language model"
          />
          <FeatureCard
            icon={<Volume2 className="w-5 h-5" />}
            title="Natural TTS"
            desc="Fluent Nepali text-to-speech playback with gTTS"
          />
        </div>
      </main>

      <footer className="text-center py-6 text-white/20 text-xs">
        Built with Next.js · FastAPI · Whisper · gTTS
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  desc,
}: {
  icon: React.ReactNode;
  title: string;
  desc: string;
}) {
  return (
    <div className="p-5 rounded-2xl border border-white/10 bg-white/5 text-left">
      <div className="w-9 h-9 rounded-xl bg-cyan-500/15 text-cyan-400 flex items-center justify-center mb-3">
        {icon}
      </div>
      <h3 className="font-medium text-sm mb-1">{title}</h3>
      <p className="text-white/40 text-xs leading-relaxed">{desc}</p>
    </div>
  );
}
