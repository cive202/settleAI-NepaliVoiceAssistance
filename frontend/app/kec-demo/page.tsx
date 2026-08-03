import Link from "next/link";
import {
  GraduationCap,
  Menu,
  BookOpen,
  Users,
  Trophy,
  ArrowRight,
  MapPin,
  Phone,
  Mail,
} from "lucide-react";
import { AskWidget } from "@/components/AskWidget";

const NAV_LINKS = ["Home", "About", "Academics", "Admissions", "Notices", "Contact"];

/**
 * Standalone mock of a college website (modeled loosely on kecktm.edu.np) —
 * a demo host page showing how the SettleAI Q&A widget (<AskWidget/>) drops
 * into an existing site's navbar as a single button.
 */
export default function KECDemoPage() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      {/* Top info bar */}
      <div className="hidden sm:flex items-center justify-end gap-6 px-8 py-1.5 bg-slate-900 text-slate-300 text-[11px]">
        <span className="flex items-center gap-1.5"><Phone className="w-3 h-3" /> +977-1-5970027</span>
        <span className="flex items-center gap-1.5"><Mail className="w-3 h-3" /> info@kecktm.edu.np</span>
        <span className="flex items-center gap-1.5"><MapPin className="w-3 h-3" /> Kathmandu, Nepal</span>
      </div>

      {/* Navbar */}
      <header className="sticky top-0 z-40 bg-white/95 backdrop-blur border-b border-slate-200">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-5 py-3.5">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-blue-700 flex items-center justify-center">
              <GraduationCap className="w-5 h-5 text-white" />
            </div>
            <div className="leading-tight">
              <p className="font-bold text-sm text-slate-900">Kathmandu Engineering College</p>
              <p className="text-[10px] text-slate-400 tracking-wide">Affiliated to Purbanchal University</p>
            </div>
          </div>

          <nav className="hidden md:flex items-center gap-6 text-sm text-slate-600">
            {NAV_LINKS.map((l) => (
              <a key={l} href="#" className="hover:text-blue-700 transition-colors">
                {l}
              </a>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <AskWidget />
            <button className="md:hidden p-2 text-slate-500">
              <Menu className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="bg-gradient-to-br from-blue-800 to-blue-950 text-white">
        <div className="max-w-6xl mx-auto px-5 py-20 grid md:grid-cols-2 gap-10 items-center">
          <div className="space-y-5">
            <p className="text-blue-300 text-xs font-semibold tracking-[0.2em] uppercase">
              Est. 1998 · Kathmandu, Nepal
            </p>
            <h1 className="text-4xl md:text-5xl font-bold leading-tight">
              Kathmandu Engineering College
            </h1>
            <p className="text-blue-100/80 text-base leading-relaxed max-w-md">
              Shaping engineers for tomorrow with quality technical education, modern
              labs, and industry-aligned programs in the heart of Bara district.
            </p>
            <div className="flex gap-3 pt-2">
              <button className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-blue-800 rounded-lg text-sm font-semibold hover:bg-blue-50 transition-colors">
                Apply for Admission <ArrowRight className="w-4 h-4" />
              </button>
              <button className="px-5 py-2.5 border border-white/30 rounded-lg text-sm font-medium hover:bg-white/10 transition-colors">
                Explore Programs
              </button>
            </div>
          </div>
          <div className="rounded-2xl bg-white/5 border border-white/10 aspect-video flex items-center justify-center">
            <GraduationCap className="w-16 h-16 text-white/20" />
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="max-w-6xl mx-auto px-5 -mt-10 relative z-10">
        <div className="grid grid-cols-3 gap-4 bg-white rounded-2xl shadow-lg border border-slate-100 p-6">
          <Stat icon={<BookOpen className="w-5 h-5" />} value="5" label="Engineering Programs" />
          <Stat icon={<Users className="w-5 h-5" />} value="1200+" label="Students Enrolled" />
          <Stat icon={<Trophy className="w-5 h-5" />} value="20+" label="Years of Excellence" />
        </div>
      </section>

      {/* About */}
      <section className="max-w-6xl mx-auto px-5 py-20 grid md:grid-cols-2 gap-10">
        <div>
          <p className="text-blue-700 text-xs font-semibold tracking-[0.2em] uppercase mb-2">
            About the College
          </p>
          <h2 className="text-2xl font-bold text-slate-900 mb-3">
            Building Nepal&apos;s next generation of engineers
          </h2>
          <p className="text-slate-500 text-sm leading-relaxed">
            Kalaiya Engineering College offers undergraduate programs in Civil, Computer,
            Electronics &amp; Communication, and Electrical Engineering, combining rigorous
            academics with hands-on lab work and industry exposure. Have a question about
            programs, fees, or admissions? Try the{" "}
            <span className="text-blue-700 font-medium">Ask KEC</span> button in the navbar
            above — it can answer in text and read the response aloud.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4">
          {["Civil Engineering", "Computer Engineering", "Electronics & Comm.", "Electrical Engineering"].map(
            (p) => (
              <div key={p} className="p-4 rounded-xl border border-slate-200 bg-white">
                <p className="text-sm font-semibold text-slate-800">{p}</p>
                <p className="text-xs text-slate-400 mt-1">4-year B.E. program</p>
              </div>
            )
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-400 text-xs">
        <div className="max-w-6xl mx-auto px-5 py-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© {new Date().getFullYear()} Kathmandu Engineering College · Demo page for SettleAI widget integration</p>
          <Link href="/" className="hover:text-white transition-colors">
            ← Back to SettleAI
          </Link>
        </div>
      </footer>
    </div>
  );
}

function Stat({ icon, value, label }: { icon: React.ReactNode; value: string; label: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-10 h-10 rounded-lg bg-blue-50 text-blue-700 flex items-center justify-center flex-shrink-0">
        {icon}
      </div>
      <div>
        <p className="text-lg font-bold text-slate-900 leading-tight">{value}</p>
        <p className="text-[11px] text-slate-400">{label}</p>
      </div>
    </div>
  );
}
