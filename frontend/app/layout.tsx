import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";

const geist = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "SettleAI — Nepali Voice Assistant",
  description:
    "Real-time Nepali voice assistant powered by Whisper ASR, LLM, and natural TTS.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ne" className={`${geist.variable} h-full antialiased`}>
      <body className="min-h-full bg-[#080d1a] text-white">{children}</body>
    </html>
  );
}
