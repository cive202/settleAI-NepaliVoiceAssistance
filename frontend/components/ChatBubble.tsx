export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
}

export function ChatBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex items-end gap-2 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-cyan-500 flex-shrink-0 flex items-center justify-center text-xs font-bold shadow-[0_0_8px_rgba(6,182,212,0.5)]">
          S
        </div>
      )}
      <div
        className={[
          "max-w-xs md:max-w-md lg:max-w-lg px-4 py-3 rounded-2xl text-sm leading-relaxed",
          isUser
            ? "bg-white/10 rounded-br-sm"
            : "bg-cyan-500/15 border border-cyan-500/20 rounded-bl-sm",
        ].join(" ")}
      >
        {message.text}
      </div>
    </div>
  );
}
