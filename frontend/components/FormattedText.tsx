// Renders plain-text LLM output (no markdown — replies are spoken aloud, so
// the model is instructed to avoid markdown syntax) with readable structure:
// blank lines become paragraph breaks, and lines that already look like a
// list (leading "-", "•", "1.") render as an actual list instead of a run-on
// paragraph with the whitespace collapsed.
const BULLET_RE = /^[-•*]\s+(.*)/;
const NUMBERED_RE = /^(\d+)[.)]\s+(.*)/;

export function FormattedText({ text }: { text: string }) {
  const paragraphs = text.trim().split(/\n{2,}/);

  return (
    <div className="space-y-2.5">
      {paragraphs.map((para, i) => {
        const lines = para.split("\n").filter((l) => l.trim().length > 0);
        const isBulletBlock = lines.length > 0 && lines.every((l) => BULLET_RE.test(l.trim()));
        const isNumberedBlock = lines.length > 0 && lines.every((l) => NUMBERED_RE.test(l.trim()));

        if (isBulletBlock) {
          return (
            <ul key={i} className="list-disc list-outside pl-4 space-y-1">
              {lines.map((l, j) => (
                <li key={j}>{l.trim().replace(BULLET_RE, "$1")}</li>
              ))}
            </ul>
          );
        }

        if (isNumberedBlock) {
          return (
            <ol key={i} className="list-decimal list-outside pl-4 space-y-1">
              {lines.map((l, j) => (
                <li key={j}>{l.trim().replace(NUMBERED_RE, "$2")}</li>
              ))}
            </ol>
          );
        }

        return (
          <p key={i} className="whitespace-pre-wrap">
            {lines.join("\n")}
          </p>
        );
      })}
    </div>
  );
}
