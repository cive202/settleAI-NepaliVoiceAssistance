/**
 * Plays base64-encoded MP3 chunks back-to-back as they arrive, instead of
 * waiting for the full reply before starting playback. Call finish() once
 * the caller knows no more chunks are coming — onDrained then fires after
 * the last queued chunk finishes playing (or immediately, if the queue was
 * already empty).
 */
export class AudioQueue {
  private queue: string[] = [];
  private playing = false;
  private finished = false;
  private current: HTMLAudioElement | null = null;
  onDrained?: () => void;

  push(base64Mp3: string): void {
    this.pushSrc(`data:audio/mp3;base64,${base64Mp3}`);
  }

  /** Queue an already-playable src (a URL, or a data: URI). The demo clip is a
   * static file rather than a streamed base64 chunk, so it enters here. */
  pushSrc(src: string): void {
    this.queue.push(src);
    if (!this.playing) this.playNext();
  }

  finish(): void {
    this.finished = true;
    if (!this.playing && this.queue.length === 0) this.onDrained?.();
  }

  stop(): void {
    this.current?.pause();
    this.current = null;
    this.queue = [];
    this.playing = false;
    this.finished = false;
  }

  private playNext(): void {
    const src = this.queue.shift();
    if (!src) {
      this.playing = false;
      if (this.finished) this.onDrained?.();
      return;
    }
    this.playing = true;
    const audio = new Audio(src);
    this.current = audio;
    audio.onended = () => this.playNext();
    audio.onerror = () => this.playNext();
    audio.play().catch(() => this.playNext());
  }
}
