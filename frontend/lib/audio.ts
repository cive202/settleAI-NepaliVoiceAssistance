export type VolumeCallback = (volume: number) => void;

export class AudioRecorder {
  private audioCtx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private gain: GainNode | null = null;
  private chunks: Float32Array[] = [];
  private sampleRate = 44100;

  get isRecording() {
    return this.stream !== null;
  }

  async start(onVolume?: VolumeCallback): Promise<void> {
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    // Request 16kHz directly so the backend can skip librosa resampling on every request.
    this.audioCtx = new AudioContext({ sampleRate: 16000 });
    this.sampleRate = this.audioCtx.sampleRate;
    this.source = this.audioCtx.createMediaStreamSource(this.stream);

    // ScriptProcessor is deprecated but universally supported; fine for a prototype.
    this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(data));
      if (onVolume) {
        let rms = 0;
        for (let i = 0; i < data.length; i++) rms += data[i] * data[i];
        onVolume(Math.sqrt(rms / data.length));
      }
    };

    // Mute output — prevent feedback while keeping the processor node alive.
    this.gain = this.audioCtx.createGain();
    this.gain.gain.value = 0;
    this.source.connect(this.processor);
    this.processor.connect(this.gain);
    this.gain.connect(this.audioCtx.destination);
  }

  async stop(): Promise<Blob> {
    if (!this.processor || !this.audioCtx) throw new Error("Not recording");

    this.gain?.disconnect();
    this.processor.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());

    const sr = this.sampleRate;
    await this.audioCtx.close();

    const total = this.chunks.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    this.audioCtx = null;
    this.stream = null;
    this.processor = null;
    this.source = null;
    this.gain = null;
    this.chunks = [];

    return new Blob([encodeWAV(merged, sr)], { type: "audio/wav" });
  }
}

function encodeWAV(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numCh = 1;
  const bps = 16;
  const dataLen = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const str = (off: number, s: string) =>
    s.split("").forEach((c, i) => v.setUint8(off + i, c.charCodeAt(0)));

  str(0, "RIFF");
  v.setUint32(4, 36 + dataLen, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, numCh, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, (sampleRate * numCh * bps) / 8, true);
  v.setUint16(32, (numCh * bps) / 8, true);
  v.setUint16(34, bps, true);
  str(36, "data");
  v.setUint32(40, dataLen, true);

  let idx = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    idx += 2;
  }
  return buf;
}
