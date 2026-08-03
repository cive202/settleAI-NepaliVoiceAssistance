import { WS_API } from "./api";

// Keep in sync with config.py's VAD block.
const CHUNK_MS = 32; // ms per chunk — must match vad.py's Silero chunk size
const CHUNK_SAMPLES = 512; // 16000 * CHUNK_MS / 1000
const PRE_SPEECH_PADDING_MS = 300; // ms of audio kept before speech starts
const RING_BUFFER_SIZE = Math.ceil(PRE_SPEECH_PADDING_MS / CHUNK_MS);

export type BargeInCallback = (preRoll: Float32Array[]) => void;

/**
 * Listens to the mic while the assistant is speaking and reports to the
 * backend's /ws/barge-in endpoint (Silero VAD) whether the user has
 * started talking over the playback. One-shot per start()/stop() cycle.
 */
export class BargeInDetector {
  private audioCtx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private gain: GainNode | null = null;
  private ws: WebSocket | null = null;
  private ringBuffer: Float32Array[] = [];
  private fired = false;

  async start(onBargeIn: BargeInCallback): Promise<void> {
    this.fired = false;
    this.ringBuffer = [];

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    this.audioCtx = new AudioContext({ sampleRate: 16000 });
    this.source = this.audioCtx.createMediaStreamSource(this.stream);
    this.processor = this.audioCtx.createScriptProcessor(CHUNK_SAMPLES, 1, 1);

    this.ws = new WebSocket(`${WS_API}/ws/barge-in`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (e) => {
      if (this.fired) return;
      const msg = JSON.parse(e.data as string);
      if (msg.type === "barge_in") {
        this.fired = true;
        const preRoll = this.ringBuffer;
        this.stop();
        onBargeIn(preRoll);
      }
    };

    this.processor.onaudioprocess = (e) => {
      const data = new Float32Array(e.inputBuffer.getChannelData(0));

      this.ringBuffer.push(data);
      if (this.ringBuffer.length > RING_BUFFER_SIZE) this.ringBuffer.shift();

      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(data.buffer);
      }
    };

    // Mute output — prevent feedback while keeping the processor node alive.
    this.gain = this.audioCtx.createGain();
    this.gain.gain.value = 0;
    this.source.connect(this.processor);
    this.processor.connect(this.gain);
    this.gain.connect(this.audioCtx.destination);
  }

  stop(): void {
    this.ws?.close();
    this.ws = null;
    this.gain?.disconnect();
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.audioCtx?.close();

    this.audioCtx = null;
    this.stream = null;
    this.processor = null;
    this.source = null;
    this.gain = null;
  }
}
