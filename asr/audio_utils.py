import io
import numpy as np
import scipy.io.wavfile


def to_wav_buffer(audio_np: np.ndarray, sample_rate: int) -> io.BytesIO:
    buf = io.BytesIO()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    scipy.io.wavfile.write(buf, sample_rate, audio_int16)
    buf.seek(0)
    buf.name = "audio.wav"
    return buf
