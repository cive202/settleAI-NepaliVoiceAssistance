def get_asr(type="nvidia"):
    if type == "nvidia":
        from .whisper_nvidia import WhisperNvidiaASR

        return WhisperNvidiaASR()

    if type == "whisper_local":
        from .whisper_local import WhisperLocalASR

        return WhisperLocalASR()

    from .indic_conformer import IndicConformerASR

    return IndicConformerASR()
