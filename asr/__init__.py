def get_asr(type="nvidia"):
    if type == "nvidia":
        from .whisper_nvidia import WhisperNvidiaASR

        return WhisperNvidiaASR()

    from .whisper_local import WhisperLocalASR

    return WhisperLocalASR()
