def get_asr(type="nvidia"):
    if type == "runpod":
        from .whisper_runpod import RunPodWhisperASR

        return RunPodWhisperASR()

    if type == "nvidia":
        from .whisper_nvidia import WhisperNvidiaASR

        return WhisperNvidiaASR()

    from .whisper_local import WhisperLocalASR

    return WhisperLocalASR()
