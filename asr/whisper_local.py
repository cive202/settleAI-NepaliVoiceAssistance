import logging
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from .base import ASRBackend
from config import WHISPER_MODEL, SAMPLE_RATE

log = logging.getLogger(__name__)


class WhisperLocalASR(ASRBackend):
    def __init__(self, model_name: str = WHISPER_MODEL) -> None:
        log.info("Loading local Whisper model: %s", model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        config = AutoConfig.from_pretrained(model_name)
        if "large-v3-turbo" in model_name.lower() and config.decoder_layers != 4:
            # kiranpantha/whisper-large-v3-turbo-nepali ships a config.json copied
            # from full whisper-large-v3 (decoder_layers=32) instead of the turbo
            # base it's actually fine-tuned from (decoder_layers=4, per
            # openai/whisper-large-v3-turbo's config). Left uncorrected,
            # transformers instantiates 28 randomly-initialized decoder layers
            # on top of the real 4-layer checkpoint and generation comes out empty.
            log.warning(
                "Correcting mismatched decoder_layers in %s config: %d -> 4",
                model_name, config.decoder_layers,
            )
            config.decoder_layers = 4

        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, config=config)

        # kiranpantha/whisper-large-v3-turbo-nepali ships model weights only —
        # it has no tokenizer/preprocessor files of its own (tokenizer.json,
        # vocab.json, etc. all 404). AutoProcessor.from_pretrained silently
        # falls back to an empty tokenizer in that case, so generation produces
        # token ids that decode to "". Load the processor from the base model
        # it was fine-tuned from instead.
        processor_source = (
            "openai/whisper-large-v3-turbo" if "large-v3-turbo" in model_name.lower() else model_name
        )
        processor = AutoProcessor.from_pretrained(processor_source)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            generate_kwargs={"language": "ne", "task": "transcribe"},
        )

    def transcribe(self, audio_np: np.ndarray) -> str:
        log.debug("Transcribing %.2fs of audio (local)", len(audio_np) / SAMPLE_RATE)
        result = self._pipe({"array": audio_np.astype(np.float32), "sampling_rate": SAMPLE_RATE})
        text = result["text"].strip()
        log.debug("Transcription: %r", text)
        return text
