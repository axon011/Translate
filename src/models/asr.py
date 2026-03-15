"""Automatic Speech Recognition using faster-whisper.

Uses primeline/whisper-tiny-german-1224 (fine-tuned for German) via the
faster-whisper CTranslate2 backend for efficient inference on limited VRAM.

Key design choices:
- faster-whisper over standard Whisper: 2-4x faster, int8 quantization
- German-specific fine-tuned tiny model: 6.3% WER, only ~0.5GB VRAM
- Beats vanilla whisper-small (10% WER) at 6x fewer parameters
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from src.utils.config import ASRConfig, get_config
from src.utils.logging import TimingContext, get_logger

logger = get_logger("asr")


@dataclass
class TranscriptionResult:
    """Result from ASR transcription."""

    text: str
    language: str
    duration_s: float
    segments: list[dict]
    confidence: float


class ASRModel:
    """Whisper-based ASR using faster-whisper for efficient inference.

    Supports:
    - Audio file transcription (wav, mp3, flac, etc.)
    - Streaming segments with timestamps
    - int8 quantization for 4GB VRAM
    - Explicit load/unload for VRAM management
    """

    def __init__(
        self,
        config: ASRConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or get_config().asr
        self.device = device or self.config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self._model = None
        self._loaded = False

    def load(self) -> None:
        """Load the Whisper model."""
        if self._loaded:
            return

        from faster_whisper import WhisperModel

        logger.info(
            "Loading ASR model",
            extra={"component": "asr", "model": self.config.model_id},
        )

        # faster-whisper uses "cuda" or "cpu" for device, and compute_type for precision
        self._model = WhisperModel(
            self.config.model_id,
            device=self.device,
            compute_type=self.config.compute_type if self.device == "cuda" else "int8",
        )

        self._loaded = True

        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**2
            logger.info(
                f"ASR model loaded, VRAM: {vram:.0f} MB",
                extra={"component": "asr", "vram_mb": round(vram, 1)},
            )

    def unload(self) -> None:
        """Unload model from GPU to free VRAM."""
        if not self._loaded:
            return

        del self._model
        self._model = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("ASR model unloaded", extra={"component": "asr"})

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (wav, mp3, flac, ogg, etc.).
            language: Language code (e.g., "de"). Auto-detects if None.

        Returns:
            TranscriptionResult with text, segments, and metadata.
        """
        if not self._loaded:
            self.load()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with TimingContext("asr_transcribe") as t:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=language or self.config.language,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=self.config.temperature,
                vad_filter=True,  # Voice Activity Detection for better accuracy
            )

            # Collect segments
            segments = []
            full_text_parts = []
            total_confidence = 0.0
            num_segments = 0

            for segment in segments_iter:
                seg_dict = {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                    "avg_logprob": round(segment.avg_logprob, 4),
                    "no_speech_prob": round(segment.no_speech_prob, 4),
                }
                segments.append(seg_dict)
                full_text_parts.append(segment.text.strip())

                total_confidence += math.exp(segment.avg_logprob)
                num_segments += 1

        full_text = " ".join(full_text_parts)
        avg_confidence = total_confidence / max(num_segments, 1)

        logger.info(
            f"Transcribed {info.duration:.1f}s audio in {t.elapsed_ms:.0f}ms "
            f"({len(segments)} segments)",
            extra={
                "component": "asr",
                "latency_ms": round(t.elapsed_ms, 1),
                "items": len(segments),
            },
        )

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            duration_s=round(info.duration, 2),
            segments=segments,
            confidence=round(avg_confidence, 4),
        )

    def transcribe_array(
        self,
        audio_array,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe from a numpy array (for API use).

        Args:
            audio_array: numpy float32 array of audio samples.
            sample_rate: Sample rate in Hz (faster-whisper expects 16kHz).
            language: Language code.

        Returns:
            TranscriptionResult.
        """
        if not self._loaded:
            self.load()

        import numpy as np

        # Resample if needed
        if sample_rate != 16000:
            import librosa

            audio_array = librosa.resample(
                audio_array.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=16000,
            )

        with TimingContext("asr_transcribe_array"):
            segments_iter, info = self._model.transcribe(
                audio_array.astype(np.float32),
                language=language or self.config.language,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=self.config.temperature,
                vad_filter=True,
            )

            segments = []
            full_text_parts = []
            total_confidence = 0.0
            num_segments = 0

            for segment in segments_iter:
                seg_dict = {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                    "avg_logprob": round(segment.avg_logprob, 4),
                    "no_speech_prob": round(segment.no_speech_prob, 4),
                }
                segments.append(seg_dict)
                full_text_parts.append(segment.text.strip())
                total_confidence += math.exp(segment.avg_logprob)
                num_segments += 1

        full_text = " ".join(full_text_parts)
        avg_confidence = total_confidence / max(num_segments, 1)

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            duration_s=round(info.duration, 2),
            segments=segments,
            confidence=round(avg_confidence, 4),
        )
