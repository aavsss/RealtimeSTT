import tempfile
import structlog
from typing import Optional, Any
from openai import OpenAI
from app.api.service_v1.transcriber import Transcriber
from config import settings
from transformers import pipeline

logger = structlog.get_logger()

class OpenAIWhisper(Transcriber):
    """
    Simple wrapper for OpenAI Whisper transcription.

    Usage:
        from app.api.service_v1.whisper.openAIWhisper import OpenAIWhisper
        transcriber = OpenAIWhisper(api_key="sk-xxx")  # or OpenAIWhisper.from_settings()
        text = transcriber.transcribe_audio(audio_bytes)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1", logger: Optional[Any] = None):
        """
        Initialize the transcriber.

        - api_key: OpenAI API key. If None, falls back to `settings.openai_api_key`.
        - model: model name to use for transcription (default "whisper-1").
        - logger: optional custom logger (defaults to structlog.get_logger()).
        """
        self.api_key = api_key or getattr(settings, "openai_api_key", None)
        if not self.api_key:
            raise ValueError("OpenAI API key is required (pass api_key or set settings.openai_api_key)")
        self.model = model
        self.logger = logger or structlog.get_logger()
        self.client = OpenAI(api_key=self.api_key)

    def transcribe_audio(self, audio_chunk: bytes, language: str = "en") -> str:
        """
        Transcribe raw audio bytes. Returns transcribed text or an error string.
        """
        if not audio_chunk:
            self.logger.warning("transcribe_audio called with empty audio_chunk")
            return ""

        try:
            # Write bytes to a temporary file (Whisper expects a file-like object)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
                tmp.write(audio_chunk)
                tmp.flush()
                tmp_path = tmp.name
                self.logger.debug("Wrote audio to temp file", path=tmp_path)

                with open(tmp_path, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=language
                    )

                # Transcript object may differ by client version; handle common shapes
                text = ""
                if hasattr(transcript, "text"):
                    text = transcript.text
                elif isinstance(transcript, dict):
                    text = transcript.get("text", "")
                else:
                    # Fallback to string representation
                    text = str(transcript)

                self.logger.info("Transcription completed", text=text)
                return text

        except Exception as e:
            self.logger.error("OpenAI transcription error", error=str(e))
            return "Transcription error"

    def transcribe_file(self, file_path: str, language: str = "en") -> str:
        """
        Transcribe an audio file on disk.
        """
        try:
            with open(file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language
                )

            if hasattr(transcript, "text"):
                return transcript.text
            if isinstance(transcript, dict):
                return transcript.get("text", "")
            return str(transcript)

        except Exception as e:
            self.logger.error("OpenAI transcription error (file)", error=str(e), path=file_path)
            return "Transcription error"

    @classmethod
    def from_settings(cls) -> "OpenAIWhisper":
        """
        Convenience constructor that reads the API key from `settings.openai_api_key`.
        """
        return cls(api_key=getattr(settings, "openai_api_key", None))