import tempfile
import structlog
from typing import Optional, Any
from openai import OpenAI
from app.api.service_v1.transcriber import Transcriber
from config import settings
from transformers import pipeline

logger = structlog.get_logger()

class ReinforcedWhisper(Transcriber):

    def __init__(self):
        self.logger = logger or structlog.get_logger()
        self.reinforcedPipeline = pipeline(
            "automatic-speech-recognition",
            model="amitpant7/whisper-small-nepali",
        )

    def transcribe_audio(self, audio_chunk: bytes, language: str = "en") -> str:
        """
            Transcribe raw audio bytes. Returns transcribed text or an error string.
            """
        if not audio_chunk:
            self.logger.warning("transcribe_audio called with empty audio_chunk")
            return ""

        try:
            # Write bytes to a temporary file (Whisper expects a file-like object)
            with (tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp):
                tmp.write(audio_chunk)
                tmp.flush()
                tmp_path = tmp.name
                self.logger.debug("Wrote audio to temp file", path=tmp_path)

                with open(tmp_path, "rb") as audio_file:
                    transcript = self.reinforcedPipeline(audio_file.read())

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