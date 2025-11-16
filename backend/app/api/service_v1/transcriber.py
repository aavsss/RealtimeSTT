from abc import ABC, abstractmethod

class Transcriber(ABC):
    """
    Interface (abstract base class) that exposes the single required method
    used by the rest of the codebase.

    Only the highlighted function is declared here:
      def transcribe_audio(self, audio_chunk: bytes, language: str = "en") -> str
    """

    @abstractmethod
    def transcribe_audio(self, audio_chunk: bytes, language: str = "en") -> str:
        """
        Transcribe raw audio bytes and return the resulting text.
        Implementations must provide this method.
        """
        pass