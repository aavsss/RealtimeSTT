from app.api.service_v1.transcriberOptions import TranscriberOptions
from app.api.service_v1.transcriber import Transcriber
from app.api.service_v1.whisper.openAiWhisper import OpenAIWhisper
from app.api.service_v1.whisper.reinforcedWhisper import ReinforcedWhisper
from app.api.service_v1.wav2vec2.wav2vec2 import Wav2Vec2Transcriber

def transcribe_audio(
        audio_chunk: bytes,
        modal: TranscriberOptions = TranscriberOptions.WAV2VEC2_NEPALI,
        language: str = 'en'
) -> str:
    """
    Transcribe audio using the specified transcription model.

    Parameters:
    - modal (str): The transcription model to use. Default is 'openai'.
    - audio_chunk (bytes): The audio data in bytes.
    - language (str): The language of the audio. Default is "en" (English).

    Returns:
    - str: The transcribed text.
    """
    match modal:
        case TranscriberOptions.OPEN_AI_WHISPER:
            transcriber: Transcriber = OpenAIWhisper.from_settings()
            return transcriber.transcribe_audio(audio_chunk, language=language)
        case TranscriberOptions.REINFORCED_AI_WHISPER:
            transcriber: Transcriber = ReinforcedWhisper()
            return transcriber.transcribe_audio(audio_chunk, language=language)
        case TranscriberOptions.WAV2VEC2_NEPALI:
            transcriber: Transcriber = Wav2Vec2Transcriber()
            return transcriber.transcribe_audio(audio_chunk, language=language)
        case _:
            # Unsupported modal — preserve previous behavior (implicit None).
            pass