from enum import Enum

class TranscriberOptions(str, Enum):
    OPEN_AI_WHISPER = "OPEN_AI_WHISPER"
    REINFORCED_AI_WHISPER = "REINFORCED_AI_WHISPER"