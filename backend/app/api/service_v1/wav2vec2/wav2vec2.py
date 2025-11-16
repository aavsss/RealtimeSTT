import io
import tempfile
import torch
import librosa
import numpy as np
import structlog
from typing import Generator, List
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from app.api.service_v1.transcriber import Transcriber
from pydub import AudioSegment
import os
from tqdm import tqdm
import soundfile as sf

MODEL_ID = "iamTangsang/Wav2Vec2_XLS-R-300m_Nepali_ASR"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 5  # model trained on <=5s segments; chunk longer audio


# device selection: cuda > mps > cpu
if torch.cuda.is_available():
    _device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    _device = torch.device("mps")
else:
    _device = torch.device("cpu")


class Wav2Vec2Transcriber(Transcriber):
    """
    Wav2Vec2 transcriber wrapper that implements `Transcriber`.
    - Loads processor + model in init.
    - Exposes `transcribe_audio(audio_bytes, language='en')` and `transcribe_file(path)`.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: int = CHUNK_SECONDS,
        device: torch.device = _device,
    ):
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.device = device
        self.logger = structlog.get_logger()

        # Load processor & model
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def transcribe_audio(self, audio_chunk: bytes, language: str = "en") -> str:
        """
        Implements Transcriber.transcribe_audio
        - Writes bytes to a temporary file and calls `transcribe_file`
        - This mimics OpenAIWhisper's approach so format reading is delegated to librosa/soundfile/ffmpeg.
        """
        if not audio_chunk:
            self.logger.warning("transcribe_audio called with empty audio_chunk")
            return ""

        try:
            # Use .wav suffix — librosa/soundfile/ffmpeg will still detect real content format.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                tmp.write(audio_chunk)
                tmp.flush()
                tmp_path = tmp.name
                self.logger.debug("Wrote audio bytes to temp file", path=tmp_path)

                with open(tmp_path, "rb") as audio_file:
                    # Convert and split the audio file
                    output_dir = "chunks"
                    os.makedirs(output_dir, exist_ok=True)
                    chunk_files = self.convert_and_split_audio(tmp_path, output_dir)

                    for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
                        # Load the audio file
                        try:
                            audio_input, sample_rate = sf.read(chunk_file)
                        except Exception as e:
                            print(f"Error reading the audio file {chunk_file}: {e}")
                            continue

                        # Ensure the audio is sampled at 16kHz as expected by the model
                        if sample_rate != 16000:
                            print(f"Expected sampling rate 16000, but got {sample_rate} for file {chunk_file}")
                            continue

                        # Preprocess the audio file
                        input_values = self.processor(
                            audio_input,
                            sampling_rate=sample_rate,
                            return_tensors="pt"
                        ).input_values.to(self.device)

                        # Perform inference
                        try:
                            with torch.no_grad():
                                logits = self.model(input_values).logits

                            # Get predicted ids
                            predicted_ids = torch.argmax(logits, dim=-1)

                            # Decode the ids to text
                            transcription = self.processor.decode(predicted_ids[0])
                            return transcription
                        except Exception as e:
                            print(f"Error during inference on file {chunk_file}: {e}")

        except Exception as e:
            self.logger.error("Wav2Vec2 transcription error", error=str(e))
            return "Transcription error"

    def convert_and_split_audio(self, input_file, output_dir, chunk_length_ms=1 * 60 * 1000, target_sample_rate=16000):
        audio = AudioSegment.from_file(input_file)

        # Convert sample rate
        audio = audio.set_frame_rate(target_sample_rate)

        # Split audio into chunks
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        output_files = []
        for idx, chunk in enumerate(chunks):
            chunk_name = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_chunk{idx}.wav")
            chunk.export(chunk_name, format="wav")
            output_files.append(chunk_name)

        return output_files

    def _load_audio_from_path(self, path: str) -> np.ndarray:
        """
        Load audio from disk into a numpy array and resample to model sample rate.
        Tries librosa first; if resampling call signature fails, falls back to
        `librosa.core.resample`.
        """
        try:
            wav, orig_sr = librosa.load(path, sr=None)
        except Exception as e:
            self.logger.warning("librosa.load failed, trying soundfile", error=str(e), path=path)
            # Try using soundfile as a fallback reader
            try:
                import soundfile as sf
                wav, orig_sr = sf.read(path, dtype="float32")
            except Exception as e2:
                self.logger.error("soundfile.read failed", error=str(e2), path=path)
                raise

        if orig_sr != self.sample_rate:
            try:
                wav = librosa.resample(wav, orig_sr, self.sample_rate)
            except TypeError:
                # Some librosa versions / environments expose a different symbol;
                # fall back to the explicit core.resample implementation.
                wav = librosa.core.resample(wav, orig_sr, self.sample_rate)
            except Exception as e:
                self.logger.error("resampling failed", error=str(e), orig_sr=orig_sr, target_sr=self.sample_rate)
                raise

        return wav

    def _load_audio_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """
        Load audio bytes into a numpy array by writing to a temporary file
        and delegating to `_load_audio_from_path`. This approach matches
        your OpenAI Whisper flow and avoids file-like incompatibilities.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name
            self.logger.debug("Wrote audio bytes to temp file for loading", path=tmp_path)
            return self._load_audio_from_path(tmp_path)

    def _chunk_audio(self, wav: np.ndarray) -> Generator[np.ndarray, None, None]:
        chunk_len = int(self.chunk_seconds * self.sample_rate)
        for start in range(0, len(wav), chunk_len):
            yield wav[start : start + chunk_len]

    def transcribe_file(self, path: str) -> str:
        wav = self._load_audio_from_path(path)
        return self._transcribe_wav_array(wav)

    def _transcribe_wav_array(self, wav: np.ndarray) -> str:
        texts: List[str] = []
        for chunk in self._chunk_audio(wav):
            if len(chunk) < 160:  # skip too tiny chunks
                continue

            inputs = self.processor(
                chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding="longest",
            )

            input_values = inputs.input_values.to(self.device)
            with torch.no_grad():
                logits = self.model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(pred_ids)[0]
            texts.append(text)

        return " ".join(texts)