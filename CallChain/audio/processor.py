import io
import numpy as np
import librosa
import soundfile as sf
from .config import AudioConfig

class AudioProcessor:
    """
    Handles audio preprocessing tasks such as loading, resampling,
    normalizing, trimming silence, and noise reduction.
    """
    def __init__(self, config: AudioConfig):
        self.config = config

    def _load_audio(self, path: str) -> tuple[np.ndarray, int]:
        """Load audio with librosa (returns float32 waveform)."""
        y, sr = librosa.load(path, sr=None, mono=True)
        return y, sr

    def _resample(self, y: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample to target_sr if needed."""
        if orig_sr != self.config.target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.config.target_sr)
        return y

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """Peak-normalize to [-1, 1]."""
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y

    def _trim_silence(self, y: np.ndarray) -> np.ndarray:
        """Trim leading/trailing silence using a simple energy threshold."""
        return librosa.effects.trim(y, top_db=20)[0]

    def _reduce_noise(self, y: np.ndarray) -> np.ndarray:
        """Apply a very light spectral gating noise reduction."""
        try:
            import noisereduce as nr
        except ImportError:
            raise RuntimeError(
                "Noise reduction requested but `noisereduce` is not installed. "
                "Run `pip install noisereduce` or set `noise_reduction=False`."
            )
        # Estimate noise from the first 0.5s (or the whole clip if shorter)
        noise_len = int(0.5 * self.config.target_sr)
        noise_clip = y[:noise_len] if len(y) > noise_len else y
        return nr.reduce_noise(y=y, sr=self.config.target_sr, y_noise=noise_clip)

    def preprocess(self, path: str) -> io.BytesIO:
        """
        Full preprocessing pipeline.
        Returns a BytesIO buffer containing the processed WAV file,
        ready for the Groq API.
        """
        y, sr = self._load_audio(path)
        y = self._resample(y, sr)

        if self.config.normalize:
            y = self._normalize(y)
        if self.config.trim_silence:
            y = self._trim_silence(y)
        if self.config.noise_reduction:
            y = self._reduce_noise(y)

        # Encode back to a temporary WAV
        buffer = io.BytesIO()
        sf.write(buffer, y, self.config.target_sr, format="WAV")
        buffer.seek(0)
        # Set the name attribute so Groq API can detect the file type
        buffer.name = "audio.wav"
        return buffer
