import os
from typing import Optional, Union
from groq import Groq
from .config import AudioConfig
from .processor import AudioProcessor

class AudioTranscriber:
    """
    A simple audio transcriber using Groq's Whisper model with optional
    preprocessing (resampling, normalization, silence trimming, noise reduction).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
        temperature: float = 0.0,
        # ---- Pre‑processing options (kept for backward compatibility) ----
        target_sr: int = 16000,
        normalize: bool = True,
        trim_silence: bool = True,
        noise_reduction: bool = False,
        # ---- New Config Object ----
        config: Optional[AudioConfig] = None
    ):
        """
        Initialize the AudioTranscriber.

        You can initialize this class in two ways:
        1. Pass individual parameters (backward compatible).
        2. Pass an `AudioConfig` object for cleaner configuration.

        Args:
            api_key: Groq API key.
            model: Whisper model name.
            language: Language code.
            temperature: Sampling temperature.
            target_sr: Target sample rate for preprocessing.
            normalize: Whether to normalize audio.
            trim_silence: Whether to trim silence.
            noise_reduction: Whether to apply noise reduction.
            config: An optional AudioConfig object. If provided, it overrides individual parameters.
        """
        if config:
            self.config = config
        else:
            self.config = AudioConfig(
                api_key=api_key,
                model=model,
                language=language,
                temperature=temperature,
                target_sr=target_sr,
                normalize=normalize,
                trim_silence=trim_silence,
                noise_reduction=noise_reduction
            )
        
        # Ensure API key is present
        if not self.config.api_key:
             raise ValueError(
                "No API key provided. Either pass it to the constructor, set it in AudioConfig, "
                "or set GROQ_API_KEY environment variable."
            )

        self.client = Groq(api_key=self.config.api_key)
        self.processor = AudioProcessor(self.config)

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text after optional preprocessing.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            The transcribed text.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # ---- Pre‑process -------------------------------------------------
        processed_audio = self.processor.preprocess(audio_path)

        # ---- Call Groq Whisper -----------------------------------------
        try:
            response = self.client.audio.transcriptions.create(
                file=processed_audio,
                model=self.config.model,
                language=self.config.language,
                temperature=self.config.temperature,
                response_format="text",  # plain text
            )
            return response
        except Exception as e:
            raise Exception(f"Error during transcription: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Example 1: Using individual parameters (Old way)
    # transcriber = AudioTranscriber()
    
    # Example 2: Using AudioConfig (New way)
    # config = AudioConfig(noise_reduction=True)
    # transcriber = AudioTranscriber(config=config)
    
    pass