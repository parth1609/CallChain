import os
from typing import Optional, Union
from .config import AudioConfig
from .processor import AudioProcessor
from .clients import AudioClient, GroqAudioClient

class AudioTranscriber:
    """
    A simple audio transcriber using a pluggable AudioClient (default: Groq)
    with optional preprocessing (resampling, normalization, silence trimming, noise reduction).
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
        config: Optional[AudioConfig] = None,
        # ---- New Client Injection ----
        client: Optional[AudioClient] = None
    ):
        """
        Initialize the AudioTranscriber.

        Args:
            api_key: API key for the default client (Groq). Ignored if `client` is provided.
            model: Model name.
            language: Language code.
            temperature: Sampling temperature.
            target_sr: Target sample rate for preprocessing.
            normalize: Whether to normalize audio.
            trim_silence: Whether to trim silence.
            noise_reduction: Whether to apply noise reduction.
            config: An optional AudioConfig object.
            client: An optional AudioClient instance. If not provided, defaults to GroqAudioClient.
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
        
        # Initialize Client
        if client:
            self.client = client
        else:
            # Default to Groq for backward compatibility
            self.client = GroqAudioClient(api_key=self.config.api_key)

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

        # ---- Call Client -------------------------------------------------
        try:
            return self.client.transcribe(
                audio_file=processed_audio,
                model=self.config.model,
                language=self.config.language,
                temperature=self.config.temperature
            )
        except Exception as e:
            raise Exception(f"Error during transcription: {str(e)}")

# Example usage
# if __name__ == "__main__":
    # # Fix for running this script directly
    # import sys
    # import os
    # from load_dotenv import load_dotenv
    # load_dotenv()
    # if __package__ is None:
    #     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    #     from CallChain.audio.config import AudioConfig
    #     from CallChain.audio.transcribe import AudioTranscriber
    #     from CallChain.audio.clients import GroqAudioClient
    
    # print("🎤 Initializing AudioTranscriber...")
    
    # try:
    #     # Example: Injecting a client explicitly
    #     groq_client = GroqAudioClient()
    #     transcriber = AudioTranscriber(client=groq_client)
    #     print("✅ Transcriber initialized with explicit Groq client")
        
    # except Exception as e:
    #     print(f"❌ Error: {e}")