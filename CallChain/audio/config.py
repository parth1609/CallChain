from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class AudioConfig:
    """
    Configuration for the AudioTranscriber and AudioProcessor.
    """
    # API Configuration
    api_key: Optional[str] = None
    
    # Model Configuration
    model: str = "whisper-large-v3-turbo"
    language: str = "en"
    temperature: float = 0.0
    
    # Preprocessing Configuration
    target_sr: int = 16000
    normalize: bool = True
    trim_silence: bool = True
    noise_reduction: bool = False

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY")
