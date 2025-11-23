from typing import Protocol, Any, Optional
import os
from groq import Groq

class AudioClient(Protocol):
    """
    Protocol defining the interface for Audio Transcription Clients.
    """
    def transcribe(
        self, 
        audio_file: Any, 
        model: str, 
        language: str, 
        temperature: float
    ) -> str:
        """
        Transcribe the given audio file.
        
        Args:
            audio_file: The audio file object (e.g., BytesIO or file path).
            model: The model name to use.
            language: The language code.
            temperature: The sampling temperature.
            
        Returns:
            The transcribed text.
        """
        ...

class GroqAudioClient:
    """
    Implementation of AudioClient using Groq's API.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass it to the constructor or set GROQ_API_KEY environment variable."
            )
        self.client = Groq(api_key=self.api_key)

    def transcribe(
        self, 
        audio_file: Any, 
        model: str, 
        language: str, 
        temperature: float
    ) -> str:
        try:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=model,
                language=language,
                temperature=temperature,
                response_format="text"
            )
            return response
        except Exception as e:
            raise Exception(f"Groq transcription failed: {str(e)}")
