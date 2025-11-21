import os
from pathlib import Path
from typing import Optional
from groq import Groq


class AudioTranscriber:
    """
    A simple audio transcriber using Groq's Whisper model.
    
    Example:
        # Initialize with your API key
        transcriber = AudioTranscriber(api_key="your_api_key")
        
        # Transcribe an audio file
        text = transcriber.transcribe("audio.mp3")
        print(text)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3-turbo",
        language: str = "en",
        temperature: float = 0.0
    ):
        """
        Initialize the AudioTranscriber.
        
        Args:
            api_key: Your Groq API key. If not provided, will try to load from GROQ_API_KEY environment variable.
            model: The Whisper model to use for transcription.
            language: Language of the audio (ISO 639-1 code).
            temperature: Sampling temperature (0.0 to 1.0).
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass it to the constructor or set GROQ_API_KEY environment variable."
            )
            
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.language = language
        self.temperature = temperature
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file to transcribe.
            
        Returns:
            The transcribed text.
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            Exception: For any errors during the API call.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        with open(audio_path, "rb") as audio_file:
            try:
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model,
                    language=self.language,
                    temperature=self.temperature,
                    response_format="text"  # Get plain text directly
                )
                return response
            except Exception as e:
                raise Exception(f"Error during transcription: {str(e)}")


# Example usage
# if __name__ == "__main__":
#     # Initialize with API key from environment variable
#     transcriber = AudioTranscriber()
    
#     # Or initialize with API key directly
#     # transcriber = AudioTranscriber(api_key="your_api_key_here")
    
#     # Transcribe an audio file
#     try:
#         text = transcriber.transcribe("satisfaction_analysis.mp3")
#         print("Transcription:")
#         print(text)
#     except Exception as e:
#         print(f"Error: {e}")