import os
from typing import Optional
from .base import Model

class GroqModel:
    """
    A wrapper for Groq's high-speed inference API.
    
    This class handles the connection to Groq and provides a simple
    interface for generating text.
    """
    
    def __init__(self, model_name: str = "llama3-8b-8192", api_key: Optional[str] = None):
        """
        Initialize the GroqModel.
        
        Args:
            model_name: The name of the model to use (default: "llama3-8b-8192").
            api_key: Groq API key. If None, loads from GROQ_API_KEY env var.
            
        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass it to the constructor or set GROQ_API_KEY environment variable."
            )
            
        from groq import Groq
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Generate text using Groq's API.
        
        Args:
            prompt: The user prompt.
            
        Returns:
            The content of the model's response.
            
        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise Exception(f"Error generating response from Groq: {str(e)}")

# Example usage
# if __name__ == "__main__":
#     try:
#         model = GroqModel()
#         print(model.generate("Say hello!"))
#     except Exception as e:
#         print(f"Error: {e}")
