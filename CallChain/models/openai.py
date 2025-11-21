import os
from typing import Optional
from .base import Model

class OpenAIModel:
    """
    A wrapper for OpenAI's chat completion API.
    
    This class handles the connection to OpenAI and provides a simple
    interface for generating text.
    """
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the OpenAIModel.
        
        Args:
            model_name: The name of the model to use (default: "gpt-4o").
            api_key: OpenAI API key. If None, loads from OPENAI_API_KEY env var.
            
        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Either pass it to the constructor or set OPENAI_API_KEY environment variable."
            )
            
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Generate text using OpenAI's API.
        
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
            raise Exception(f"Error generating response from OpenAI: {str(e)}")

# Example usage
# if __name__ == "__main__":
#     try:
#         model = OpenAIModel()
#         print(model.generate("Say hello!"))
#     except Exception as e:
#         print(f"Error: {e}")
