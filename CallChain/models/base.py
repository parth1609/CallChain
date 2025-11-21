from typing import Protocol

class Model(Protocol):
    """
    Protocol defining the interface for Language Models.
    
    Any class implementing this protocol must provide a generate method
    that takes a prompt string and returns a response string.
    """
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input text to send to the model.
            
        Returns:
            The generated text response.
        """
        ...
