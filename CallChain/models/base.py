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

class PromptTemplate(Protocol):
    """
    Protocol defining the interface for Prompt Templates.
    """
    def format(self, **kwargs) -> str:
        ...

class StringPromptTemplate:
    """
    A simple prompt template that uses Python's string formatting.
    """
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """
        Format the template with the given variables.
        """
        # This replaces {key} in the template with the value from kwargs
        return self.template.format(**kwargs)