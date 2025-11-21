from typing import Dict, Any, List
from ..models.base import Model

class Chain:
    """
    A class to manage and execute a sequence of LLM steps.
    
    The Chain class allows you to define a series of steps, where each step
    uses a model to generate text based on a template. The output of previous
    steps can be used in subsequent steps.
    """
    
    def __init__(self):
        """Initialize an empty Chain."""
        self.steps: List[Dict[str, Any]] = []

    def step(self, name: str, model: Model, template: str) -> 'Chain':
        """
        Add a step to the chain.
        
        Args:
            name: The name of the step (used as key in results).
            model: The LLM model to use for this step.
            template: The prompt template string (e.g., "Hello {name}").
            
        Returns:
            The Chain instance itself (for method chaining).
        """
        self.steps.append({
            "name": name,
            "model": model,
            "template": template
        })
        return self

    def run(self, **kwargs) -> Dict[str, str]:
        """
        Execute the chain with the given initial context.
        
        Args:
            **kwargs: Initial variables for the prompt templates.
            
        Returns:
            A dictionary containing the output of each step.
            
        Raises:
            ValueError: If a required variable is missing from the context.
            Exception: If a model fails to generate a response.
        """
        context = kwargs.copy()
        results = {}
        
        for step in self.steps:
            # Format the template with current context
            try:
                prompt = step["template"].format(**context)
            except KeyError as e:
                raise ValueError(f"Missing variable {e} for step '{step['name']}'")
            
            # Generate response
            try:
                output = step["model"].generate(prompt)
            except Exception as e:
                raise Exception(f"Step '{step['name']}' failed: {str(e)}")
            
            # Update context and results
            results[step["name"]] = output
            context[step["name"]] = output
            
        return results

# Example usage
# if __name__ == "__main__":
#     from ..models.openai import OpenAIModel
#     try:
#         chain = (
#             Chain()
#             .step("intro", OpenAIModel(), "Say hello to {name}")
#         )
#         print(chain.run(name="World"))
#     except Exception as e:
#         print(f"Error: {e}")
