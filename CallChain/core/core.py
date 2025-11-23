from typing import Dict, Any, List, Union
from CallChain.models.base import Model, StringPromptTemplate, PromptTemplate

class Chain:
    """
    A class to manage and execute a sequence of LLM steps.
    
    The Chain class allows you to define a series of steps, where each step
    uses a model to generate text based on a PromptTemplate. The output of previous
    steps can be used in subsequent steps.
    """
    
    def __init__(self):
        """Initialize an empty Chain."""
        self.steps: List[Dict[str, Any]] = []

    def step(self, name: str, model: Model, prompt_template: Union[str, Any]) -> 'Chain':
        """
        Add a step to the chain.
        
        Args:
            name: The name of the step (used as key in results).
            model: The LLM model to use for this step.
            prompt_template: The prompt template (string or PromptTemplate object).
            
        Returns:
            The Chain instance itself (for method chaining).
        """
        # If user passes a string, wrap it in our StringPromptTemplate
        if isinstance(prompt_template, str):
            template_obj = StringPromptTemplate(prompt_template)
        else:
            template_obj = prompt_template 

        self.steps.append({
            "name": name,
            "model": model,
            "PromptTemplate": template_obj
        })
        return self

    def run(self, **kwargs) -> Dict[str, str]:
        """
        Execute the chain with the given initial context.
        
        Args:
            **kwargs: Initial variables for the prompt PromptTemplate.
            
        Returns:
            A dictionary containing the output of each step.
            
        Raises:
            ValueError: If a required variable is missing from the context.
            Exception: If a model fails to generate a response.
        """
        
        # Use a copy of kwargs to avoid modifying the original input dictionary
        context = kwargs.copy()
        results = {}
        
        for step in self.steps:
            # Format the template with current context
            try:
                prompt = step["PromptTemplate"].format(**context)
                print(f"--- Step: {step['name']} ---\nPrompt: {prompt}")
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
#     from CallChain.models.groq import GroqModel
#     from load_dotenv import load_dotenv
#     load_dotenv()

#     config = {
#         "name":"parth",
#         "location":"mumbai"
#     }

#     try:
#         chain = (
#             Chain()
#             .step("intro", GroqModel(model_name="openai/gpt-oss-20b"), 'suggest meaning of my name {name} and location {location}')
#         )
#         print(chain.run(**config))
#     except Exception as e:
#         print(f"Error: {e}")