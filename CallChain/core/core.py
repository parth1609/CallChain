from typing import Dict, Any, List, Union
from CallChain.models.base import Model, StringPromptTemplate

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

    def step(self, name: str, model: Model, PromptTemplate: Union[str, Any]) -> 'Chain':
        """
        Add a step to the chain.
        
        Args:
            name: The name of the step (used as key in results).
            model: The LLM model to use for this step.
            PromptTemplate: The prompt template (string or PromptTemplate object).
            
        Returns:
            The Chain instance itself (for method chaining).
        """
        # If user passes a string, wrap it in our StringPromptTemplate
        if isinstance(PromptTemplate, str):
            template_obj = StringPromptTemplate(PromptTemplate)
        else:
            template_obj = PromptTemplate 

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
        
        # context = kwargs.copy()
        # results = {}
        
        for step in self.steps:
            # Format the PromptTemplate with current context
            try:
                prompt = step["PromptTemplate"].format(**kwargs)
                print(f"--- Step: {step['name']} ---\nPrompt: {prompt}")
            except KeyError as e:
                raise ValueError(f"Missing variable {e}")
            
            # Generate response
            try:
                output = step["model"].generate(prompt)
            except Exception as e:
                raise Exception(f"failed: {str(e)}")
            
            
        return output

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