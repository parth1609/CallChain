import pytest
from unittest.mock import MagicMock
from CallChain import Chain, Model

class MockModel:
    def generate(self, prompt: str) -> str:
        return f"Mock response to: {prompt}"

def test_chain_execution():
    # Setup
    model = MockModel()
    chain = Chain()
    
    # Define chain
    chain.step("step1", model, "Hello {name}")
    chain.step("step2", model, "Echo {step1}")
    
    # Execute
    results = chain.run(name="World")
    
    # Verify
    assert results["step1"] == "Mock response to: Hello World"
    assert results["step2"] == "Mock response to: Echo Mock response to: Hello World"

def test_missing_variable():
    model = MockModel()
    chain = Chain()
    chain.step("step1", model, "Hello {missing}")
    
    with pytest.raises(ValueError, match="Missing variable"):
        chain.run(name="World")
