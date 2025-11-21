# CallChain

CallChain is a lightweight, professional Python library for chaining Large Language Model (LLM) calls. It provides a clean, composable API to build complex workflows by connecting multiple LLM steps together.

## Core Concepts

The library is built around three main abstractions:

### 1. `Chain` (`CallChain/core/core.py`)

The `Chain` class is the orchestrator. It manages:
- **Sequence of Steps**: It holds a list of steps to be executed in order.
- **Context Management**: It maintains a dictionary of variables (the "context") that flows through the chain.
- **Execution**: The `run()` method executes each step, updating the context with the results.

**Why this abstraction?**
By encapsulating the flow logic in `Chain`, we decouple the *structure* of the workflow from the *execution* details. This allows you to define a chain once and run it multiple times with different inputs.

### 2. `Model` Protocol (`CallChain/models/base.py`)

This is a Python `Protocol` (interface) that defines what a "Model" must look like. It enforces a single method: `generate(prompt: str) -> str`.

**Why this abstraction?**
This allows `CallChain` to be **agnostic** to the underlying LLM provider. You can swap OpenAI for Groq, Anthropic, or a local model without changing your chain logic. As long as the class implements `generate`, it works.

### 3. `Concrete Models` (`CallChain/models/`)

We provide concrete implementations of the `Model` protocol:
- **`OpenAIModel`** (`CallChain/models/openai.py`): Connects to OpenAI's API.
- **`GroqModel`** (`CallChain/models/groq.py`): Connects to Groq's high-speed API.

These classes handle the specific API details (authentication, request formatting, error handling), exposing a clean `generate` method to the `Chain`.

## Directory Structure

```
CallChain/
├── CallChain/
│   ├── __init__.py          # Exports the main classes
│   ├── core/
│   │   └── core.py          # Contains the Chain class logic
│   └── models/
│       ├── base.py          # Defines the Model protocol
│       ├── openai.py        # OpenAI implementation
│       └── groq.py          # Groq implementation
├── example_usage.py         # Demo script
├── pyproject.toml           # Build configuration
└── README.md                # This file
```

## Usage

```python
from CallChain import Chain, OpenAIModel

# 1. Initialize the chain
chain = Chain()

# 2. Define steps
# The output of 'intro' becomes available as {intro} in subsequent steps
chain.step(
    name="intro",
    model=OpenAIModel(),
    template="Write a greeting for {name}."
)

chain.step(
    name="translation",
    model=OpenAIModel(),
    template="Translate this to Spanish: {intro}"
)

# 3. Run the chain
results = chain.run(name="Alice")
print(results["translation"])
```
