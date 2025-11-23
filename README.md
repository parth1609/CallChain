# CallChain 🔗

**CallChain** is a lightweight, modular Python library designed to simplify the integration of Large Language Models (LLMs) and audio processing into your applications. It provides a clean interface for chaining LLM steps and transcribing audio with advanced preprocessing.

## 🚀 Why Use CallChain?

- **Sequential Chaining**: Build complex workflows where the output of one step feeds into the next.
- **Audio Powerhouse**: Built-in audio preprocessing (resampling, normalization, noise reduction) for superior transcription accuracy.
- **Type-Safe**: Fully typed codebase with Protocol-based interfaces for better developer experience.
- **Extensible**: Create custom clients or models by implementing simple protocols.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🛠️ How to Use

### 1. Basic LLM Chain

Create a sequence of steps to process information.

```python
from CallChain import Chain, GroqModel
from dotenv import load_dotenv

load_dotenv()

# Initialize Model
model = GroqModel(model_name="openai/gpt-oss-20b")

# Create Chain
chain = Chain()

# Step 1: Analyze a topic
chain.step(
    "analysis",
    model,
    "Analyze this topic and provide 3 key points: {topic}"
)

# Step 2: Relate to another field (uses output from 'analysis')
chain.step(
    "physics_relation",
    model,
    "Based on this analysis: {analysis}, how does it relate to physics?"
)

# Run
result = chain.run(topic="benefits of renewable energy")
print(result["physics_relation"])
```

### 2. Audio Transcription

Transcribe audio with automatic preprocessing.

```python
from CallChain.audio import AudioTranscriber, AudioConfig

# Configure Preprocessing
config = AudioConfig(
    model="whisper-large-v3-turbo",
    normalize=True,       # Normalize volume
    trim_silence=True,    # Remove silence
    noise_reduction=True  # Reduce background noise
)

# Initialize Transcriber
transcriber = AudioTranscriber(config=config)

# Transcribe
text = transcriber.transcribe("path/to/audio.mp3")
print(text)
```

### 3. Custom Audio Client

Inject your own client implementation.

```python
from CallChain.audio import AudioTranscriber

# Your custom client
class MyCustomClient:
    def transcribe(self, audio_file, model, language, temperature):
        return "Custom transcription result"

# Inject it
transcriber = AudioTranscriber(client=MyCustomClient())
print(transcriber.transcribe("audio.wav"))
```

## 📂 Project Structure

- `CallChain/core`: Core logic for Chains.
- `CallChain/models`: LLM wrappers (Groq, OpenAI).
- `CallChain/audio`: Audio processing and transcription.

## 📄 License

MIT
