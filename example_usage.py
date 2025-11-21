import os
from CallChain import Chain, OpenAIModel, GroqModel
from CallChain.audio import AudioTranscriber
from load_dotenv import load_dotenv

load_dotenv()

def main():
    """
    Demonstrate the usage of CallChain.
    """
    # Ensure API keys are set
    # if not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"):
    #     print("Please set OPENAI_API_KEY or GROQ_API_KEY environment variables.")
    #     return
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"):
        print("Please set OPENAI_API_KEY or GROQ_API_KEY environment variables.")
        return

    print("Initializing Chain...")
    chain = Chain()

    # Add steps to the chain
    # Note: You can mix and match models if you have keys for both.
    # Here we use OpenAI for demonstration, but you can swap with GroqModel().
    
    try:
        # model = OpenAIModel(model_name="gpt-4o")
        model = GroqModel(model_name="openai/gpt-oss-20b") 
        
        chain.step(
            name="intro",
            model=model,
            PromptTemplate="Write a short, one-sentence greeting for {name}."
        )
        
        chain.step(
            name="translation",
            model=model,
            PromptTemplate="Translate the following greeting to Spanish: {intro}"
        )
        
        print("Running Chain...")
        results = chain.run(name="Alice")
        
        print("\nResults:")
        for step_name, output in results.items():
            print(f"[{step_name}]: {output}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Audio Transcription Test ---")
    # Replace with your actual audio file path
    AUDIO_FILE_PATH = "temp__\satisfaction_analysis.mp3" 
    
    if os.path.exists(AUDIO_FILE_PATH):
        try:
            print(f"Transcribing {AUDIO_FILE_PATH}...")
            transcriber = AudioTranscriber()
            text = transcriber.transcribe(AUDIO_FILE_PATH)
            print("Transcription Result:")
            print(text)
        except Exception as e:
            print(f"Audio transcription failed: {e}")
    else:
        print(f"Audio file not found at: {AUDIO_FILE_PATH}")
        print("Please update AUDIO_FILE_PATH in example_usage.py to test audio transcription.")

if __name__ == "__main__":
    main()
