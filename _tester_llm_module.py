import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm import LLMGenerator, LLMConfig, LLMType

if __name__ == "__main__":
    with open("api_key.txt", "r", encoding="utf-8") as file:
        api_key = file.read().strip()

    # --- Test Gemini ---
    print("=" * 40)
    print("Testing Gemini...")
    gemini_config = LLMConfig(
        llm_type=LLMType.GEMINI,
        api_key=api_key,
        model_name="gemini-3.1-flash-lite-preview"
    )
    gemini = LLMGenerator(gemini_config)
    response = gemini.generate_text("What is the capital of Vietnam?")
    print(f"Content : {response.content.strip()}")
    print(f"Model   : {response.model_name}")
    print(f"Type    : {response.llm_type}")
