from typing import Optional, Iterator
import google.generativeai as genai
from .llm_generator import LLMClient, LLMResponse, LLMType

class GeminiClient(LLMClient):
    def generate_text(self, user_prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            genai.configure(api_key=self.config.api_key)
            model = genai.GenerativeModel(
                model_name=self.config.model_name,
                system_instruction=system_prompt
            )
            response = model.generate_content(user_prompt)
            return LLMResponse(
                content=response.text,
                llm_type=LLMType.GEMINI,
                model_name=self.config.model_name
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def generate_text_stream(self, user_prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        try:
            genai.configure(api_key=self.config.api_key)
            model = genai.GenerativeModel(
                model_name=self.config.model_name,
                system_instruction=system_prompt
            )
            response = model.generate_content(user_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise RuntimeError(f"Gemini stream error: {str(e)}")
