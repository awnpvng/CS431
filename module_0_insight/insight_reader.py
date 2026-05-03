"""
InsightReader: Phân loại intent từ user query (SEARCH / ANALYTIC).
Sử dụng LLM (Gemini) + prompt template từ insight_prompt.txt.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import LLMGenerator, LLMConfig


class InsightReader:
    """
    Đọc ý định (intent) từ câu hỏi của user.
    Trả về: 'SEARCH' hoặc 'ANALYTIC'.
    """

    def __init__(self, llm_config: LLMConfig, prompt_path: str = None):
        self._llm = LLMGenerator(llm_config)

        if prompt_path is None:
            prompt_path = os.path.join(
                os.path.dirname(__file__), "insight_prompt.txt"
            )
        self._prompt_template = self._load_prompt(prompt_path)

    @staticmethod
    def _load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def classify(self, user_query: str) -> str:
        """
        Phân loại intent từ user query.

        Returns:
            'SEARCH' hoặc 'ANALYTIC'
        """
        prompt = self._prompt_template.replace("{user_query}", user_query)

        response = self._llm.generate_text(user_prompt=prompt)
        result = response.content.strip().upper()

        # Sanitize output — chỉ chấp nhận SEARCH hoặc ANALYTIC
        if "ANALYTIC" in result:
            return "ANALYTIC"
        elif "SEARCH" in result:
            return "SEARCH"
        else:
            # Fallback: mặc định SEARCH nếu không xác định được
            print(f"[InsightReader] WARNING: Không xác định được intent '{result}', fallback SEARCH")
            return "SEARCH"
