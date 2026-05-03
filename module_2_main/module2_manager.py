"""
Module2Manager: Tổng hợp kết quả từ Module 1 (DataFrame) và Flow 2 (similar items)
để tạo câu trả lời tự nhiên qua LLM.
"""

import sys
import os
import pandas as pd
from typing import List, Optional, Iterator

# Thêm path để import llm module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import LLMGenerator, LLMConfig, LLMType
from .prompt_engine import PromptEngine


class Module2Manager:
    """
    Điều phối Module 2: Response Synthesizer.

    Logic xử lý:
    - Nếu df_result có dữ liệu: Trình bày sản phẩm + gợi ý thêm từ similar_items.
    - Nếu df_result rỗng/lỗi: Thông báo hết hàng + dùng similar_items làm giải pháp thay thế.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        prompt_template_path: str = None,
    ):
        """
        Args:
            llm_config: Cấu hình LLM (Gemini/Groq).
            prompt_template_path: Đường dẫn tới file system prompt (tùy chọn).
        """
        self._llm = LLMGenerator(llm_config)
        self._prompt_engine = PromptEngine(prompt_template_path)

    def synthesize_search_response(
        self,
        user_query: str,
        df_result: Optional[pd.DataFrame],
        similar_items: list = None,
    ) -> str:
        """
        Sinh câu trả lời cho intent SEARCH.

        Args:
            user_query: Câu hỏi gốc của user.
            df_result: DataFrame từ Module 1 (Black Box).
            similar_items: List[ProductContext] từ Flow 2.

        Returns:
            Câu trả lời tự nhiên từ LLM.
        """
        similar_items = similar_items or []

        # Build prompt
        user_prompt = self._prompt_engine.build_search_prompt(
            user_query=user_query,
            df_result=df_result,
            similar_items=similar_items,
        )

        # Gọi LLM
        response = self._llm.generate_text(
            user_prompt=user_prompt,
            system_prompt=self._prompt_engine.system_prompt,
        )
        return response.content.strip()

    def synthesize_search_response_stream(
        self,
        user_query: str,
        df_result: Optional[pd.DataFrame],
        similar_items: list = None,
    ) -> Iterator[str]:
        """
        Sinh câu trả lời streaming cho intent SEARCH (dùng với Streamlit st.write_stream).
        """
        similar_items = similar_items or []

        user_prompt = self._prompt_engine.build_search_prompt(
            user_query=user_query,
            df_result=df_result,
            similar_items=similar_items,
        )

        return self._llm.generate_text_stream(
            user_prompt=user_prompt,
            system_prompt=self._prompt_engine.system_prompt,
        )

    def synthesize_analytic_response(
        self,
        user_query: str,
        df_result: Optional[pd.DataFrame],
    ) -> str:
        """
        Sinh câu trả lời cho intent ANALYTIC.

        Args:
            user_query: Câu hỏi phân tích gốc của user.
            df_result: DataFrame từ Module 1.

        Returns:
            Câu trả lời phân tích từ LLM.
        """
        user_prompt = self._prompt_engine.build_analytic_prompt(
            user_query=user_query,
            df_result=df_result,
        )

        response = self._llm.generate_text(
            user_prompt=user_prompt,
            system_prompt=self._prompt_engine.system_prompt,
        )
        return response.content.strip()

    def synthesize_analytic_response_stream(
        self,
        user_query: str,
        df_result: Optional[pd.DataFrame],
    ) -> Iterator[str]:
        """
        Sinh câu trả lời streaming cho intent ANALYTIC.
        """
        user_prompt = self._prompt_engine.build_analytic_prompt(
            user_query=user_query,
            df_result=df_result,
        )

        return self._llm.generate_text_stream(
            user_prompt=user_prompt,
            system_prompt=self._prompt_engine.system_prompt,
        )
