"""
PromptEngine: Lắp ghép user_query, df_result, similar_items vào prompt template
để gửi cho LLM sinh câu trả lời tự nhiên.
"""

import os
import pandas as pd
from typing import List, Optional

# Sử dụng import tương đối cho ProductContext nếu cần type hint
# nhưng để tránh circular import, dùng string annotation


class PromptEngine:
    """
    Xây dựng prompt tối ưu dựa trên intent, kết quả SQL, và sản phẩm tương tự.
    """

    def __init__(self, prompt_template_path: str = None):
        """
        Args:
            prompt_template_path: Đường dẫn tới file prompt template (tùy chọn).
        """
        self._template_path = prompt_template_path
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt từ file hoặc dùng mặc định."""
        if self._template_path and os.path.exists(self._template_path):
            with open(self._template_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        return self._default_system_prompt()

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "Bạn là trợ lý tư vấn phim DVD chuyên nghiệp của cửa hàng DS2 Store. "
            "Nhiệm vụ của bạn là trả lời câu hỏi của khách hàng một cách tự nhiên, thân thiện và hữu ích. "
            "Hãy trình bày thông tin rõ ràng, dễ đọc. "
            "Nếu có gợi ý sản phẩm tương tự, hãy trình bày chúng như một cố vấn am hiểu điện ảnh. "
            "Luôn trả lời bằng tiếng Việt."
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    # =====================================================================
    # BUILD PROMPT CHO TRƯỜNG HỢP SEARCH (có df_result + similar_items)
    # =====================================================================

    def build_search_prompt(
        self,
        user_query: str,
        df_result: Optional[pd.DataFrame],
        similar_items: list,
    ) -> str:
        """
        Build prompt cho intent SEARCH.
        Xử lý 2 case:
          - df_result có dữ liệu → trình bày + gợi ý thêm
          - df_result rỗng → thông báo hết hàng + gợi ý thay thế
        """
        has_data = self._has_data(df_result)

        parts = [f'Câu hỏi của khách hàng: "{user_query}"\n']

        if has_data:
            parts.append("--- KẾT QUẢ TÌM KIẾM TỪ DATABASE ---")
            parts.append(self._format_dataframe(df_result))
            parts.append("")
            parts.append(
                "Hãy trình bày kết quả tìm kiếm ở trên cho khách hàng một cách tự nhiên và dễ đọc."
            )
        else:
            parts.append("--- KẾT QUẢ TÌM KIẾM ---")
            parts.append("Không tìm thấy sản phẩm nào khớp chính xác với yêu cầu trong database.")
            parts.append("")
            parts.append(
                "Hãy thông báo lịch sự rằng sản phẩm khách tìm hiện không có/hết hàng."
            )

        if similar_items:
            parts.append("")
            parts.append("--- SẢN PHẨM TƯƠNG TỰ (GỢI Ý) ---")
            parts.append(self._format_similar_items(similar_items))
            parts.append("")

            if has_data:
                parts.append(
                    "Ngoài kết quả chính, hãy gợi ý thêm các sản phẩm tương tự ở trên "
                    "với giọng điệu như một người am hiểu điện ảnh, giải thích vì sao chúng phù hợp."
                )
            else:
                parts.append(
                    "Vì không tìm thấy sản phẩm chính xác, hãy sử dụng các sản phẩm tương tự ở trên "
                    "làm giải pháp thay thế. Trình bày chúng như những lựa chọn hấp dẫn mà khách có thể quan tâm."
                )

        return "\n".join(parts)

    # =====================================================================
    # BUILD PROMPT CHO TRƯỜNG HỢP ANALYTIC (chỉ df_result)
    # =====================================================================

    def build_analytic_prompt(
        self,
        user_query: str,
        df_result: Optional[pd.DataFrame],
    ) -> str:
        """
        Build prompt cho intent ANALYTIC.
        Dữ liệu phân tích từ Module 1, không cần gợi ý sản phẩm.
        """
        parts = [f'Câu hỏi phân tích của khách hàng: "{user_query}"\n']

        if self._has_data(df_result):
            parts.append("--- KẾT QUẢ PHÂN TÍCH TỪ DATABASE ---")
            parts.append(self._format_dataframe(df_result))
            parts.append("")
            parts.append(
                "Hãy phân tích và trình bày kết quả trên cho khách hàng. "
                "Giải thích ý nghĩa của dữ liệu, rút ra nhận xét/xu hướng nếu có."
            )
        else:
            parts.append(
                "Không có dữ liệu phân tích từ database. "
                "Hãy thông báo lịch sự rằng hệ thống không thể lấy được dữ liệu cho câu hỏi này "
                "và gợi ý khách hàng thử hỏi theo cách khác."
            )

        return "\n".join(parts)

    # =====================================================================
    # PRIVATE HELPERS
    # =====================================================================

    @staticmethod
    def _has_data(df: Optional[pd.DataFrame]) -> bool:
        """Kiểm tra DataFrame có dữ liệu không."""
        if df is None:
            return False
        if isinstance(df, str):  # Trường hợp SQL error
            return False
        return len(df) > 0

    @staticmethod
    def _format_dataframe(df: pd.DataFrame) -> str:
        """Format DataFrame thành text dễ đọc cho LLM."""
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return "(Không có dữ liệu)"

        # Giới hạn hiển thị max 20 dòng để tránh token quá dài
        display_df = df.head(20)
        result = display_df.to_string(index=False)

        if len(df) > 20:
            result += f"\n... và {len(df) - 20} dòng nữa (tổng cộng {len(df)} dòng)"

        return result

    @staticmethod
    def _format_similar_items(similar_items: list) -> str:
        """Format list ProductContext thành text cho prompt."""
        lines = []
        for i, item in enumerate(similar_items, 1):
            # Hỗ trợ cả dict và ProductContext dataclass
            if hasattr(item, "title"):
                lines.append(
                    f"{i}. [{item.title}]\n"
                    f"   - Diễn viên: {item.actor}\n"
                    f"   - Thể loại: {item.category_name}\n"
                    f"   - Giá: ${item.price:.2f}\n"
                    f"   - Tồn kho: {item.quan_in_stock}\n"
                    f"   - Lý do gợi ý: {item.relevance_reason}\n"
                    f"   - Độ tương đồng: {item.similarity_score:.3f}"
                )
            elif isinstance(item, dict):
                lines.append(
                    f"{i}. [{item.get('title', 'N/A')}]\n"
                    f"   - Diễn viên: {item.get('actor', 'N/A')}\n"
                    f"   - Thể loại: {item.get('category_name', 'N/A')}\n"
                    f"   - Giá: ${item.get('price', 0):.2f}\n"
                    f"   - Tồn kho: {item.get('quan_in_stock', 0)}\n"
                    f"   - Lý do: {item.get('relevance_reason', 'Tương tự')}"
                )
        return "\n".join(lines)
