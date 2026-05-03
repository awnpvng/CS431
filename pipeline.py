"""
Pipeline Orchestrator: Điều phối toàn bộ hệ thống Text-to-SQL + Recommendation.

Quy trình:
  1. InsightReader → Phân loại intent (SEARCH / ANALYTIC)
  2. Nếu ANALYTIC: Module 1 → DataFrame → Module 2 → Câu trả lời
  3. Nếu SEARCH:   Module 1 + Flow 2 (song song) → Gộp kết quả → Module 2 → Câu trả lời
"""

import sys
import os
import concurrent.futures
from dotenv import load_dotenv

# Thêm paths
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.getcwd(), "module_1_model"))
sys.path.append(os.path.join(os.getcwd(), "module_1_main"))

# Load environment variables
load_dotenv()

from llm import LLMGenerator, LLMConfig, LLMType
from module_0_insight.insight_reader import InsightReader
from module_1_model.database import Metadata
from module_1_main.sql_generator import SQLGenerator
from module_1_main.sql_executor import SQLExecutor, DatabaseInfor, DialectType
from flow_2.flow2_manager import Flow2Manager
from module_2_main.module2_manager import Module2Manager


class PipelineOrchestrator:
    """
    Orchestrator chính: Kết nối tất cả các module.

    Usage:
        pipeline = PipelineOrchestrator()
        answer = pipeline.run("Tìm phim hành động giá dưới 20 đô")
        print(answer)
    """

    def __init__(
        self,
        db_path: str = "ds2.db",
        metadata_path: str = "metadata.json",
        model_path: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        faiss_index_dir: str = "flow_2/faiss_data",
        gemini_api_key: str = None,
        gemini_model_name: str = None,
    ):
        # --- Config ---
        self._db_path = db_path
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "").strip()
        model_name = gemini_model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite-preview").strip()

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found! Set it in .env or pass directly.")

        self._llm_config = LLMConfig(
            llm_type=LLMType.GEMINI,
            api_key=api_key,
            model_name=model_name,
        )

        # --- Module 0: Intent Reader ---
        print("[Pipeline] Khởi tạo InsightReader...")
        self._insight_reader = InsightReader(self._llm_config)

        # --- Module 1: Text-to-SQL (Black Box) ---
        print("[Pipeline] Khởi tạo Module 1 (Text-to-SQL)...")
        self._schema_summary, self._db_id = self._load_schema(metadata_path)

        if model_path is None:
            model_path = os.path.join(
                os.getcwd(), "module_1_model", "finetune", "Qwen2.5_Coder_3B_Instruct_adapter"
            )
        self._sql_generator = SQLGenerator(model_path)

        self._db_infor = DatabaseInfor(
            dialect_type=DialectType.SQLITE,
            config={"database_path": db_path},
        )
        self._sql_executor = SQLExecutor(
            connection_string=self._db_infor.get_connection_string
        )

        # --- Flow 2: RAG Retrieval ---
        print("[Pipeline] Khởi tạo Flow 2 (RAG Retrieval)...")
        self._flow2 = Flow2Manager(
            index_dir=faiss_index_dir,
            embedding_model=embedding_model,
        )
        self._flow2.ensure_index(db_path)

        # --- Module 2: Response Synthesizer ---
        print("[Pipeline] Khởi tạo Module 2 (Response Synthesizer)...")
        rec_prompt_path = os.path.join("module_2_main", "rec_prompt.txt")
        self._module2 = Module2Manager(
            llm_config=self._llm_config,
            prompt_template_path=rec_prompt_path,
        )

        print(f"[Pipeline] ✅ Khởi tạo hoàn tất! Database: {self._db_id}")

    # =====================================================================
    # MAIN RUN
    # =====================================================================

    def run(self, user_query: str) -> dict:
        """
        Chạy toàn bộ pipeline.

        Returns:
            dict với keys:
            - 'intent': SEARCH | ANALYTIC
            - 'answer': Câu trả lời tự nhiên
            - 'df_result': DataFrame (nếu có)
            - 'similar_items': List[ProductContext] (nếu có)
            - 'sql_steps': Dict từ Module 1 (nếu có)
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline] Câu hỏi: {user_query}")
        print(f"{'='*60}")

        # Step 1: Phân loại intent
        print("[Pipeline] Step 1: Phân loại intent...")
        intent = self._insight_reader.classify(user_query)
        print(f"[Pipeline] Intent = {intent}")

        if intent == "ANALYTIC":
            return self._run_analytic(user_query, intent)
        else:  # SEARCH
            return self._run_search(user_query, intent)

    def _run_analytic(self, user_query: str, intent: str) -> dict:
        """
        Intent ANALYTIC: Module 1 → DataFrame → Module 2 → Answer.
        Không cần Flow 2.
        """
        print("[Pipeline] Step 2: Chạy Module 1 (Text-to-SQL)...")
        sql_steps, df_result = self._execute_module1(user_query)

        print("[Pipeline] Step 3: Chạy Module 2 (Sinh câu trả lời)...")
        answer = self._module2.synthesize_analytic_response(
            user_query=user_query,
            df_result=df_result,
        )

        return {
            "intent": intent,
            "answer": answer,
            "df_result": df_result,
            "similar_items": [],
            "sql_steps": sql_steps,
        }

    def _run_search(self, user_query: str, intent: str) -> dict:
        """
        Intent SEARCH: Module 1 + Flow 2 (song song) → Module 2 → Answer.
        """
        print("[Pipeline] Step 2: Chạy SONG SONG Module 1 + Flow 2...")

        df_result = None
        sql_steps = None
        similar_items = []

        # Chạy song song Module 1 và Flow 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_module1 = executor.submit(self._execute_module1, user_query)
            future_flow2 = executor.submit(self._flow2.retrieve, user_query)

            # Thu kết quả Module 1
            try:
                sql_steps, df_result = future_module1.result(timeout=120)
                print(f"[Pipeline] Module 1 hoàn tất. Rows = {len(df_result) if hasattr(df_result, '__len__') else 'N/A'}")
            except Exception as e:
                print(f"[Pipeline] Module 1 ERROR: {e}")

            # Thu kết quả Flow 2
            try:
                similar_items = future_flow2.result(timeout=60)
                print(f"[Pipeline] Flow 2 hoàn tất. Similar items = {len(similar_items)}")
            except Exception as e:
                print(f"[Pipeline] Flow 2 ERROR: {e}")

        # Step 3: Module 2 — Tổng hợp
        print("[Pipeline] Step 3: Chạy Module 2 (Sinh câu trả lời)...")
        answer = self._module2.synthesize_search_response(
            user_query=user_query,
            df_result=df_result,
            similar_items=similar_items,
        )

        return {
            "intent": intent,
            "answer": answer,
            "df_result": df_result,
            "similar_items": similar_items,
            "sql_steps": sql_steps,
        }

    # =====================================================================
    # PRIVATE HELPERS
    # =====================================================================

    def _execute_module1(self, user_query: str):
        """Chạy Module 1 (Black Box): Generate SQL → Execute → DataFrame."""
        if not self._sql_executor.connect_database():
            print("[Pipeline] Lỗi kết nối database!")
            return None, None

        try:
            gen_result = self._sql_generator.generate(user_query, self._schema_summary)
            sql_query = gen_result.get("sql", "N/A")
            print(f"[Pipeline] SQL: {sql_query}")
            df_result = self._sql_executor.execute_query(sql_query)
            return gen_result, df_result
        except Exception as e:
            print(f"[Pipeline] Module 1 error: {e}")
            return None, None
        finally:
            self._sql_executor.close_connection()

    @staticmethod
    def _load_schema(filepath: str):
        """Load rich schema summary từ metadata.json."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        db_name = data.get("database_name", "ds2")
        summary = f"Database Schema for '{db_name}' with Rich Descriptions:\n\n"

        tables = data.get("schemas", [])[0].get("tables", [])
        for table in tables:
            t_name = table["table_name"]
            t_desc = table.get("description", "")
            summary += f"Table [{t_name}]: {t_desc}\n"

            for col in table.get("columns", []):
                c_name = col["name"]
                c_type = col.get("type", "unknown")
                c_desc = col.get("description", "")
                summary += f"  - {c_name} ({c_type}): {c_desc}\n"

                if "value" in col:
                    val_mapping = ", ".join(
                        [f"'{k}'={v}" for k, v in col["value"].items()]
                    )
                    summary += f"      (Value Map: {val_mapping})\n"

            pks = table.get("primary_key", [])
            if pks:
                summary += f"  => Primary Key: {', '.join(pks)}\n"

            bls = table.get("business_logic", [])
            if bls:
                summary += f"  => Business Rules: {'; '.join(bls)}\n"
            summary += "\n"

        relations = data.get("relations", [])
        if relations:
            summary += "Relationships (Foreign Keys):\n"
            for rel in relations:
                t1, c1 = rel["source_table"], rel["source_column"]
                t2, c2 = rel["target_table"], rel["target_column"]
                summary += f"- {t1}.{c1} = {t2}.{c2}\n"

        return summary, db_name


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DS2 STORE — Text-to-SQL + Recommendation Pipeline")
    print("=" * 60)

    pipeline = PipelineOrchestrator()

    # Test queries
    test_queries = [
        "Tìm phim thể loại Action có giá dưới 20 đô",
        "Diễn viên nào có doanh thu cao nhất trong thể loại Comedy?",
        "Phim nào bán chạy nhất?",
    ]

    for query in test_queries:
        result = pipeline.run(query)
        print(f"\n{'─'*60}")
        print(f"📌 Intent: {result['intent']}")
        print(f"💬 Answer:\n{result['answer']}")
        if result.get("similar_items"):
            print(f"\n🔍 Similar items ({len(result['similar_items'])}):")
            for item in result["similar_items"]:
                print(f"   - {item.title} ({item.category_name}) - ${item.price:.2f}")
        print(f"{'─'*60}\n")
