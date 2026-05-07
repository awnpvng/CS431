"""
Flow2Manager: Điều phối luồng RAG Retrieval.
Nhận user_query → Embedding → Vector Search → List[ProductContext]
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .embedding_service import EmbeddingService
from .vector_store import VectorStoreConnector


@dataclass
class ProductContext:
    """
    Data contract cho mỗi sản phẩm tương tự được tìm thấy.
    """
    prod_id: int
    title: str
    actor: str
    category_name: str
    price: float
    quan_in_stock: int
    similarity_score: float = 0.0
    relevance_reason: str = ""  # "Cùng thể loại", "Cùng diễn viên", etc.


class Flow2Manager:
    """
    Điều phối Flow 2: RAG Retrieval Pipeline.

    Usage:
        flow2 = Flow2Manager(index_dir="flow_2/faiss_data")
        # Nạp dữ liệu lần đầu:
        flow2.build_index_from_db("ds2.db")
        # Hoặc load index đã build:
        flow2.load_index()
        # Tìm sản phẩm tương tự:
        results = flow2.retrieve("phim hành động")
    """

    def __init__(
        self,
        index_dir: str = "flow_2/faiss_data",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        self._index_dir = index_dir
        self._top_k = top_k
        self._embedding_service = EmbeddingService(model_name=embedding_model)
        self._vector_store: Optional[VectorStoreConnector] = None

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def build_index_from_db(self, db_path: str = "ds2.db") -> None:
        """
        Export dữ liệu sản phẩm từ SQLite → Tạo embeddings → Nạp vào FAISS.
        Chỉ cần chạy 1 lần, sau đó dùng load_index().
        """
        import sqlite3

        print("[Flow2] Đang export dữ liệu sản phẩm từ SQLite...")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query JOIN products + categories + inventory
        query = """
            SELECT
                p.prod_id,
                p.title,
                p.actor,
                c.categoryname,
                p.category,
                p.price,
                p.special,
                COALESCE(i.quan_in_stock, 0) AS quan_in_stock,
                COALESCE(i.sales, 0) AS sales
            FROM products p
            LEFT JOIN categories c ON p.category = c.category
            LEFT JOIN inventory i ON p.prod_id = i.prod_id
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        print(f"[Flow2] Tìm thấy {len(rows)} sản phẩm. Đang tạo embeddings...")

        # Tạo document text để embed — kết hợp title + actor + category
        documents = []
        metadata_list = []

        for row in rows:
            prod_id, title, actor, cat_name, cat_id, price, special, stock, sales = row
            cat_name = cat_name or "Unknown"
            actor = actor or "Unknown"
            title = title or "Unknown"

            # Composite text cho embedding (Enhance description with English conversational keywords)
            doc_text = f"A {cat_name} movie titled {title}, starring {actor}. Priced at ${price if price else 0}."
            documents.append(doc_text)

            metadata_list.append({
                "prod_id": prod_id,
                "title": title,
                "actor": actor,
                "category_name": cat_name,
                "category_id": cat_id,
                "price": float(price) if price else 0.0,
                "special": special,
                "quan_in_stock": stock,
                "sales": sales,
                "document_text": doc_text,
            })

        # Embed tất cả documents
        vectors = self._embedding_service.embed_documents(documents)

        # Khởi tạo Vector Store và nạp dữ liệu
        self._vector_store = VectorStoreConnector(
            dimension=self._embedding_service.dimension
        )
        self._vector_store.add_documents(vectors, metadata_list)

        # Lưu index
        self._vector_store.save(self._index_dir)
        print(f"[Flow2] Index built thành công! Total vectors: {self._vector_store.size}")

    def load_index(self) -> None:
        """Load FAISS index đã build trước đó."""
        if self._vector_store is None:
            self._vector_store = VectorStoreConnector(
                dimension=self._embedding_service.dimension
            )
        self._vector_store.load(self._index_dir)

    def is_index_ready(self) -> bool:
        """Kiểm tra xem index đã sẵn sàng chưa."""
        return (
            self._vector_store is not None
            and self._vector_store.size > 0
        )

    def ensure_index(self, db_path: str = "ds2.db") -> None:
        """Tự động build hoặc load index."""
        index_file = os.path.join(self._index_dir, "faiss_index.bin")
        if os.path.exists(index_file):
            self.load_index()
        else:
            print("[Flow2] Index chưa tồn tại, đang build từ database...")
            self.build_index_from_db(db_path)

    def retrieve(self, query: str, top_k: int = None) -> List[ProductContext]:
        """
        Nhận user_query → Embed → Search → Trả về List[ProductContext].

        Args:
            user_query: Câu hỏi/từ khóa tìm kiếm của user.
            top_k: Số lượng kết quả trả về (mặc định dùng giá trị init).

        Returns:
            List[ProductContext] chứa các sản phẩm tương tự.
        """
        if not self.is_index_ready():
            print("[Flow2] WARNING: Vector index chưa sẵn sàng!")
            return []

        k = top_k or self._top_k

        # 1. Embed query
        query_vector = self._embedding_service.embed_query(query)

        # 2. Search
        raw_results = self._vector_store.search(query_vector, top_k=k)

        # 3. Convert to ProductContext + Xác định relevance_reason (removed custom logic, let Gemini handle it)
        product_contexts = []
        for result in raw_results:
            ctx = ProductContext(
                prod_id=result.get("prod_id", 0),
                title=result.get("title", "N/A"),
                actor=result.get("actor", "N/A"),
                category_name=result.get("category_name", "N/A"),
                price=result.get("price", 0.0),
                quan_in_stock=result.get("quan_in_stock", 0),
                similarity_score=result.get("similarity_score", 0.0),
                relevance_reason="", # Leave empty so LLM generates the reason
            )
            product_contexts.append(ctx)

        return product_contexts

    def retrieve(self, query: str, top_k: int = None) -> List[ProductContext]:
        """
        Thực hiện tìm kiếm ngữ nghĩa cho user_query.
        """
        assert self.is_index_ready(), "[Flow2] Lỗi: Vector Index chưa được load."

        k = top_k or self._top_k
        query_vector = self._embedding_service.embed_query(query)

        # Chạy query truyền vào hàm search được upgrade RRF BM25
        search_results = self._vector_store.search(query=query, query_vector=query_vector, top_k=k)

        # Chuyển đổi thành domain objects
        context_list = []
        for res in search_results:
            ctx = ProductContext(
                prod_id=res["prod_id"],
                title=res["title"],
                actor=res["actor"],
                category_name=res["category_name"],
                price=res["price"],
                quan_in_stock=res["quan_in_stock"],
                similarity_score=res.get("rrf_score", res.get("similarity_score", 0.0)),
                relevance_reason=f"Vector Match (Score: {res.get('rrf_score', res.get('similarity_score', 0.0)):.2f})"
            )
            context_list.append(ctx)

        return context_list
