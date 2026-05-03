"""
Script để build FAISS index từ database SQLite.
Chạy 1 lần để nạp dữ liệu vào Vector DB trước khi dùng pipeline.

Usage:
    python build_faiss_index.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flow_2.flow2_manager import Flow2Manager


def main():
    db_path = "ds2.db"
    index_dir = "flow_2/faiss_data"
    embedding_model = "all-MiniLM-L6-v2"

    if not os.path.exists(db_path):
        print(f"❌ Không tìm thấy database: {db_path}")
        print("   Hãy đảm bảo file ds2.db nằm cùng thư mục.")
        return

    print("=" * 50)
    print("  BUILD FAISS INDEX — DS2 Product Database")
    print("=" * 50)

    flow2 = Flow2Manager(
        index_dir=index_dir,
        embedding_model=embedding_model,
    )
    flow2.build_index_from_db(db_path)

    print(f"\n✅ Build thành công!")
    print(f"   Index directory: {index_dir}")
    print(f"   Total vectors: {flow2._vector_store.size}")

    # Quick test
    print("\n--- Quick Search Test ---")
    results = flow2.retrieve("Action movie", top_k=3)
    for item in results:
        print(f"   🎬 {item.title} ({item.category_name}) - ${item.price:.2f} [score={item.similarity_score:.3f}]")


if __name__ == "__main__":
    main()
