"""
VectorStoreConnector: Kết nối FAISS Vector Database.
Component-based — có thể swap sang ChromaDB, Pinecone, etc.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """Abstract base class cho Vector Store."""

    @abstractmethod
    def add(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        pass

    @abstractmethod
    def load(self, directory: str) -> None:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based Vector Store.
    Lưu trữ vectors + metadata song song.
    """

    def __init__(self, dimension: int):
        import faiss
        self._dimension = dimension
        # IndexFlatIP cho cosine similarity (vectors phải được normalize trước)
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        """
        Thêm vectors và metadata tương ứng vào FAISS index.
        Vectors sẽ được L2-normalize để dùng cosine similarity.
        """
        import faiss
        assert vectors.shape[0] == len(metadata_list), \
            f"Số lượng vectors ({vectors.shape[0]}) != metadata ({len(metadata_list)})"
        assert vectors.shape[1] == self._dimension, \
            f"Vector dimension ({vectors.shape[1]}) != expected ({self._dimension})"

        # L2 normalize để Inner Product = Cosine Similarity
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm top_k vectors gần nhất.
        Trả về list metadata kèm similarity score.
        """
        import faiss
        if self._index.ntotal == 0:
            return []

        # Reshape nếu cần
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        # Giới hạn top_k không vượt quá số lượng vectors hiện có
        actual_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS trả -1 nếu không đủ kết quả
                continue
            result = self._metadata[idx].copy()
            result["similarity_score"] = float(score)
            results.append(result)

        return results

    def save(self, directory: str) -> None:
        """Lưu FAISS index + metadata ra đĩa."""
        import faiss
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "metadata.json")

        faiss.write_index(self._index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        print(f"[VectorStore] Saved {self._index.ntotal} vectors to {directory}")

    def load(self, directory: str) -> None:
        """Load FAISS index + metadata từ đĩa."""
        import faiss
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")

        self._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        print(f"[VectorStore] Loaded {self._index.ntotal} vectors from {directory}")

    @property
    def size(self) -> int:
        return self._index.ntotal


class VectorStoreConnector:
    """
    Facade cho Vector Store.
    Mặc định dùng FAISS, có thể inject store khác qua constructor.
    """

    def __init__(self, dimension: int, store: BaseVectorStore = None):
        if store is not None:
            self._store = store
        else:
            self._store = FAISSVectorStore(dimension)

    def add_documents(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        self._store.add(vectors, metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._store.search(query_vector, top_k)

    def save(self, directory: str) -> None:
        self._store.save(directory)

    def load(self, directory: str) -> None:
        self._store.load(directory)

    @property
    def size(self) -> int:
        return self._store.size
