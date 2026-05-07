"""
VectorStoreConnector: Kết nối FAISS Vector Database.
Component-based — có thể swap sang ChromaDB, Pinecone, etc.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """Abstract base class cho Vector Store."""

    @abstractmethod
    def add(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
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
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: List[Dict[str, Any]] = []
        self._bm25 = None
        self._tokenized_corpus = []

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

        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._metadata.extend(metadata_list)
        
        for item in metadata_list:
            text = item.get("document_text", "")
            if not text:
                text = f"{item.get('title', '')} {item.get('actor', '')} {item.get('category_name', '')}"
            self._tokenized_corpus.append(text.lower().split())

    def search(self, query: str, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm top_k vectors gần nhất.
        Trả về list metadata kèm similarity score.
        """
        if self._index.ntotal == 0:
            return []
            
        dense_results = self._search_dense(query_vector, top_k=top_k * 2)
        sparse_results = self._search_sparse(query, top_k=top_k * 2)
        
        return self._rrf(dense_results, sparse_results, top_k)
        
    def _search_dense(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        import faiss
        # Reshape nếu cần
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = query_vector.astype(np.float32)
        faiss.normalize_L2(query_vector)

        actual_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = self._metadata[idx].copy()
            result["similarity_score"] = float(score)
            results.append(result)

        return results

    def _search_sparse(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._bm25:
            return []
            
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n:
            if scores[idx] > 0:
                result = self._metadata[idx].copy()
                result["similarity_score"] = float(scores[idx])
                results.append(result)
                
        return results

    def _rrf(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], top_k: int = 5, k: int = 60) -> List[Dict[str, Any]]:
        scores = {}
        items = {}
        
        for rank, item in enumerate(dense_results):
            id_ = item.get("id") or str(item)
            scores[id_] = scores.get(id_, 0) + 1 / (k + rank + 1)
            items[id_] = item
            
        for rank, item in enumerate(sparse_results):
            id_ = item.get("id") or str(item)
            scores[id_] = scores.get(id_, 0) + 1 / (k + rank + 1)
            items[id_] = item
            
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for id_, score in sorted_scores:
            item = items[id_].copy()
            item["rrf_score"] = score
            final_results.append(item)
            
        return final_results

    def save(self, directory: str) -> None:
        """Lưu FAISS index + metadata + bm25 ra đĩa."""
        from rank_bm25 import BM25Okapi
        import faiss
        
        if self._tokenized_corpus and not self._bm25:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "metadata.json")
        bm25_path = os.path.join(directory, "bm25.pkl")

        faiss.write_index(self._index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
            
        if self._bm25:
            with open(bm25_path, "wb") as f:
                pickle.dump(self._bm25, f)
                
        print(f"[VectorStore] Saved {self._index.ntotal} vectors to {directory}")

    def load(self, directory: str) -> None:
        """Load FAISS index + metadata + bm25 từ đĩa."""
        import faiss
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "metadata.json")
        bm25_path = os.path.join(directory, "bm25.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")

        self._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
            
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)

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

    def search(self, query: str, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._store.search(query, query_vector, top_k)

    def save(self, directory: str) -> None:
        self._store.save(directory)

    def load(self, directory: str) -> None:
        self._store.load(directory)

    @property
    def size(self) -> int:
        return self._store.size
