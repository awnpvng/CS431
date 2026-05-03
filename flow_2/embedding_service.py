"""
EmbeddingService: Chuyển đổi text thành vector sử dụng sentence-transformers.
Component-based design — có thể dễ dàng thay thế bằng API embedding khác.
"""

import numpy as np
from typing import List, Union
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class cho Embedding — dễ swap sang OpenAI, Gemini, etc."""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) thành numpy array of vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Trả về chiều của vector embedding."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedding sử dụng sentence-transformers (local, offline-friendly).
    Model mặc định: 'all-MiniLM-L6-v2' — nhẹ, nhanh, 384 chiều.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"[EmbeddingService] Đang load model: {model_name}...")
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        print(f"[EmbeddingService] Model loaded! Dimension = {self._dimension}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingService:
    """
    Facade / Factory cho Embedding.
    Mặc định dùng SentenceTransformerEmbedder, có thể inject embedder khác.
    """

    def __init__(self, embedder: BaseEmbedder = None, model_name: str = "all-MiniLM-L6-v2"):
        if embedder is not None:
            self._embedder = embedder
        else:
            self._embedder = SentenceTransformerEmbedder(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed một câu query → vector 1D."""
        return self._embedder.encode(query)[0]

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed nhiều documents → matrix (N, dim)."""
        return self._embedder.encode(documents)

    @property
    def dimension(self) -> int:
        return self._embedder.dimension
