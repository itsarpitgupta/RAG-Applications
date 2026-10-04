import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from typing import List, Union

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu", normalize: bool = True):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._dimension = 384  # Default for MiniLM-L6-v2 and BGE-small-en-v1.5

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is None:
            return self._dimension
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generates dense embeddings for a list of text strings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generates dense embedding for a single query string."""
        emb = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        return emb.astype(np.float32)
