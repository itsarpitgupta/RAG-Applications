import json
import threading
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import faiss

class FAISSManager:
    def __init__(self, index_dir: Path, dimension: int = 384, metric: str = "inner_product"):
        self.index_dir = index_dir
        self.dimension = dimension
        self.metric = metric
        self.index_path = self.index_dir / "index.bin"
        self.meta_path = self.index_dir / "index_meta.json"
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> faiss.Index:
        with self.lock:
            if self.index_path.exists():
                try:
                    index = faiss.read_index(str(self.index_path))
                    print(f"[FAISSManager] Loaded existing FAISS index with {index.ntotal} vectors from {self.index_path}")
                    return index
                except Exception as e:
                    print(f"[FAISSManager] Error loading index from {self.index_path}: {e}. Creating new index.")
            
            # Create fresh index
            if self.metric == "inner_product":
                index = faiss.IndexFlatIP(self.dimension)
            else:
                index = faiss.IndexFlatL2(self.dimension)
            
            print(f"[FAISSManager] Created new FAISS index ({self.metric}, dim={self.dimension})")
            return index

    @property
    def total_vectors(self) -> int:
        with self.lock:
            return self.index.ntotal

    def add_vectors(self, vectors: np.ndarray) -> int:
        """
        Incrementally adds new vectors to the FAISS index without modifying existing vectors.
        Returns the starting FAISS ID for these vectors.
        """
        if len(vectors) == 0:
            return self.total_vectors

        with self.lock:
            start_id = self.index.ntotal
            # Ensure float32 and contiguous
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            self.index.add(vectors)
            self.save_index()
            print(f"[FAISSManager] Added {len(vectors)} vectors (Range: {start_id} to {self.index.ntotal - 1}). Total vectors: {self.index.ntotal}")
            return start_id

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> Tuple[List[int], List[float]]:
        """
        Performs similarity search.
        Returns: (list_of_faiss_ids, list_of_similarity_scores)
        """
        with self.lock:
            if self.index.ntotal == 0:
                return [], []

            actual_k = min(top_k, self.index.ntotal)
            query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
            
            scores, indices = self.index.search(query_vector, actual_k)
            
            flat_indices = indices[0].tolist()
            flat_scores = scores[0].tolist()
            
            # Filter out invalid indices (-1)
            valid_results = [(idx, score) for idx, score in zip(flat_indices, flat_scores) if idx != -1]
            if not valid_results:
                return [], []

            res_ids, res_scores = zip(*valid_results)
            return list(res_ids), list(res_scores)

    def save_index(self):
        with self.lock:
            # Atomic save via temp file
            temp_path = self.index_dir / "index.bin.tmp"
            faiss.write_index(self.index, str(temp_path))
            temp_path.replace(self.index_path)

            meta = {
                "ntotal": self.index.ntotal,
                "dimension": self.dimension,
                "metric": self.metric
            }
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
