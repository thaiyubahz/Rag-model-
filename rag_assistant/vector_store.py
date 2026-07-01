"""
vector_store.py
----------------
FAISS-backed vector store for the RAG research citation assistant.

Stores embeddings in a FAISS index (file-based, persisted to disk) and
keeps parallel metadata (chunk text, source PDF, page number, chunk id)
in a JSON sidecar file, plus the raw vectors in a .npy sidecar so the
index can be rebuilt on deletion without re-embedding everything.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (free, local, 384-dim).
No external API calls are made anywhere in this module.
"""

import os
import json
import threading
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    A minimal, persistent, file-based vector store using FAISS.

    Index type: IndexFlatIP (cosine similarity via normalized inner product).
    Research-paper corpora per session are small (hundreds to low thousands
    of chunks), so flat search is fast enough and needs no training step.
    """

    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, storage_dir: str = "./vector_data", index_name: str = "faiss_index"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.index_path = os.path.join(self.storage_dir, f"{index_name}.faiss")
        self.metadata_path = os.path.join(self.storage_dir, f"{index_name}_metadata.json")
        self.vectors_path = os.path.join(self.storage_dir, f"{index_name}_vectors.npy")

        self._lock = threading.Lock()

        # Loaded once, shared across all calls
        self.embedder = SentenceTransformer(self.EMBEDDING_MODEL_NAME)

        self.metadata: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = np.zeros((0, self.EMBEDDING_DIM), dtype="float32")

        self._load_or_init()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_or_init(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            if os.path.exists(self.vectors_path):
                self.vectors = np.load(self.vectors_path)
            else:
                # Rebuild vectors array from index if the .npy is missing
                self.vectors = self.index.reconstruct_n(0, self.index.ntotal) if self.index.ntotal else \
                    np.zeros((0, self.EMBEDDING_DIM), dtype="float32")
        else:
            self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self.metadata = []
            self.vectors = np.zeros((0, self.EMBEDDING_DIM), dtype="float32")
            self._persist()

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        np.save(self.vectors_path, self.vectors)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # required for cosine sim via inner product
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add a batch of text chunks to the store.

        Each chunk dict must contain:
            - "text": str      the chunk content
            - "source": str    paper title (used for grouping / deletion)
            - "page": int      page number, for citation display
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        new_vectors = self._embed(texts)

        with self._lock:
            start_id = len(self.metadata)
            self.index.add(new_vectors)
            self.vectors = np.vstack([self.vectors, new_vectors]) if self.vectors.size else new_vectors

            for i, chunk in enumerate(chunks):
                self.metadata.append({
                    "id": start_id + i,
                    "text": chunk["text"],
                    "source": chunk.get("source", "unknown"),
                    "page": chunk.get("page", None),
                    "chunk_id": chunk.get("chunk_id", start_id + i),
                })

            self._persist()

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the store for the top_k most relevant chunks to `query`.
        Returns metadata dicts augmented with a "score" field (cosine similarity).
        """
        if self.index.ntotal == 0:
            return []

        query_vec = self._embed([query])
        with self._lock:
            scores, ids = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            results.append(meta)

        return results

    def list_sources(self) -> List[Dict[str, Any]]:
        """Return [{source, chunks}] aggregated across all papers currently indexed."""
        counts: Dict[str, int] = {}
        for m in self.metadata:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return [{"source": src, "chunks": n} for src, n in counts.items()]

    def remove_by_source(self, source: str) -> int:
        """
        Remove every chunk belonging to `source` (a paper title) and rebuild
        the index from the remaining vectors. Returns number of chunks removed.
        """
        with self._lock:
            keep_idx = [i for i, m in enumerate(self.metadata) if m["source"] != source]
            removed = len(self.metadata) - len(keep_idx)
            if removed == 0:
                return 0

            new_vectors = self.vectors[keep_idx] if keep_idx else np.zeros((0, self.EMBEDDING_DIM), dtype="float32")
            new_metadata = []
            for new_id, old_idx in enumerate(keep_idx):
                m = dict(self.metadata[old_idx])
                m["id"] = new_id
                new_metadata.append(m)

            new_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            if new_vectors.shape[0]:
                new_index.add(new_vectors)

            self.index = new_index
            self.vectors = new_vectors
            self.metadata = new_metadata
            self._persist()

        return removed

    def clear(self):
        """Wipe the index and metadata (fresh session / new upload set)."""
        with self._lock:
            self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self.metadata = []
            self.vectors = np.zeros((0, self.EMBEDDING_DIM), dtype="float32")
            self._persist()

    def count(self) -> int:
        return self.index.ntotal


# ----------------------------------------------------------------------
# Convenience singleton accessor for use in FastAPI routes
# ----------------------------------------------------------------------
_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
    return _store_instance
