"""Hybrid retrieval: BM25 + dense embeddings, merged with Reciprocal Rank Fusion (Section 3.3)."""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .models import Embedder, tokenize

RRF_K = 60  # smoothing constant from Cormack et al. (2009)


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])

    def search(self, query: str, allowed: set[int] | None = None, top_k: int = 20) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        return [int(i) for i in order if (allowed is None or i in allowed) and scores[i] > 0][:top_k]


class DenseRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.index_text for c in chunks]
        self.embedder = Embedder(texts)
        self.matrix = self.embedder.encode(texts)

    def search(self, query: str, allowed: set[int] | None = None, top_k: int = 20) -> list[int]:
        q = self.embedder.encode([query], is_query=True)[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)
        return [int(i) for i in order if allowed is None or i in allowed][:top_k]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """RRF(d) = sum_r 1 / (k + rank_r(d)), ranks starting at 1."""
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, idx in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: -scores[i])
