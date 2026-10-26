"""End-to-end pipeline and the five evaluation configurations (Section 3.7)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, load_corpus
from .generation import generate
from .query_rewrite import rewrite
from .reranking import DEFAULT_LAMBDA, Reranker
from .retrieval import BM25Retriever, DenseRetriever, reciprocal_rank_fusion

CANDIDATES = 20   # fused candidates passed to the reranker
CONTEXT_K = 5     # chunks given to the generator


@dataclass(frozen=True)
class Config:
    name: str
    bm25: bool
    dense: bool
    rerank: bool
    recency: bool


CONFIGS = [
    Config("1. BM25 only", True, False, False, False),
    Config("2. Dense retrieval only", False, True, False, False),
    Config("3. Hybrid (BM25 + dense, RRF)", True, True, False, False),
    Config("4. Hybrid + cross-encoder", True, True, True, False),
    Config("5. Hybrid + cross-encoder + recency (full)", True, True, True, True),
]
FULL = CONFIGS[-1]


class RAGPipeline:
    def __init__(self, corpus_dir: str | Path = "data/sample_corpus", chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_corpus(corpus_dir)
        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(self.chunks)
        self.reranker = Reranker(self.chunks)
        self.lam = DEFAULT_LAMBDA
        from . import generation
        from .models import tokenize
        generation.CORPUS_VOCAB = {t for c in self.chunks for t in tokenize(c.index_text)}

    @property
    def backends(self) -> dict:
        return {"embedding": self.dense.embedder.backend, "cross_encoder": self.reranker.scorer.backend}

    def retrieve(self, question: str, cfg: Config = FULL, lam: float | None = None,
                 use_service_filter: bool = True, use_hyde: bool = False,
                 service: str | None = None) -> tuple[list[int], dict]:
        rw = rewrite(question, use_hyde=use_hyde)
        if service in ("aadhaar", "oci"):
            rw["service"] = service  # user picked the service explicitly
        allowed = None
        if use_service_filter and rw["service"]:
            allowed = {i for i, c in enumerate(self.chunks) if c.service == rw["service"]}
        lists = []
        if cfg.bm25:
            lists.append(self.bm25.search(question, allowed, CANDIDATES))
        if cfg.dense:
            lists.append(self.dense.search(rw["dense_query"], allowed, CANDIDATES))
        ranked = lists[0] if len(lists) == 1 else reciprocal_rank_fusion(lists)
        ranked = ranked[:CANDIDATES]
        info = {"service": rw["service"], "removed_superseded": []}
        if cfg.rerank:
            ranked, rinfo = self.reranker.rerank(question, ranked, cfg.recency,
                                                 self.lam if lam is None else lam)
            info.update(rinfo)
        return ranked, info

    def answer(self, question: str, cfg: Config = FULL, use_llm: bool | None = None,
               use_hyde: bool = False, service: str | None = None) -> dict:
        ranked, info = self.retrieve(question, cfg, use_hyde=use_hyde, service=service)
        top = [self.chunks[i] for i in ranked[:CONTEXT_K]]
        out = generate(question, top, use_llm=use_llm)
        stale = self.reranker.all_superseded
        for ref in out["references"]:
            ref["superseded"] = ref["chunk_id"].split("#")[0] in stale
        out.update({"retrieved": top, "service": info["service"],
                    "removed_superseded": [self.chunks[i] for i in info.get("removed_superseded", [])]})
        return out
