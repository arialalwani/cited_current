"""Cross-encoder reranking with the recency rule (Section 3.4).

1. Supersession filter: if a candidate comes from a document that (directly or
   through a chain) supersedes another candidate's document, the older chunk
   is removed.
2. Date boost: score(d) = s_ce(d) + lambda * recency(d), where s_ce is
   min-max normalised over the candidates and recency(d) is the issue date
   scaled to [0, 1] within the group of candidates that overlap d in topic
   (text similarity >= OVERLAP_THRESHOLD, different documents).
   A chunk with no overlapping candidate gets recency = 1, so an old document
   that is still the only source is never penalised for its age.
"""
from __future__ import annotations

from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk
from .models import CrossScorer, tokenize

OVERLAP_THRESHOLD = 0.35
DEFAULT_LAMBDA = 0.5


def _ord(d: str) -> int | None:
    try:
        return date.fromisoformat(d).toordinal()
    except (ValueError, TypeError):
        return None


def superseded_closure(chunks: list[Chunk]) -> dict[str, set[str]]:
    """doc_id -> every doc_id it supersedes, directly or transitively."""
    direct: dict[str, set[str]] = {}
    for c in chunks:
        direct.setdefault(c.doc_id, set()).update(c.supersedes)
    closure: dict[str, set[str]] = {}

    def walk(d: str, seen: set[str]) -> set[str]:
        out = set()
        for s in direct.get(d, ()):
            if s not in seen:
                out |= {s} | walk(s, seen | {s})
        return out

    for d in direct:
        closure[d] = walk(d, {d})
    return closure


class Reranker:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.scorer = CrossScorer([c.index_text for c in chunks])
        self.closure = superseded_closure(chunks)
        self.all_superseded = set().union(*self.closure.values()) if self.closure else set()
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                      sublinear_tf=True).fit([c.text for c in chunks])
        self._vecs = self._tfidf.transform([c.text for c in chunks])

    # ------------------------------------------------------------------ steps
    def cross_scores(self, query: str, cand: list[int]) -> dict[int, float]:
        raw = self.scorer.score(query, [self.chunks[i].index_text for i in cand])
        if len(raw) == 0:
            return {}
        lo, hi = float(raw.min()), float(raw.max())
        norm = (raw - lo) / (hi - lo) if hi > lo else np.ones_like(raw)
        return {i: float(s) for i, s in zip(cand, norm)}

    def supersession_filter(self, cand: list[int]) -> tuple[list[int], list[int]]:
        docs = {self.chunks[i].doc_id for i in cand}
        dropped_docs = set()
        for d in docs:
            dropped_docs |= self.closure.get(d, set()) & docs
        kept = [i for i in cand if self.chunks[i].doc_id not in dropped_docs]
        removed = [i for i in cand if self.chunks[i].doc_id in dropped_docs]
        return kept, removed

    def recency(self, cand: list[int]) -> dict[int, float]:
        if not cand:
            return {}
        sims = cosine_similarity(self._vecs[cand])
        out = {}
        for a, i in enumerate(cand):
            group = [cand[b] for b in range(len(cand))
                     if b != a and sims[a, b] >= OVERLAP_THRESHOLD
                     and self.chunks[cand[b]].doc_id != self.chunks[i].doc_id]
            dates = [_ord(self.chunks[j].issue_date) for j in group + [i]]
            dates = [d for d in dates if d is not None]
            mine = _ord(self.chunks[i].issue_date)
            if not group or mine is None or len(dates) < 2 or max(dates) == min(dates):
                out[i] = 1.0  # no rival on the same topic (or no usable date): no age penalty
            else:
                out[i] = (mine - min(dates)) / (max(dates) - min(dates))
        return out

    # -------------------------------------------------------------- combined
    def rerank(self, query: str, cand: list[int], use_recency: bool,
               lam: float = DEFAULT_LAMBDA) -> tuple[list[int], dict]:
        info: dict = {"removed_superseded": []}
        if use_recency:
            cand, removed = self.supersession_filter(cand)
            info["removed_superseded"] = removed
        ce = self.cross_scores(query, cand)
        rec = self.recency(cand) if use_recency else {i: 0.0 for i in cand}
        final = {i: ce[i] + (lam * rec[i] if use_recency else 0.0) for i in cand}
        info.update({"ce": ce, "recency": rec, "final": final})
        return sorted(cand, key=lambda i: -final[i]), info
