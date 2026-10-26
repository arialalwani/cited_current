"""Model back-ends with graceful fallbacks.

Preferred (used automatically when the models can be downloaded):
  * embeddings   : BAAI/bge-small-en-v1.5   (override with EMBED_MODEL)
  * cross-encoder: BAAI/bge-reranker-base    (override with RERANK_MODEL)

Fallbacks (no internet / no GPU): a TF-IDF + LSA embedding and a lexical
cross-scorer, so the pipeline and the evaluation always run. The backend in
use is printed and written into every results file, so reported numbers can
always be traced to the models that produced them.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
FORCE_FALLBACK = os.getenv("CC_OFFLINE", "0") == "1"

STOPWORDS = set("""a an the and or of to in on for with by at from is are was were be been being
do does did can could should would will shall may might must i my me we our you your it its this that
these those what which who whom am im how when where why if than then so as not no yes any all also into
about there their they them he she his her get got have has had need needs want""".split())


try:
    from nltk.stem import PorterStemmer
    _porter = PorterStemmer()
except ImportError:  # nltk optional
    _porter = None


def tokenize(text: str) -> list[str]:
    text = re.sub(r"\bre-(?=[a-z])", "re", text.lower())  # re-issue -> reissue
    toks = re.findall(r"[a-z0-9]+", text)
    return [_stem(t) for t in toks if t not in STOPWORDS]


def _stem(t: str) -> str:
    if _porter is not None:
        return _porter.stem(t)
    for suf in ("ing", "ed", "es", "s"):  # light fallback
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


# ------------------------------------------------------------------ embedder
class Embedder:
    def __init__(self, corpus_texts: list[str]):
        self.backend = "tfidf-lsa (fallback)"
        self._st = None
        if not FORCE_FALLBACK:
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(EMBED_MODEL)
                self.backend = EMBED_MODEL
            except Exception as e:  # offline, blocked, or not installed
                print(f"[models] embedding model unavailable ({type(e).__name__}); using TF-IDF+LSA fallback")
        if self._st is None:
            word = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2),
                                   sublinear_tf=True, token_pattern=None)
            char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            n_comp = max(2, min(96, len(corpus_texts) - 1))
            self._pipe = make_pipeline(make_union(word, char), TruncatedSVD(n_comp, random_state=0),
                                       Normalizer(copy=False))
            self._pipe.fit(corpus_texts)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if self._st is not None:
            if is_query:
                texts = [BGE_QUERY_PREFIX + t for t in texts]
            return np.asarray(self._st.encode(texts, normalize_embeddings=True, batch_size=32))
        return self._pipe.transform(texts)


# ------------------------------------------------------------- cross-encoder
class CrossScorer:
    """Scores (question, passage) pairs jointly. Returns higher = more relevant."""

    def __init__(self, corpus_texts: list[str]):
        self.backend = "lexical cross-scorer (fallback)"
        self._ce = None
        if not FORCE_FALLBACK:
            try:
                from sentence_transformers import CrossEncoder
                self._ce = CrossEncoder(RERANK_MODEL, max_length=512)
                self.backend = RERANK_MODEL
            except Exception as e:
                print(f"[models] cross-encoder unavailable ({type(e).__name__}); using lexical fallback")
        if self._ce is None:
            n = len(corpus_texts)
            df = Counter(t for txt in corpus_texts for t in set(tokenize(txt)))
            self._idf = {t: math.log(1 + n / c) for t, c in df.items()}
            self._default_idf = math.log(1 + n)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.array([])
        if self._ce is not None:
            return np.asarray(self._ce.predict([(query, p) for p in passages]), dtype=float)
        q = tokenize(query)
        q_bi = set(zip(q, q[1:]))
        # numbers in a question are usually the user's own facts (age, year), not topic words
        w = lambda t: self._idf.get(t, self._default_idf) * (0.3 if t.isdigit() else 1.0)
        total = sum(w(t) for t in set(q)) or 1.0
        out = []
        for p in passages:
            pt = tokenize(p)
            ps, p_bi = set(pt), set(zip(pt, pt[1:]))
            cover = sum(w(t) for t in set(q) if t in ps) / total
            bigram = len(q_bi & p_bi) / max(1, len(q_bi))
            brevity = 1.0 / (1.0 + math.log1p(len(pt)) / 10)  # mild preference for focused chunks
            head = set(tokenize(p.split(". ")[1] if p.count(". ") >= 2 else ""))
            head_cover = sum(w(t) for t in set(q) if t in head) / total
            out.append(cover + 0.5 * bigram + 0.6 * head_cover + 0.1 * brevity)
        return np.array(out)
