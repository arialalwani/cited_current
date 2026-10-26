# Evaluation results

Corpus: `data/sample_corpus` | held-out questions: 36 | dev (lambda tuning): 18
Embedding: tfidf-lsa (fallback) | Cross-encoder: lexical cross-scorer (fallback) | Generator: extractive (no LLM) | lambda = 0.0

| Configuration | Recall@1 | Recall@5 | MRR | Citation accuracy | Stale-source rate |
|---|---|---|---|---|---|
| 1. BM25 only | 0.765 | 0.971 | 0.849 | 0.646 | 0.147 |
| 2. Dense retrieval only | 0.794 | 0.971 | 0.861 | 0.625 | 0.176 |
| 3. Hybrid (BM25 + dense, RRF) | 0.765 | 0.971 | 0.846 | 0.625 | 0.206 |
| 4. Hybrid + cross-encoder | 0.765 | 0.971 | 0.851 | 0.617 | 0.176 |
| 5. Hybrid + cross-encoder + recency (full) | 0.853 | 0.941 | 0.896 | 0.705 | 0.000 |

## Diagnostics

| Configuration | Stale-source rate (superseded topics) | MRR (superseded topics) | Recall@5 (old single-source docs) | Abstention on unanswerable | False abstention |
|---|---|---|---|---|---|
| 1. BM25 only | 0.667 | 0.352 | 1.000 | 0.500 | 0.000 |
| 2. Dense retrieval only | 0.833 | 0.407 | 1.000 | 0.500 | 0.000 |
| 3. Hybrid (BM25 + dense, RRF) | 0.833 | 0.324 | 1.000 | 0.500 | 0.000 |
| 4. Hybrid + cross-encoder | 0.833 | 0.435 | 1.000 | 0.500 | 0.000 |
| 5. Hybrid + cross-encoder + recency (full) | 0.000 | 0.854 | 1.000 | 0.500 | 0.000 |

## Failure analysis (full system)

- Table content split or misread: 2
- Retrieval or ranking miss: 1
- Wrong or missing date / superseded source: 1
- Ambiguous question (Aadhaar vs OCI): 1
- Answer missing from collected documents: 1
