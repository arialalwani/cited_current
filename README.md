# Cited & Current: A Recency-Aware RAG Assistant for Indian Government Document Processes

NMIMS MPSTME AI mini-project · Aria Lalwani (C066), Mahi Shah (C086), Vidisha Ajmera (C084)

Answers Aadhaar and OCI procedure questions only from official documents, cites the exact
document and circular for every statement, and prefers newer circulars over superseded ones.

## Quick start (10 minutes)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.evaluate               # 5-configuration evaluation -> outputs/
python server.py                     # web app -> http://localhost:8000
python ask.py "Till when is the online document update in Aadhaar free?"   # CLI
```

The first run downloads the BGE embedding model (~130 MB) and the BGE cross-encoder (~1.1 GB).
On a slow laptop set `RERANK_MODEL=BAAI/bge-reranker-v2-m3` or keep the default; both work on CPU.

**LLM answers (optional):** set one key, e.g. `export GEMINI_API_KEY=...` (Google AI Studio has a
free tier). Without a key, an extractive generator answers by quoting cited sentences and table rows.

## Web application

`python server.py` starts the citizen-facing app (FastAPI backend + `web/index.html`):

- **Ask a question**: plain-language answer, every sentence linked to its source; source cards show
  issuer, circular number, issue date and an "In force / Replaced" status; an "Older rules set aside"
  panel shows which superseded circulars were dropped and what replaced them; "not found" when the
  documents do not cover the question.
- **Document library**: every document with its date and supersession chain, filterable and searchable.
- **How it works**: the pipeline and the latest evaluation table (read from `outputs/`).
- Text-size control, keyboard navigation, mobile layout, and a clear "independent tool, not a
  Government of India website" notice.

API: `POST /api/ask {"question", "service": "auto|aadhaar|oci"}`, `GET /api/documents`, `GET /api/about`.
`app.py` (Streamlit) is kept as a developer console for comparing configurations.

## Pipeline (maps to report Section 3)

| Report section | Stage | File |
|---|---|---|
| 3.1 | Corpus collection: scraper with metadata (URL, type, circular no., issue date, supersession) | `src/scraper.py`, `data/sources.json` |
| 3.2 | Structure-aware chunking: tables whole (header kept), procedures heading-to-heading, prose by sentence with overlap | `src/ingest.py`, `src/chunking.py` |
| 3.3 | Hybrid retrieval: BM25 + BGE dense, fused with Reciprocal Rank Fusion (k = 60) | `src/retrieval.py` |
| 3.4 | Cross-encoder rerank + recency rule (supersession filter, date boost `s_ce + λ·recency`) | `src/reranking.py` |
| 3.5 | Query rewriting: Aadhaar/OCI detection, optional HyDE | `src/query_rewrite.py` |
| 3.6 | Cited generation and citation post-processing | `src/generation.py`, `src/llm.py` |
| 3.7 / 4 | Recall@1, Recall@5, MRR, citation accuracy, stale-source rate; λ tuned on dev split | `src/evaluate.py` |

Five configurations are defined in `src/pipeline.py` (`CONFIGS`): BM25, Dense, Hybrid, Hybrid + CE, full system.

## Using the real corpus

The repository ships a **sample corpus** (`data/sample_corpus/`, 22 documents) so everything runs
immediately. These are paraphrased test documents built to exercise the pipeline, including
superseded notices (free document-update deadline extensions, Aadhaar fee revision, OCI re-issue
relaxation). Their IDs start with `SAMPLE-` and they are **not official text**. Do not report results
on them as results on official documents.

To build the real corpus:

```bash
python -m src.scraper        # fetches seeds in data/sources.json -> data/raw/ + manifest.json
# check data/raw/manifest.json; fix wrong/missing dates or supersession in data/manual_overrides.json
python -m src.ingest         # -> data/corpus/*.md (normalised, with front matter)
python -m src.evaluate --corpus data/corpus --questions data/eval/questions_real.json
```

Government sites change layout often; if a seed fails, download the PDF by hand into `data/raw/`
and add an entry to `manifest.json` (same fields as the scraper writes).

## Labelling evaluation questions

`data/eval/questions.json` holds 54 labelled questions for the sample corpus. Each question lists its
gold `[doc_id, section]` pairs (the section is the `##` heading the chunk sits under), so labels
survive changes to chunk size. Extra flags:

- `stale_topic: true` — a newer circular superseded an older one on this topic
- `old_single_source: true` — only an old document answers it (checks the recency rule does not penalise it)
- `type` — `prose | table | procedure | superseded | ambiguous | unanswerable` (used for failure analysis)

Every third question is used to tune λ; the rest are held out for the results table.

## Outputs (`outputs/`)

`results.md` / `results.csv` (report Table 4.2), `results_chart.png`, `per_question.csv`,
`failures.csv` (grouped by the causes in Section 4.3), `lambda_tuning.csv`, `run_meta.json`
(records which embedding, cross-encoder and generator produced the numbers).

## Offline fallbacks

If the BGE models cannot be downloaded, the code falls back to TF-IDF + LSA embeddings and a
lexical cross-scorer so it never crashes. `run_meta.json` and the results header state which
back-end was used. Report numbers from the BGE run.
