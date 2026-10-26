"""Web application:  python server.py   ->  http://localhost:8000

FastAPI backend serving the citizen-facing frontend in web/ and a JSON API:
  POST /api/ask         {question, service?: "auto"|"aadhaar"|"oci", config?: 1-5}
  GET  /api/documents   corpus library with supersession links
  GET  /api/about       pipeline back-ends and latest evaluation results
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import llm
from src.pipeline import CONFIGS, RAGPipeline

ROOT = Path(__file__).parent
CORPUS = os.getenv("CORPUS_DIR") or ("data/corpus" if (ROOT / "data/corpus").is_dir()
                                     and any((ROOT / "data/corpus").glob("*.md")) else "data/sample_corpus")

app = FastAPI(title="Cited & Current", version="1.0")
pipe = RAGPipeline(ROOT / CORPUS)
try:
    pipe.lam = json.loads((ROOT / "outputs/run_meta.json").read_text())["lambda"]
except Exception:
    pass

# ---- document-level metadata for the library and for "set aside" explanations
DOCS: dict[str, dict] = {}
for c in pipe.chunks:
    d = DOCS.setdefault(c.doc_id, {
        "doc_id": c.doc_id, "title": c.title, "source": c.source, "service": c.service,
        "doc_type": c.doc_type, "circular_no": c.circular_no, "issue_date": c.issue_date,
        "url": c.url, "supersedes": list(c.supersedes), "superseded_by": [], "sections": [],
        "is_sample": c.is_sample})
    if c.section not in d["sections"]:
        d["sections"].append(c.section)
for d in DOCS.values():
    for old in d["supersedes"]:
        if old in DOCS:
            DOCS[old]["superseded_by"].append(d["doc_id"])
STALE = pipe.reranker.all_superseded


def _latest_replacement(doc_id: str) -> dict | None:
    """Follow the superseded_by chain to the document currently in force."""
    seen, cur = set(), doc_id
    while DOCS.get(cur, {}).get("superseded_by") and cur not in seen:
        seen.add(cur)
        cur = max(DOCS[cur]["superseded_by"], key=lambda x: DOCS[x]["issue_date"])
    return DOCS.get(cur) if cur != doc_id else None


def _chunk_json(c) -> dict:
    return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "section": c.section,
            "chunk_type": c.chunk_type, "text": c.text, "issue_date": c.issue_date,
            "circular_no": c.circular_no, "url": c.url, "source": c.source, "doc_type": c.doc_type}


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    service: str = "auto"
    config: int = Field(5, ge=1, le=5)


@app.post("/api/ask")
def ask(body: AskIn):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")
    t0 = time.time()
    res = pipe.answer(q, CONFIGS[body.config - 1], service=None if body.service == "auto" else body.service)
    by_id = {c.chunk_id: c for c in res["retrieved"]}
    refs = []
    for r in res["references"]:
        c = by_id[r["chunk_id"]]
        refs.append({**_chunk_json(c), "n": r["n"], "status": "superseded" if c.doc_id in STALE else "current",
                     "replaced_by": (_latest_replacement(c.doc_id) or {}).get("title")})
    set_aside, seen = [], set()
    for c in res["removed_superseded"]:
        if c.doc_id in seen:
            continue
        seen.add(c.doc_id)
        rep = _latest_replacement(c.doc_id)
        if not rep or rep["doc_id"] not in {r["doc_id"] for r in refs}:
            continue  # only show older rules that the cited source actually replaced
        set_aside.append({"doc_id": c.doc_id, "title": c.title, "issue_date": c.issue_date,
                          "circular_no": c.circular_no,
                          "replaced_by": rep and {"title": rep["title"], "issue_date": rep["issue_date"],
                                                  "circular_no": rep["circular_no"]}})
    return {"question": q, "answer": res["answer"], "abstained": res["abstained"],
            "service": res["service"], "references": refs, "set_aside": set_aside,
            "passages": [_chunk_json(c) for c in res["retrieved"]],
            "mode": res["mode"], "config": CONFIGS[body.config - 1].name,
            "elapsed_ms": int((time.time() - t0) * 1000)}


@app.get("/api/documents")
def documents():
    docs = sorted(DOCS.values(), key=lambda d: (d["service"], d["issue_date"]), reverse=False)
    return {"corpus": CORPUS, "documents": [{**d, "status": "superseded" if d["doc_id"] in STALE else "current"}
                                            for d in docs]}


@app.get("/api/about")
def about():
    results = []
    p = ROOT / "outputs/results.csv"
    if p.exists():
        with open(p) as f:
            results = list(csv.DictReader(f))
    meta = {}
    if (ROOT / "outputs/run_meta.json").exists():
        meta = json.loads((ROOT / "outputs/run_meta.json").read_text())
    return {"corpus": CORPUS, "is_sample": "sample" in CORPUS, "n_documents": len(DOCS),
            "n_chunks": len(pipe.chunks), "backends": pipe.backends, "lambda": pipe.lam,
            "generator": f"{llm.provider()} / {llm.model_name()}" if llm.provider() else "extractive (no LLM key)",
            "configs": [c.name for c in CONFIGS], "results": results, "run_meta": meta}


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


if __name__ == "__main__":
    print(f"Corpus: {CORPUS} | backends: {pipe.backends}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
