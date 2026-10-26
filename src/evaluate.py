"""Evaluation (Sections 3.7 and 4): Recall@1, Recall@5, MRR, citation accuracy
and stale-source rate for the five configurations, with lambda tuned on a
development split and results reported on the held-out split.

Usage:
    python -m src.evaluate                       # sample corpus, auto LLM if key set
    python -m src.evaluate --corpus data/corpus --questions data/eval/my_questions.json
    python -m src.evaluate --no-llm              # force extractive generator
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import llm
from .generation import generate
from .pipeline import CONFIGS, CONTEXT_K, FULL, RAGPipeline

LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
FAILURE_CAUSE = {
    "table": "Table content split or misread",
    "procedure": "Procedure cut across chunks",
    "ambiguous": "Ambiguous question (Aadhaar vs OCI)",
    "unanswerable": "Answer missing from collected documents",
    "superseded": "Wrong or missing date / superseded source",
    "prose": "Retrieval or ranking miss",
}


def split(questions: list[dict], dev_every: int = 3) -> tuple[list[dict], list[dict]]:
    """Deterministic split: every 3rd question tunes lambda, the rest is held out."""
    dev = [q for i, q in enumerate(questions) if i % dev_every == 0]
    test = [q for i, q in enumerate(questions) if i % dev_every != 0]
    return dev, test


def rank_metrics(pipe: RAGPipeline, q: dict, ranked: list[int]) -> dict:
    gold = {tuple(g) for g in q["gold"]}
    rel = [(pipe.chunks[i].doc_id, pipe.chunks[i].section) in gold for i in ranked]
    first = next((r for r, ok in enumerate(rel, start=1) if ok), None)
    return {"r1": float(any(rel[:1])), "r5": float(any(rel[:5])),
            "rr": 1.0 / first if first else 0.0, "first_rank": first}


def tune_lambda(pipe: RAGPipeline, dev: list[dict]) -> tuple[float, list[dict]]:
    rows = []
    answerable = [q for q in dev if q["gold"]]
    for lam in LAMBDA_GRID:
        ms = [rank_metrics(pipe, q, pipe.retrieve(q["question"], FULL, lam=lam)[0]) for q in answerable]
        rows.append({"lambda": lam, "dev_mrr": np.mean([m["rr"] for m in ms]),
                     "dev_recall@1": np.mean([m["r1"] for m in ms])})
    best = max(rows, key=lambda r: (round(r["dev_mrr"], 6), r["dev_recall@1"], -r["lambda"]))
    return best["lambda"], rows


def run(corpus: str, questions_path: str, out_dir: str, use_llm: bool | None) -> None:
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    questions = json.loads(Path(questions_path).read_text())
    pipe = RAGPipeline(corpus)
    use_llm = (llm.provider() is not None) if use_llm is None else use_llm
    gen_mode = f"LLM {llm.provider()}/{llm.model_name()}" if use_llm else "extractive (no LLM)"
    stale_docs = pipe.reranker.all_superseded

    dev, test = split(questions)
    best_lam, tuning = tune_lambda(pipe, dev)
    pipe.lam = best_lam
    print(f"backends: {pipe.backends} | generator: {gen_mode}")
    print(f"lambda tuned on {len(dev)} dev questions -> {best_lam}")

    results, per_q = [], []
    for cfg in CONFIGS:
        ms, cites_ok, cites_all, stale_ans, ans_n = [], 0, 0, 0, 0
        stale_topic_ans, stale_topic_hits = 0, 0
        abst_unans, n_unans, false_abst = 0, 0, 0
        for q in test:
            ranked, _ = pipe.retrieve(q["question"], cfg)
            top = [pipe.chunks[i] for i in ranked[:CONTEXT_K]]
            g = generate(q["question"], top, use_llm=use_llm)
            gold = {tuple(x) for x in q["gold"]}
            by_id = {c.chunk_id: c for c in top}
            cited = list(dict.fromkeys(g["citations"]))
            if not q["gold"]:
                n_unans += 1
                abst_unans += int(g["abstained"])
                per_q.append({"config": cfg.name, "id": q["id"], "type": q["type"], "question": q["question"],
                              "recall@1": "", "recall@5": "", "rr": "", "abstained": g["abstained"],
                              "citations": " ".join(cited), "stale_cited": ""})
                continue
            m = rank_metrics(pipe, q, ranked)
            ms.append(m)
            if g["abstained"]:
                false_abst += 1
            ok = sum((by_id[c].doc_id, by_id[c].section) in gold for c in cited)
            cites_ok += ok; cites_all += len(cited)
            is_stale = any(c.split("#")[0] in stale_docs for c in cited)
            if cited:
                ans_n += 1; stale_ans += int(is_stale)
                if q.get("stale_topic"):
                    stale_topic_ans += 1; stale_topic_hits += int(is_stale)
            per_q.append({"config": cfg.name, "id": q["id"], "type": q["type"], "question": q["question"],
                          "recall@1": m["r1"], "recall@5": m["r5"], "rr": round(m["rr"], 3),
                          "abstained": g["abstained"], "citations": " ".join(cited), "stale_cited": is_stale})
        stale_q = [m for m, q in zip(ms, [q for q in test if q["gold"]]) if q.get("stale_topic")]
        single = [m for m, q in zip(ms, [q for q in test if q["gold"]]) if q.get("old_single_source")]
        results.append({
            "Configuration": cfg.name,
            "Recall@1": np.mean([m["r1"] for m in ms]),
            "Recall@5": np.mean([m["r5"] for m in ms]),
            "MRR": np.mean([m["rr"] for m in ms]),
            "Citation accuracy": cites_ok / cites_all if cites_all else 0.0,
            "Stale-source rate": stale_ans / ans_n if ans_n else 0.0,
            "Stale-source rate (superseded topics)": stale_topic_hits / stale_topic_ans if stale_topic_ans else 0.0,
            "MRR (superseded topics)": np.mean([m["rr"] for m in stale_q]) if stale_q else float("nan"),
            "Recall@5 (old single-source docs)": np.mean([m["r5"] for m in single]) if single else float("nan"),
            "Abstention on unanswerable": abst_unans / n_unans if n_unans else float("nan"),
            "False abstention": false_abst / len(ms) if ms else 0.0,
        })

    # failure analysis on the full system
    fails = []
    for row in per_q:
        if row["config"] != FULL.name:
            continue
        failed = (row["type"] == "unanswerable" and not row["abstained"]) or \
                 (row["type"] != "unanswerable" and (row["recall@1"] == 0.0 or row["stale_cited"] is True))
        if failed:
            fails.append({**row, "cause": FAILURE_CAUSE.get(row["type"], "Other")})

    meta = {"corpus": corpus, "questions": questions_path, "n_dev": len(dev), "n_test": len(test),
            "lambda": best_lam, "generator": gen_mode, **pipe.backends,
            "runtime_s": round(time.time() - t0, 1)}
    _write(out, results, per_q, fails, tuning, meta)


def _write(out: Path, results, per_q, fails, tuning, meta) -> None:
    def to_csv(path, rows):
        if not rows:
            path.write_text(""); return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    to_csv(out / "results.csv", results)
    to_csv(out / "per_question.csv", per_q)
    to_csv(out / "failures.csv", fails)
    to_csv(out / "lambda_tuning.csv", tuning)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    main_cols = ["Recall@1", "Recall@5", "MRR", "Citation accuracy", "Stale-source rate"]
    lines = ["# Evaluation results", "",
             f"Corpus: `{meta['corpus']}` | held-out questions: {meta['n_test']} | dev (lambda tuning): {meta['n_dev']}",
             f"Embedding: {meta['embedding']} | Cross-encoder: {meta['cross_encoder']} | "
             f"Generator: {meta['generator']} | lambda = {meta['lambda']}", "",
             "| Configuration | " + " | ".join(main_cols) + " |",
             "|---|" + "---|" * len(main_cols)]
    for r in results:
        lines.append(f"| {r['Configuration']} | " + " | ".join(f"{r[c]:.3f}" for c in main_cols) + " |")
    extra = ["Stale-source rate (superseded topics)", "MRR (superseded topics)",
             "Recall@5 (old single-source docs)", "Abstention on unanswerable", "False abstention"]
    lines += ["", "## Diagnostics", "", "| Configuration | " + " | ".join(extra) + " |", "|---|" + "---|" * len(extra)]
    for r in results:
        lines.append(f"| {r['Configuration']} | " + " | ".join(f"{r[c]:.3f}" for c in extra) + " |")
    lines += ["", "## Failure analysis (full system)", ""]
    if fails:
        from collections import Counter
        for cause, n in Counter(f["cause"] for f in fails).most_common():
            lines.append(f"- {cause}: {n}")
    else:
        lines.append("- No failures on the held-out set.")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.8))
        w = 0.15
        x = np.arange(len(results))
        colors = ["#2a6f97", "#61a5c2", "#014f86", "#e9c46a", "#c1121f"]
        for k, col in enumerate(main_cols):
            ax.bar(x + (k - 2) * w, [r[col] for r in results], w, label=col, color=colors[k])
        ax.set_xticks(x, [r["Configuration"].split(".")[0] for r in results])
        ax.set_xlabel("Configuration (1 BM25, 2 Dense, 3 Hybrid, 4 +Cross-encoder, 5 +Recency)")
        ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
        ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(out / "results_chart.png", dpi=200)
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/sample_corpus")
    ap.add_argument("--questions", default="data/eval/questions.json")
    ap.add_argument("--out", default="outputs")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--llm", dest="use_llm", action="store_true", default=None)
    g.add_argument("--no-llm", dest="use_llm", action="store_false")
    a = ap.parse_args()
    run(a.corpus, a.questions, a.out, a.use_llm)
