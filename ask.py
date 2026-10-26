"""Command-line interface.

    python ask.py "Till when is the online document update in Aadhaar free?"
    python ask.py --config 4 "..."        # compare a pipeline configuration (1-5)
    python ask.py                         # interactive mode
"""
import argparse

from src.pipeline import CONFIGS, RAGPipeline


def show(pipe, q, cfg):
    r = pipe.answer(q, cfg)
    print(f"\nQ: {q}\n[{cfg.name} | service: {r['service'] or 'both'} | {r['mode']}]\n")
    print(r["answer"], "\n")
    for ref in r["references"]:
        print(f"  [{ref['n']}] {ref['citation']}{'  (SUPERSEDED)' if ref['superseded'] else ''}")
    if r["removed_superseded"]:
        print("  removed as superseded:", ", ".join(c.chunk_id for c in r["removed_superseded"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--corpus", default="data/sample_corpus")
    ap.add_argument("--config", type=int, default=5, choices=range(1, 6))
    a = ap.parse_args()
    pipe, cfg = RAGPipeline(a.corpus), CONFIGS[a.config - 1]
    if a.question:
        show(pipe, " ".join(a.question), cfg)
    else:
        while (q := input("\nAsk (blank to quit): ").strip()):
            show(pipe, q, cfg)
