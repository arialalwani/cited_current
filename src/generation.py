"""Answer generation with citations (Section 3.6).

The model receives the top chunks with their chunk IDs, must answer only from
them, tag every statement with [chunk_id], and abstain when the chunks do not
contain the answer. Post-processing swaps chunk IDs for numbered references
that show title, circular number, issue date and URL.
"""
from __future__ import annotations

import re

from . import llm
from .chunking import Chunk
from .models import tokenize

CORPUS_VOCAB: set[str] = set()  # filled by RAGPipeline: every token in the collection
# everyday question words that are not topic words, so their absence from the corpus means nothing
GENERIC = set(tokenize("""many much cost costs price last latest like back remember taken take happen happens
lose lost long often know tell please help want give make made still able allowed possible exactly
right wrong correct current currently today now again way someone anyone person people thing things"""))

ABSTAIN = "I could not find this in the collected official documents."
CITE_RE = re.compile(r"\[([A-Za-z0-9_\-/.]+#\d+)\]")

SYSTEM_PROMPT = f"""You answer questions about Indian government document procedures (Aadhaar, OCI).
Rules:
1. Use ONLY the numbered source chunks provided. Never use outside knowledge.
2. After every sentence, cite the chunk it relies on in square brackets using its exact ID, e.g. [DOC-ID#03].
3. If two chunks disagree, follow the one with the later issue date and say so.
4. If the chunks do not contain the answer, reply exactly: "{ABSTAIN}"
5. Be concise: at most 5 sentences or a short numbered list."""


def _context(chunks: list[Chunk]) -> str:
    blocks = []
    for c in chunks:
        meta = f"{c.title} | {c.doc_type} | circular {c.circular_no or '-'} | issued {c.issue_date or 'unknown'}"
        blocks.append(f"[{c.chunk_id}] ({meta})\nSection: {c.section}\n{c.text}")
    return "\n\n".join(blocks)


def llm_answer(question: str, chunks: list[Chunk]) -> str:
    user = f"Source chunks:\n\n{_context(chunks)}\n\nQuestion: {question}"
    return llm.complete(SYSTEM_PROMPT, user).strip()


def extractive_answer(question: str, chunks: list[Chunk], min_cover: float = 0.34,
                      max_units: int = 2) -> str:
    """No-LLM fallback: picks the sentences / table rows / procedures that best
    cover the question terms and cites each one. Abstains below min_cover."""
    q = set(tokenize(question))
    if not q or not chunks:
        return ABSTAIN
    units = []
    for rank, c in enumerate(chunks):
        if c.chunk_type == "procedure":
            pieces = [c.text]
        elif c.chunk_type == "table":
            rows = [r.strip().strip("|").strip() for r in c.text.splitlines()
                    if r.strip() and not re.match(r"^:?-{2,}", r.strip("| "))]
            header = [h.strip() for h in rows[0].split("|")] if rows else []
            pieces = []
            for r in rows[1:]:  # one unit per row, labelled with the header
                cells = [x.strip() for x in r.split("|")]
                pieces.append(f"{c.section}: " + "; ".join(
                    f"{h}: {v}" if i else v for i, (h, v) in enumerate(zip(header, cells))))
        else:
            pieces = [s for s in re.split(r"(?<=[.!?])\s+", c.text) if s.strip()]
        for p in pieces:
            toks = set(tokenize(c.section + " " + p))
            wt = lambda t: 0.3 if t.isdigit() else 1.0  # numbers in questions are usually user facts
            cover = sum(wt(t) for t in q & toks) / sum(wt(t) for t in q)
            units.append((cover - 0.03 * rank, cover, p.strip(), c.chunk_id))
    units.sort(key=lambda u: -u[0])
    if not units or units[0][1] < min_cover:
        return ABSTAIN
    # a distinctive question word that appears in none of the retrieved chunks
    # (e.g. "tatkal") signals the documents do not cover the topic
    seen = CORPUS_VOCAB or set().union(*(set(tokenize(c.index_text)) for c in chunks))
    missing = [t for t in q if not t.isdigit() and len(t) >= 4 and t not in seen and t not in GENERIC]
    if missing and units[0][1] < 0.75:
        return ABSTAIN
    top_cover = units[0][1]
    best = [u for u in units[:max_units] if u[1] >= 0.85 * top_cover]
    return " ".join(f"{u[2]} [{u[3]}]" for u in best)


def generate(question: str, chunks: list[Chunk], use_llm: bool | None = None) -> dict:
    use_llm = (llm.provider() is not None) if use_llm is None else use_llm
    mode = f"llm:{llm.provider()}/{llm.model_name()}" if use_llm else "extractive (no LLM key)"
    try:
        raw = llm_answer(question, chunks) if use_llm else extractive_answer(question, chunks)
    except Exception as e:
        raw, mode = extractive_answer(question, chunks), f"extractive (LLM error: {type(e).__name__})"
    by_id = {c.chunk_id: c for c in chunks}
    cited = [cid for cid in CITE_RE.findall(raw) if cid in by_id]
    order: list[str] = []
    for cid in cited:
        if cid not in order:
            order.append(cid)
    rendered = raw
    for n, cid in enumerate(order, start=1):
        rendered = rendered.replace(f"[{cid}]", f"[{n}]")
    rendered = CITE_RE.sub("", rendered)  # drop IDs the model invented
    refs = [{"n": n, "chunk_id": cid, "citation": by_id[cid].citation(),
             "superseded": False} for n, cid in enumerate(order, start=1)]
    return {"answer": rendered.strip(), "raw": raw, "citations": cited, "references": refs,
            "abstained": raw.strip().startswith(ABSTAIN[:30]), "mode": mode}
