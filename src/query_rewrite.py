"""Query rewriting (Section 3.5): service detection plus optional HyDE expansion."""
from __future__ import annotations

import re

from . import llm

AADHAAR_TERMS = r"\b(aadhaar|aadhar|adhaar|uidai|uid|eid|myaadhaar|baal|pvc|enrolment centre|seva kendra|e-aadhaar)\b"
OCI_TERMS = r"\b(oci|overseas citizen|pio|frro|ociservices|foreign national|foreign passport|consulate|mission)\b"


def detect_service(question: str) -> str | None:
    q = question.lower()
    a, o = bool(re.search(AADHAAR_TERMS, q)), bool(re.search(OCI_TERMS, q))
    if a and not o:
        return "aadhaar"
    if o and not a:
        return "oci"
    return None  # ambiguous or unknown: search both services


def hyde_expand(question: str) -> str:
    """HyDE (Gao et al., 2023): append a hypothetical answer so the dense
    retriever matches answer-like passages. Needs an LLM key; otherwise no-op."""
    if llm.provider() is None:
        return question
    try:
        hypo = llm.complete(
            "You write short, factual-sounding passages in the style of Indian government FAQs.",
            f"Write a 2-3 sentence passage that would answer: {question}", max_tokens=150)
        return f"{question}\n{hypo.strip()}"
    except Exception:
        return question


def rewrite(question: str, use_hyde: bool = False) -> dict:
    return {"service": detect_service(question),
            "dense_query": hyde_expand(question) if use_hyde else question}
