"""Structure-aware parsing and chunking (Section 3.2).

Every source document is first normalised to a Markdown file with a YAML
front-matter block that carries its metadata (see ingest.py). This module
reads those files and splits them along document structure:

* tables      -> one chunk per table, header row kept (re-attached if split)
* procedures  -> one chunk per numbered procedure, heading to heading
* prose       -> sentence-bounded windows with overlap
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

MAX_PROSE_WORDS = 120      # target size of a prose chunk
OVERLAP_SENTENCES = 1      # sentences repeated between consecutive prose chunks
MAX_TABLE_ROWS = 25        # very long tables are split, header repeated
MAX_PROCEDURE_WORDS = 400  # a procedure longer than this is split by steps

NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+")
BULLET = re.compile(r"^\s*[-*•]\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    service: str
    doc_type: str
    circular_no: str
    issue_date: str            # ISO yyyy-mm-dd ("" if unknown)
    url: str
    supersedes: list = field(default_factory=list)
    section: str = ""
    chunk_type: str = "prose"  # prose | table | procedure
    text: str = ""
    is_sample: bool = False

    @property
    def index_text(self) -> str:
        """Text used for retrieval: title and section give context to short chunks."""
        return f"{self.title}. {self.section}. {self.text}"

    def citation(self) -> str:
        parts = [self.title]
        if self.circular_no:
            parts.append(f"Circular {self.circular_no}")
        if self.issue_date:
            parts.append(f"dated {self.issue_date}")
        return ", ".join(parts) + f" ({self.url})"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- io
def read_markdown_doc(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path} has no front matter")
    _, fm, body = raw.split("---", 2)
    meta = yaml.safe_load(fm) or {}
    meta["issue_date"] = str(meta.get("issue_date") or "")
    meta["supersedes"] = list(meta.get("supersedes") or [])
    meta["circular_no"] = str(meta.get("circular_no") or "")
    return meta, body.strip()


# --------------------------------------------------------------- segmentation
def _sections(body: str) -> list[tuple[str, list[str]]]:
    """Split the body at ##/### headings. The # title line is dropped."""
    sections, heading, lines = [], "Overview", []
    for line in body.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            if len(m.group(1)) == 1:
                continue
            if any(l.strip() for l in lines):
                sections.append((heading, lines))
            heading, lines = m.group(2).strip(), []
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((heading, lines))
    return sections


def _blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group a section's lines into table / list / prose blocks."""
    blocks: list[tuple[str, list[str]]] = []
    for line in lines:
        s = line.rstrip()
        if not s.strip():
            if blocks and blocks[-1][0] == "prose":
                blocks.append(("break", []))
            continue
        kind = "table" if s.lstrip().startswith("|") else (
            "list" if NUMBERED.match(s) or BULLET.match(s) else "prose")
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(s)
        else:
            blocks.append((kind, [s]))
    return [b for b in blocks if b[0] != "break"]


def _split_table(rows: list[str]) -> list[str]:
    rows = [r for r in rows if r.strip()]
    if len(rows) < 2:
        return ["\n".join(rows)]
    header = rows[:2] if re.match(r"^\|?\s*:?-{2,}", rows[1].strip().strip("|").strip()) else rows[:1]
    data = rows[len(header):]
    out = []
    for i in range(0, max(len(data), 1), MAX_TABLE_ROWS):
        out.append("\n".join(header + data[i:i + MAX_TABLE_ROWS]))
    return out


def _split_prose(text: str) -> list[str]:
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > MAX_PROSE_WORDS:
            chunks.append(" ".join(cur))
            cur = cur[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_document(meta: dict, body: str, is_sample: bool = False) -> list[Chunk]:
    chunks: list[Chunk] = []

    def add(section: str, ctype: str, text: str):
        chunks.append(Chunk(
            chunk_id=f"{meta['doc_id']}#{len(chunks) + 1:02d}",
            doc_id=meta["doc_id"], title=meta.get("title", ""),
            source=meta.get("source", ""), service=meta.get("service", ""),
            doc_type=meta.get("doc_type", ""), circular_no=meta["circular_no"],
            issue_date=meta["issue_date"], url=meta.get("url", ""),
            supersedes=meta["supersedes"], section=section, chunk_type=ctype,
            text=text.strip(), is_sample=is_sample))

    for heading, lines in _sections(body):
        blocks = _blocks(lines)
        has_steps = any(k == "list" and any(NUMBERED.match(l) for l in ls) for k, ls in blocks)
        if has_steps:
            # Procedure: keep heading-to-heading text (intro, steps, notes) together.
            parts = [("\n".join(ls) if k in ("list", "table") else " ".join(x.strip() for x in ls))
                     for k, ls in blocks]
            text = "\n".join(parts)
            if len(text.split()) <= MAX_PROCEDURE_WORDS:
                add(heading, "procedure", text)
            else:  # rare: split by steps, never inside a step
                steps, cur = [], []
                for line in text.splitlines():
                    if NUMBERED.match(line) and cur and sum(len(c.split()) for c in cur) > MAX_PROCEDURE_WORDS // 2:
                        steps.append("\n".join(cur)); cur = []
                    cur.append(line)
                steps.append("\n".join(cur))
                for s in steps:
                    add(heading, "procedure", s)
            continue
        prose_buf: list[str] = []
        for kind, ls in blocks:
            if kind == "table":
                if prose_buf:
                    for p in _split_prose(" ".join(prose_buf)):
                        add(heading, "prose", p)
                    prose_buf = []
                for t in _split_table(ls):
                    add(heading, "table", t)
            else:
                prose_buf.append(" ".join(x.strip() for x in ls))
        if prose_buf:
            for p in _split_prose(" ".join(prose_buf)):
                add(heading, "prose", p)
    return chunks


def load_corpus(corpus_dir: str | Path, is_sample: bool | None = None) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    sample = "sample" in corpus_dir.name if is_sample is None else is_sample
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        meta, body = read_markdown_doc(path)
        chunks.extend(chunk_document(meta, body, is_sample=sample))
    return chunks


if __name__ == "__main__":
    import sys
    from collections import Counter
    cs = load_corpus(sys.argv[1] if len(sys.argv) > 1 else "data/sample_corpus")
    print(f"{len(cs)} chunks from {len({c.doc_id for c in cs})} documents")
    print(Counter(c.chunk_type for c in cs))
    for c in cs[:3]:
        print("-" * 60, "\n", c.chunk_id, c.chunk_type, c.section, "\n", c.text)
