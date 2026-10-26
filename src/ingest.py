"""Convert scraped HTML/PDF files into normalised Markdown with front matter.

    python -m src.ingest            # data/raw -> data/corpus

* HTML: headings become ##, <table> becomes a pipe table, <ol> items become
  numbered steps, so chunking.py can keep tables and procedures whole.
* PDF : tables are extracted with pdfplumber (header row kept) and removed
  from the running text; short title-like lines become ## headings.
* Supersession: circular numbers found after "in supersession of" are mapped
  to the doc_ids that carry those circular numbers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from bs4 import BeautifulSoup

RAW = Path("data/raw")
OUT = Path("data/corpus")
OVERRIDES = Path("data/manual_overrides.json")


def _cell(x) -> str:
    return re.sub(r"\s+", " ", str(x or "")).replace("|", "/").strip()


def _pipe_table(rows: list[list]) -> str:
    rows = [[_cell(c) for c in r] for r in rows if r and any(_cell(c) for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def html_to_md(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "noscript", "form"]):
        t.decompose()
    main = soup.find("main") or soup.find(id=re.compile("content|main", re.I)) or soup.body or soup
    out = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"], recursive=True):
        if el.find_parent("table") and el.name != "table":
            continue
        if el.name == "table":
            out.append(_pipe_table([[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                                    for tr in el.find_all("tr")]))
        elif el.name in ("h1", "h2", "h3", "h4"):
            out.append(f"\n## {el.get_text(' ', strip=True)}")
        elif el.name == "li":
            if el.find_parent("table"):
                continue
            parent = el.find_parent(["ol", "ul"])
            if parent is not None and parent.name == "ol":
                idx = [li for li in parent.find_all("li", recursive=False)].index(el) + 1 if el in parent.find_all("li", recursive=False) else 1
                out.append(f"{idx}. {el.get_text(' ', strip=True)}")
            else:
                out.append(f"- {el.get_text(' ', strip=True)}")
        else:
            txt = el.get_text(" ", strip=True)
            if txt:
                out.append(txt + "\n")
    return "\n".join(out)


def pdf_to_md(path: Path) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]
            for t in tables:
                parts.append(_pipe_table(t.extract()))
            def outside(obj):
                return not any(b[0] <= obj["x0"] <= b[2] and b[1] <= obj["top"] <= b[3] for b in bboxes)
            text = page.filter(outside).extract_text() or ""
            for line in text.splitlines():
                s = line.strip()
                if not s:
                    continue
                is_heading = (len(s.split()) <= 10 and not s.endswith(".") and
                              (s.isupper() or re.match(r"^(\d+(\.\d+)*|[A-Z]\.|[IVX]+\.)\s+[A-Z]", s)))
                parts.append(f"\n## {s.title() if s.isupper() else s}" if is_heading else s)
    return "\n".join(parts)


def ingest() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW / "manifest.json").read_text())
    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    by_circ = {d["circular_no"]: d["doc_id"] for d in manifest if d.get("circular_no")}
    for d in manifest:
        d.update(overrides.get(d["doc_id"], {}))
        path = Path(d["local_path"])
        body = pdf_to_md(path) if path.suffix == ".pdf" else html_to_md(path.read_text(errors="ignore"))
        supersedes = d.get("supersedes") or [by_circ[c] for c in d.get("supersedes_circulars", []) if c in by_circ]
        meta = {k: d.get(k, "") for k in ("doc_id", "title", "source", "service", "doc_type",
                                          "circular_no", "issue_date", "url")}
        meta["supersedes"] = supersedes
        meta["retrieved_on"] = d.get("retrieved_on", "")
        md = "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n# " + meta["title"] + "\n\n" + body
        (OUT / f"{d['doc_id']}.md").write_text(md, encoding="utf-8")
        print(f"  {d['doc_id']}: {len(body.split())} words, date={meta['issue_date'] or 'MISSING'}")
    print(f"Wrote {len(manifest)} documents to {OUT}/")


if __name__ == "__main__":
    ingest()
