"""Corpus collection (Section 3.1).

Fetches the seed pages in data/sources.json, downloads linked PDFs whose link
text or URL matches the seed's keywords, and records metadata for every file
in data/raw/manifest.json:

    doc_id, url, title, service, doc_type, circular_no, issue_date,
    supersedes_circulars, retrieved_on, local_path, date_source

Issue dates and circular numbers are extracted with regular expressions; any
value can be corrected by hand in data/manual_overrides.json (keyed by doc_id),
because the recency rule depends on correct dates.

Run on a machine with normal internet access:
    python -m src.scraper
    python -m src.ingest          # converts raw files into data/corpus/*.md
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

RAW = Path("data/raw")
HEADERS = {"User-Agent": "CitedAndCurrent-StudentProject/1.0 (academic research; contact: NMIMS MPSTME)"}
DELAY_S = 1.5  # be polite to government servers

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20\d{2})\b"), "dmy"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})[,]?\s+(20\d{{2}})\b", re.I), "d_month_y"),
    (re.compile(rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b", re.I), "month_d_y"),
]
CIRCULAR_RE = re.compile(r"\b(?:No\.?|Number|Circular No\.?|F\.?\s?No\.?)\s*[:.]?\s*([A-Z0-9][A-Za-z0-9()/.\-]{5,60})")
SUPERSEDE_RE = re.compile(r"in supersession of[^.]{0,200}?(?:No\.?|Number)\s*[:.]?\s*([A-Z0-9][A-Za-z0-9()/.\-]{5,60})", re.I)


def find_date(text: str) -> str:
    """Earliest-in-text plausible date, returned as ISO (the header date on a circular)."""
    best = None
    for pat, kind in DATE_PATTERNS:
        for m in pat.finditer(text[:4000]):
            try:
                if kind == "dmy":
                    d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                elif kind == "d_month_y":
                    d = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").date()
                else:
                    d = datetime.strptime(f"{m.group(2)} {m.group(1)} {m.group(3)}", "%d %B %Y").date()
            except ValueError:
                continue
            if 2000 <= d.year <= date.today().year and (best is None or m.start() < best[0]):
                best = (m.start(), d)
    return best[1].isoformat() if best else ""


def find_circular_no(text: str) -> str:
    m = CIRCULAR_RE.search(text[:3000])
    return m.group(1).rstrip(".") if m else ""


def find_supersedes(text: str) -> list[str]:
    return sorted({m.group(1).rstrip(".") for m in SUPERSEDE_RE.finditer(text)})


def _doc_id(url: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(urlparse(url).path).stem or urlparse(url).netloc)[:40].strip("-")
    return f"{stem}-{hashlib.md5(url.encode()).hexdigest()[:6]}".upper()


def _get(session: requests.Session, url: str) -> requests.Response | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        time.sleep(DELAY_S)
        return r
    except requests.RequestException as e:
        print(f"  ! {url}: {e}")
        return None


def _pdf_text(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages[:3])
    except Exception:
        return ""


def scrape(sources_path: str = "data/sources.json") -> list[dict]:
    seeds = json.loads(Path(sources_path).read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest_path = RAW / "manifest.json"
    manifest = {d["doc_id"]: d for d in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}

    for seed in seeds:
        print(f"[seed] {seed['url']}")
        r = _get(session, seed["url"])
        if r is None:
            continue
        targets = [(seed["url"], seed.get("title", ""), r)]
        if seed.get("follow_pdfs"):
            soup = BeautifulSoup(r.text, "html.parser")
            kws = [k.lower() for k in seed.get("keywords", [])]
            for a in soup.find_all("a", href=True):
                href, text = urljoin(seed["url"], a["href"]), a.get_text(" ", strip=True)
                if href.lower().endswith(".pdf") and (not kws or any(k in (text + href).lower() for k in kws)):
                    targets.append((href, text, None))
        for url, link_text, resp in targets[: seed.get("max_docs", 40)]:
            doc_id = _doc_id(url)
            if doc_id in manifest:
                continue
            resp = resp or _get(session, url)
            if resp is None:
                continue
            is_pdf = url.lower().endswith(".pdf") or "pdf" in resp.headers.get("content-type", "")
            local = RAW / f"{doc_id}.{'pdf' if is_pdf else 'html'}"
            local.write_bytes(resp.content)
            if is_pdf:
                text = _pdf_text(local)
                title = link_text or local.stem
            else:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(" ", strip=True)
                title = link_text or (soup.title.get_text(strip=True) if soup.title else doc_id)
            issue = find_date(text)
            manifest[doc_id] = {
                "doc_id": doc_id, "url": url, "title": title[:200], "service": seed["service"],
                "source": seed.get("source", urlparse(url).netloc), "doc_type": seed.get("doc_type", "faq"),
                "circular_no": find_circular_no(text), "issue_date": issue,
                "date_source": "extracted" if issue else "missing - set in manual_overrides.json",
                "supersedes_circulars": find_supersedes(text),
                "retrieved_on": date.today().isoformat(), "local_path": str(local)}
            print(f"  + {doc_id} | {issue or 'NO DATE'} | {title[:60]}")
    manifest_path.write_text(json.dumps(list(manifest.values()), indent=2))
    missing = [d["doc_id"] for d in manifest.values() if not d["issue_date"]]
    print(f"\n{len(manifest)} documents in manifest; {len(missing)} without a date (fix in data/manual_overrides.json)")
    return list(manifest.values())


if __name__ == "__main__":
    scrape()
