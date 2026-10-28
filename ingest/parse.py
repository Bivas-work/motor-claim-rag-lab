"""Parse the raw FCA HTML pages and policy PDFs into data/parsed.jsonl.

One record per FCA provision (rule or guidance) and one record per PDF page.

Run from the project root with the .venv Python:
    uv run python ingest/parse.py
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCES_CSV = DATA_DIR / "sources.csv"
OUTPUT_FILE = DATA_DIR / "parsed.jsonl"

DATE_PATTERN = re.compile(r"\d{2}/\d{2}/\d{4}")


def clean_text(text: str) -> str:
    """Collapse runs of spaces/tabs and blank lines, keep single line breaks."""
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def parse_html(path: Path, source: dict) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    title = soup.title.get_text(strip=True).removeprefix("FCA Handbook - ") if soup.title else None

    records = []
    for provision in soup.select("div.provision"):
        header = provision.select_one("div.header")
        rule_id = provision_type = effective_date = None
        if header is not None:
            label = header.select_one("label")
            rule_id = label.get_text(strip=True) if label else None
            type_span = header.select_one("span.provison-type")  # sic: the FCA site's own spelling
            provision_type = type_span.get_text(strip=True) if type_span else None
            date_match = DATE_PATTERN.search(header.get_text(" ", strip=True))
            effective_date = date_match.group() if date_match else None
            # Remove the header so the rule number/date/type aren't repeated in the text.
            header.decompose()

        # " " as separator: glossary links (e.g. "insurer") would otherwise split sentences.
        text = clean_text(provision.get_text(" ", strip=True))
        if not text:
            continue

        records.append({
            "doc_id": path.stem,
            "doc_type": source["doc_type"],
            "access": source["access"],
            "source_url": source["source_url"],
            "title": title,
            "rule_id": rule_id,
            "provision_type": provision_type,
            "effective_date": effective_date,
            "page": None,
            "text": text,
        })
    return records


def parse_pdf(path: Path, source: dict) -> list[dict]:
    reader = PdfReader(path)
    title = (reader.metadata.title if reader.metadata else None) or path.stem

    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue
        records.append({
            "doc_id": path.stem,
            "doc_type": source["doc_type"],
            "access": source["access"],
            "source_url": source["source_url"],
            "title": title,
            "rule_id": None,
            "provision_type": None,
            "effective_date": None,
            "page": page_number,
            "text": text,
        })
    return records


def main() -> None:
    with SOURCES_CSV.open(encoding="utf-8", newline="") as f:
        sources = list(csv.DictReader(f))

    all_records = []
    for source in sources:
        path = DATA_DIR / source["file"]
        if path.suffix == ".html":
            records = parse_html(path, source)
        elif path.suffix == ".pdf":
            records = parse_pdf(path, source)
        else:
            print(f"Skipping unsupported file: {path}")
            continue
        print(f"{path.name}: {len(records)} records")
        all_records.extend(records)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter(r["doc_type"] for r in all_records)
    print(f"\nWrote {len(all_records)} records to {OUTPUT_FILE}: {dict(counts)}")


if __name__ == "__main__":
    main()
