"""Clean parsed records: remove PDF noise that would pollute chunks and search results.

Reads data/parsed.jsonl, writes data/cleaned.jsonl.

Run from the project root with the .venv Python:
    uv run python ingest/clean.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "parsed.jsonl"
OUTPUT_FILE = DATA_DIR / "cleaned.jsonl"

# A line that appears on more than this share of a PDF's pages is a running
# header/footer/sidebar (e.g. esure repeats its contents list on every page).
REPEATED_LINE_SHARE = 0.5
MIN_TEXT_CHARS = 100  # pages shorter than this after cleaning are covers/blank pages

DOT_LEADERS = re.compile(r"\.{4,}")         # "Contents ........ 12"
PAGE_NUMBER_LINE = re.compile(r"^\d{1,3}$")  # a line that is just "6"


def find_repeated_lines(pages: list[dict]) -> set[str]:
    counts = Counter(line.strip() for page in pages for line in set(page["text"].split("\n")))
    threshold = len(pages) * REPEATED_LINE_SHARE
    return {line for line, n in counts.items() if n > threshold and line}


def join_broken_lines(lines: list[str]) -> str:
    """PDF text breaks lines where the column ends. Re-join a line onto the previous
    one when it starts lower-case, i.e. it continues the same sentence."""
    joined: list[str] = []
    for line in lines:
        if joined and line[:1].islower():
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return "\n".join(joined)


def clean_pdf_page(text: str, repeated: set[str]) -> str:
    text = DOT_LEADERS.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [
        line for line in lines
        if line and line not in repeated and not PAGE_NUMBER_LINE.match(line)
    ]
    return re.sub(r" {2,}", " ", join_broken_lines(lines)).strip()


def main() -> None:
    records = [json.loads(line) for line in INPUT_FILE.open(encoding="utf-8")]

    pdf_pages = defaultdict(list)
    for record in records:
        if record["page"] is not None:
            pdf_pages[record["doc_id"]].append(record)
    repeated_by_doc = {doc_id: find_repeated_lines(pages) for doc_id, pages in pdf_pages.items()}

    cleaned, dropped = [], Counter()
    for record in records:
        if record["page"] is not None:
            record["text"] = clean_pdf_page(record["text"], repeated_by_doc[record["doc_id"]])
        # FCA HTML records were already cleaned in parse.py.
        if len(record["text"]) < MIN_TEXT_CHARS:
            dropped[record["doc_id"]] += 1
            continue
        cleaned.append(record)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in cleaned:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for doc_id, lines in repeated_by_doc.items():
        print(f"{doc_id}: removed {len(lines)} repeated lines, e.g. {sorted(lines)[:3]}")
    print(f"Dropped short records: {dict(dropped)}")
    print(f"Wrote {len(cleaned)} of {len(records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
