"""Split cleaned records into chunks using three strategies, one output file each.

  fixed      - LangChain TokenTextSplitter: fixed-size token windows with overlap
               over the whole document; ignores rule/page boundaries
  recursive  - LangChain RecursiveCharacterTextSplitter: split on paragraphs, then
               lines, then sentences, then words, up to the size limit, over the whole document
  structure  - our own: one chunk per FCA rule / PDF page, split with the recursive
               splitter only if too long

A fixed or recursive chunk can span several rules/pages, so it cites all of them.

Reads data/cleaned.jsonl, writes data/chunks_<strategy>.jsonl.

Run from the project root with the .venv Python:
    uv run python ingest/chunk.py
"""
import json
from bisect import bisect_right
from pathlib import Path
from statistics import median

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "cleaned.jsonl"

# Sizes are in tokens (tiktoken's cl100k_base). 400 tokens stays inside the
# 512-token limit of small embedding models such as bge-small.
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 60
ENCODING = "cl100k_base"
RECORD_SEPARATOR = "\n\n"  # between rules/pages when a document is joined into one text

tokenizer = tiktoken.get_encoding(ENCODING)

fixed_splitter = TokenTextSplitter(
    encoding_name=ENCODING, chunk_size=CHUNK_TOKENS, chunk_overlap=OVERLAP_TOKENS,
    add_start_index=True,
)
recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=ENCODING, chunk_size=CHUNK_TOKENS, chunk_overlap=OVERLAP_TOKENS,
    separators=["\n\n", "\n", ". ", "; ", " ", ""],
    add_start_index=True,
)


def split_document(splitter, doc: list[tuple[int, dict]]) -> list[tuple[str, list[int]]]:
    """Join a document's records into one text, split it, and map each chunk back to the
    records it overlaps using the chunk's start position in the joined text."""
    starts, parts, offset = [], [], 0
    for _, record in doc:
        starts.append(offset)
        parts.append(record["text"])
        offset += len(record["text"]) + len(RECORD_SEPARATOR)
    text = RECORD_SEPARATOR.join(parts)

    chunks = []
    for chunk in splitter.create_documents([text]):
        start = chunk.metadata["start_index"]
        end = start + len(chunk.page_content)
        first, last = bisect_right(starts, start) - 1, bisect_right(starts, end - 1) - 1
        chunks.append((chunk.page_content, [doc[i][0] for i in range(max(first, 0), last + 1)]))
    return chunks


def chunk_fixed(doc: list[tuple[int, dict]]) -> list[tuple[str, list[int]]]:
    return split_document(fixed_splitter, doc)


def chunk_recursive(doc: list[tuple[int, dict]]) -> list[tuple[str, list[int]]]:
    return split_document(recursive_splitter, doc)


def chunk_structure(doc: list[tuple[int, dict]]) -> list[tuple[str, list[int]]]:
    # Each record is already one structural unit (an FCA provision or a PDF page);
    # the splitter returns it unchanged when it fits.
    return [(text, [i]) for i, record in doc for text in recursive_splitter.split_text(record["text"])]


STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "structure": chunk_structure,
}


def citation(record: dict) -> str:
    if record["rule_id"]:
        return f"{record['rule_id']} {record['provision_type'] or ''}".strip()
    return f"{record['doc_id']} p.{record['page']}"


def main() -> None:
    records = [json.loads(line) for line in INPUT_FILE.open(encoding="utf-8")]

    docs: dict[str, list[tuple[int, dict]]] = {}
    for i, record in enumerate(records):
        docs.setdefault(record["doc_id"], []).append((i, record))

    for name, split in STRATEGIES.items():
        chunks = []
        for doc_id, doc in docs.items():
            for part, (text, sources) in enumerate(split(doc)):
                first = records[sources[0]]
                metadata = {k: v for k, v in first.items() if k not in ("text", "rule_id", "provision_type", "effective_date", "page")}
                chunks.append({
                    "chunk_id": f"{name}-{doc_id}-{part}",
                    "strategy": name,
                    **metadata,
                    "rule_ids": [records[s]["rule_id"] for s in sources if records[s]["rule_id"]],
                    "pages": [records[s]["page"] for s in sources if records[s]["page"] is not None],
                    "citations": [citation(records[s]) for s in sources],
                    "n_tokens": len(tokenizer.encode(text)),
                    "text": text,
                })

        output_file = DATA_DIR / f"chunks_{name}.jsonl"
        with output_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        sizes = [c["n_tokens"] for c in chunks]
        multi = sum(len(c["citations"]) > 1 for c in chunks)
        print(f"{name:>9}: {len(chunks):4} chunks, tokens min/median/max "
              f"{min(sizes)}/{median(sizes):.0f}/{max(sizes)}, {multi} span >1 rule/page -> {output_file.name}")


if __name__ == "__main__":
    main()
