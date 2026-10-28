"""Embed the chunks of every strategy and store them in Postgres (pgvector).

Reads data/chunks_<strategy>.jsonl, writes rows to the `chunks` table.
Safe to re-run: each strategy's old rows are replaced.

Needs the database running first:
    docker start ragdb

Run from the project root with the .venv Python:
    uv run python ingest/embed.py
"""
import json
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:rag@localhost:5432/rag")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small's output size
BATCH_SIZE = 32
STRATEGIES = ["fixed", "recursive", "structure"]

CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    text PRIMARY KEY,
    strategy    text NOT NULL,
    doc_id      text NOT NULL,
    doc_type    text NOT NULL,
    access      text NOT NULL,
    source_url  text,
    title       text,
    rule_ids    text[],
    pages       int[],
    citations   text[],
    n_tokens    int,
    text        text NOT NULL,
    embedding   vector({EMBEDDING_DIM}) NOT NULL
);
-- HNSW: an approximate nearest-neighbour index, so search doesn't compare against every row.
-- vector_cosine_ops matches the <=> (cosine distance) operator used when searching.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy);
"""

INSERT = """
INSERT INTO chunks (chunk_id, strategy, doc_id, doc_type, access, source_url, title,
                    rule_ids, pages, citations, n_tokens, text, embedding)
VALUES (%(chunk_id)s, %(strategy)s, %(doc_id)s, %(doc_type)s, %(access)s, %(source_url)s, %(title)s,
        %(rule_ids)s, %(pages)s, %(citations)s, %(n_tokens)s, %(text)s, %(embedding)s)
"""


def main() -> None:
    print(f"Loading {MODEL_NAME} (downloads ~130 MB on first run)...")
    model = SentenceTransformer(MODEL_NAME)

    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)  # lets psycopg send numpy arrays as pgvector values
        conn.execute(CREATE_TABLE)

        for strategy in STRATEGIES:
            chunk_file = DATA_DIR / f"chunks_{strategy}.jsonl"
            chunks = [json.loads(line) for line in chunk_file.open(encoding="utf-8")]

            # Chunks are embedded as-is. The BGE query instruction prefix is only
            # added to questions at search time.
            # normalize_embeddings=True gives unit-length vectors, so cosine distance is meaningful.
            embeddings = model.encode(
                [c["text"] for c in chunks],
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            rows = [
                {**{k: c[k] for k in ("chunk_id", "strategy", "doc_id", "doc_type", "access",
                                      "source_url", "title", "rule_ids", "pages", "citations",
                                      "n_tokens", "text")},
                 "embedding": embedding}
                for c, embedding in zip(chunks, embeddings)
            ]

            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
                with conn.cursor() as cur:
                    cur.executemany(INSERT, rows)
            print(f"{strategy:>9}: stored {len(rows)} chunks")

        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"\nTable `chunks` now holds {total} rows.")


if __name__ == "__main__":
    main()
