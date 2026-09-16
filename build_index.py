"""
Chunks the DRAM Express knowledge base docs, embeds them locally with
sentence-transformers, and stores them in a persistent ChromaDB collection.

Run once (or any time you edit the docs) to rebuild the index:
    python build_index.py

Requires:
    pip install chromadb sentence-transformers
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("data/docs")
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dram_express_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, free, runs on CPU fine

CHUNK_SIZE = 300       # characters per chunk (smaller = sections stay distinct)
CHUNK_OVERLAP = 50     # characters shared between consecutive chunks


def _word_boundary_tail(text, max_chars):
    """Returns the tail of `text`, up to max_chars, but snapped to start at
    a word boundary rather than mid-word."""
    tail = text[-max_chars:]
    # if we cut mid-word, drop the partial fragment at the start
    space_index = tail.find(" ")
    if space_index != -1 and space_index < len(tail) - 1:
        tail = tail[space_index + 1:]
    return tail


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window character chunker. Splits on paragraph
    boundaries where possible so chunks don't cut mid-sentence."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying a bit of overlap from the tail of the
            # last one, snapped to a word boundary so it doesn't start mid-word
            if current:
                overlap_text = _word_boundary_tail(current, overlap)
                current = f"{overlap_text}\n\n{para}" if overlap_text else para
            else:
                current = para
    if current:
        chunks.append(current)

    return chunks


def load_docs():
    """Returns a list of (filename, chunk_text) tuples for every .md file in DOCS_DIR."""
    entries = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            entries.append(
                {
                    "id": f"{path.stem}-{i}",
                    "text": chunk,
                    "source": path.name,
                }
            )
    return entries


def main():
    print(f"Loading docs from {DOCS_DIR} ...")
    entries = load_docs()
    print(f"  Found {len(entries)} chunks across {len(list(DOCS_DIR.glob('*.md')))} files")

    print(f"Loading embedding model ({EMBEDDING_MODEL}) ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks ...")
    texts = [e["text"] for e in entries]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"Writing to ChromaDB at {CHROMA_PATH} ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # drop and recreate so re-running this script always reflects the current docs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[e["id"] for e in entries],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": e["source"]} for e in entries],
    )

    print(f"Done. Indexed {len(entries)} chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()