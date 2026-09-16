"""
Standalone test for RAG retrieval — no LLM involved yet.
Confirms the vector index actually returns sensible chunks for a question
before you wire retrieval into the full RAG agent.

Run:
    python test_retrieval.py
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "dram_express_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TEST_QUESTIONS = [
    "Can I get a refund?",
    "Do you ship to Florida?",
    "What happens if I miss a payment?",
    "How do I cancel my order?",
    "Is DRAM Express trustworthy?",
    "How much DRAM do I get with the Legendary tier?",
    "Can I switch from Basic to Deluxe?",
    "How long does shipping take?",
    "What does the warranty cover?",
    "Where can I leave a review?",
]


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    for question in TEST_QUESTIONS:
        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print("=" * 70)

        query_embedding = model.encode([question]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=1)

        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n--- match {i + 1} (source: {meta['source']}) ---")
            print(doc[:600] + ("..." if len(doc) > 600 else ""))


if __name__ == "__main__":
    main()