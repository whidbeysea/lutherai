"""Quick manual sanity check: embed a test query with voyage-4-lite and retrieve
the top matches from the ChromaDB collection built by embed_corpus.py."""
import os
import sys

import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "luther_corpus"
QUERY_EMBED_MODEL = "voyage-4-lite"


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "faith and works"

    api_key = os.environ.get("VOYAGE_API_KEY")
    voyage = voyageai.Client(api_key=api_key)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    result = voyage.embed([query], model=QUERY_EMBED_MODEL, input_type="query")
    hits = collection.query(query_embeddings=result.embeddings, n_results=5)

    print(f"Query: {query!r}\n")
    for doc, meta, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0]):
        print(f"[{meta['source']}, {meta['year']}] (distance={dist:.3f})")
        print(doc[:250].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
