"""Query-time retrieval against the ChromaDB corpus collection built by
scripts/embed_corpus.py. Kept separate from the chat/response logic so both the
CLI test harness and the FastAPI app can share it."""
import os
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "luther_corpus"
QUERY_EMBED_MODEL = "voyage-4-lite"
TOP_K = 5

_voyage_client = None
_chroma_collection = None


def _voyage() -> voyageai.Client:
    global _voyage_client
    if _voyage_client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set")
        _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


def _collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return up to top_k {text, metadata, distance} records most relevant to query."""
    result = _voyage().embed([query], model=QUERY_EMBED_MODEL, input_type="query")
    hits = _collection().query(query_embeddings=result.embeddings, n_results=top_k)

    records = []
    for doc, meta, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0]):
        records.append({"text": doc, "metadata": meta, "distance": dist})
    return records
