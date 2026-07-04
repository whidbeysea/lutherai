"""
Embed corpus/chunks.jsonl via Voyage AI and store the vectors + metadata in a local
ChromaDB collection under chroma_db/.

One-time cost: the whole corpus is ~1-2M tokens, well under Voyage's 200M free tier
across all voyage-4 models. Uses voyage-4-large for corpus embedding (better quality;
still free-tier) -- queries at retrieval time should use voyage-4-lite (cheaper), since
both share the same embedding space per Voyage's docs.
"""
import json
import os
import time
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv
from voyageai.error import RateLimitError

load_dotenv()

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "chunks.jsonl"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "luther_corpus"
EMBED_MODEL = "voyage-4-large"

# A payment method is now on file with Voyage, which lifts the unverified-account
# throttle (was 3 req/min, 10K tokens/min) -- standard rate limits apply, so batches
# can be large and back-to-back. Kept as constants (not inlined) in case a future
# fresh account ever needs the conservative pacing again.
MAX_TOKENS_PER_BATCH = 100_000
SECONDS_BETWEEN_REQUESTS = 0
WORDS_PER_TOKEN_ESTIMATE = 1.3


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * WORDS_PER_TOKEN_ESTIMATE)


def make_batches(chunks: list[dict]) -> list[list[dict]]:
    batches, current, current_tokens = [], [], 0
    for chunk in chunks:
        t = approx_tokens(chunk["text"])
        if current and current_tokens + t > MAX_TOKENS_PER_BATCH:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(chunk)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


def embed_with_retry(voyage: "voyageai.Client", texts: list[str], model: str, input_type: str):
    delay = 5  # retry backoff base, independent of the (now zero) inter-batch pacing
    for attempt in range(10):
        try:
            return voyage.embed(texts, model=model, input_type=input_type)
        except RateLimitError:
            print(f"    rate-limited, backing off {delay}s...")
            time.sleep(delay)
            delay = min(delay * 1.5, 180)
    raise RuntimeError("Exceeded retry attempts against Voyage rate limit")


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    import sys

    fresh = "--fresh" in sys.argv

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit("VOYAGE_API_KEY not set (add it to .env). Aborting before any API calls.")

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    voyage = voyageai.Client(api_key=api_key)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    already_embedded = 0
    if fresh:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(COLLECTION_NAME)
    else:
        # Resume support: if a prior run got rate-limited into a crash partway through,
        # pick up where it left off rather than re-embedding (and re-spending rate-limit
        # budget on) chunks already stored. Safe as long as chunks.jsonl hasn't changed,
        # since chunk_N ids are assigned in the same deterministic order every run.
        try:
            collection = client.get_collection(COLLECTION_NAME)
            already_embedded = collection.count()
        except Exception:
            collection = client.create_collection(COLLECTION_NAME)

    batches = make_batches(chunks)
    print(f"Split into {len(batches)} batches (~{MAX_TOKENS_PER_BATCH} tokens each, "
          f"paced {SECONDS_BETWEEN_REQUESTS}s apart for the unverified-account rate limit)")

    # Skip whole batches already covered by what's stored.
    chunk_index = 0
    start_batch = 0
    for i, batch in enumerate(batches):
        if chunk_index + len(batch) > already_embedded:
            start_batch = i
            break
        chunk_index += len(batch)
    else:
        start_batch = len(batches)

    if already_embedded:
        print(f"Resuming: {already_embedded} chunks already in the collection, "
              f"starting at batch {start_batch + 1}/{len(batches)}")

    total_tokens = 0
    for batch_num in range(start_batch, len(batches)):
        batch = batches[batch_num]
        texts = [c["text"] for c in batch]

        result = embed_with_retry(voyage, texts, EMBED_MODEL, "document")
        total_tokens += result.total_tokens

        collection.add(
            ids=[f"chunk_{chunk_index + i}" for i in range(len(batch))],
            embeddings=result.embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )
        chunk_index += len(batch)
        print(f"  batch {batch_num + 1}/{len(batches)}: embedded {chunk_index}/{len(chunks)} chunks "
              f"(running total this run: {total_tokens:,} tokens)")

        if batch_num < len(batches) - 1:
            time.sleep(SECONDS_BETWEEN_REQUESTS)

    print(f"\nDone. {chunk_index} chunks in collection, {total_tokens:,} tokens used this run "
          f"(free tier: 200,000,000). Collection stored at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
