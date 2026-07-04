"""
Chunk cleaned corpus texts into retrieval-sized records with metadata, written as
JSONL to corpus/chunks.jsonl (input to embed_corpus.py).

Special-case chunking (per the gameplan):
  - ninety_five_theses: one numbered thesis per chunk
  - large_catechism / small_catechism: one Q&A unit per chunk
  - table_talk: one numbered entry (roman-numeral marker) per chunk
  - everything else: paragraph-grouped chunks, ~400-600 tokens, ~50-100 token overlap

Token counts are approximated as words * 1.3 (close enough for chunk sizing without
pulling in a tokenizer dependency).
"""
import json
import re
from pathlib import Path

CLEAN_DIR = Path(__file__).resolve().parent.parent / "corpus" / "clean"
OUT_PATH = Path(__file__).resolve().parent.parent / "corpus" / "chunks.jsonl"

# work slug -> (year, category) -- mirrors fetch_corpus.py's SOURCES metadata
WORK_META = {
    "ninety_five_theses": (1517, "theology"),
    "freedom_of_a_christian": (1520, "theology"),
    "address_to_christian_nobility": (1520, "political"),
    "treatise_on_good_works": (1520, "pastoral"),
    "commentary_on_galatians": (1535, "theology"),
    "large_catechism": (1529, "catechetical"),
    "small_catechism": (1529, "catechetical"),
    "selected_sermons": (1518, "pastoral"),
    "heidelberg_disputation": (1518, "theology"),
    "bondage_of_the_will": (1525, "theology"),
    "table_talk": (1566, "pastoral"),
}

WORK_TITLES = {
    "ninety_five_theses": "The Ninety-Five Theses",
    "freedom_of_a_christian": "On the Freedom of a Christian",
    "address_to_christian_nobility": "Address to the Christian Nobility",
    "treatise_on_good_works": "A Treatise on Good Works",
    "commentary_on_galatians": "Commentary on Galatians",
    "large_catechism": "The Large Catechism",
    "small_catechism": "The Small Catechism",
    "selected_sermons": "Sermon on Threefold Righteousness",
    "heidelberg_disputation": "The Heidelberg Disputation",
    "bondage_of_the_will": "The Bondage of the Will",
    "table_talk": "Table Talk",
}

WORDS_PER_TOKEN = 1.3
TARGET_MIN_TOKENS, TARGET_MAX_TOKENS = 400, 600
OVERLAP_TOKENS = 75


def approx_tokens(text: str) -> int:
    return int(len(text.split()) * WORDS_PER_TOKEN)


def make_record(slug: str, text: str, extra: dict | None = None) -> dict:
    year, category = WORK_META[slug]
    record = {
        "text": text.strip(),
        "metadata": {
            "source": WORK_TITLES[slug],
            "year": year,
            "category": category,
        },
    }
    if extra:
        record["metadata"].update(extra)
    return record


def chunk_theses(slug: str, text: str) -> list[dict]:
    # Theses are numbered "N." alone on a line, followed by the thesis body.
    parts = re.split(r"\n(\d{1,3})\.\n", "\n" + text)
    records = []
    # parts alternates: [preamble, num, body, num, body, ...]
    for i in range(1, len(parts), 2):
        num, body = parts[i], parts[i + 1]
        body = body.strip()
        if body:
            records.append(make_record(slug, body, {"thesis_number": int(num)}))
    return records


def chunk_catechism(slug: str, text: str) -> list[dict]:
    # Q./A. pairs on their own lines, question and answer text follow each marker.
    lines = text.split("\n")
    records = []
    i = 0
    pending_q = None
    while i < len(lines):
        if lines[i].strip() == "Q.":
            q_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "A.":
                q_lines.append(lines[i])
                i += 1
            pending_q = "\n".join(q_lines).strip()
        elif lines[i].strip() == "A.":
            a_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() not in ("Q.", "A."):
                a_lines.append(lines[i])
                i += 1
            answer = "\n".join(a_lines).strip()
            if pending_q and answer:
                combined = f"Q: {pending_q}\nA: {answer}"
                records.append(make_record(slug, combined))
            pending_q = None
        else:
            i += 1
    return records


def chunk_table_talk(slug: str, text: str) -> list[dict]:
    # Numbered entries marked by a lone roman numeral line, e.g. "XXV."
    roman_marker = re.compile(r"^[IVXLCDM]+\.$")
    lines = text.split("\n")
    entries: list[list[str]] = []
    current: list[str] = []
    started = False
    for line in lines:
        if roman_marker.match(line.strip()):
            if started and current:
                entries.append(current)
            current = []
            started = True
        elif started:
            current.append(line)
    if current:
        entries.append(current)

    records = []
    for entry_lines in entries:
        body = "\n".join(entry_lines).strip()
        if body and approx_tokens(body) >= 15:  # skip stray/empty markers
            records.append(make_record(slug, body))
    return records


def split_oversized(paragraph: str) -> list[str]:
    """Break a paragraph that alone exceeds the max chunk size into sentence-bounded pieces."""
    if approx_tokens(paragraph) <= TARGET_MAX_TOKENS:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, buf, buf_tokens = [], [], 0
    for sent in sentences:
        t = approx_tokens(sent)
        if buf_tokens + t > TARGET_MAX_TOKENS and buf:
            pieces.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(sent)
        buf_tokens += t
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def chunk_paragraphs(slug: str, text: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        # front matter didn't preserve blank-line breaks; fall back to single-newline
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # Wittenberg HTML often yields a handful of very long block-level paragraphs;
    # break any paragraph that alone exceeds the target chunk size.
    expanded = []
    for p in paragraphs:
        expanded.extend(split_oversized(p))
    paragraphs = expanded

    records = []
    buffer: list[str] = []
    buffer_tokens = 0
    part_num = 0

    def flush():
        nonlocal buffer, buffer_tokens, part_num
        if buffer:
            chunk_text = "\n\n".join(buffer)
            records.append(make_record(slug, chunk_text, {"part": part_num}))
            part_num += 1

    for para in paragraphs:
        para_tokens = approx_tokens(para)
        if buffer_tokens + para_tokens > TARGET_MAX_TOKENS and buffer_tokens >= TARGET_MIN_TOKENS:
            flush()
            # keep the tail of the previous buffer as overlap
            overlap_buf: list[str] = []
            overlap_tokens = 0
            for p in reversed(buffer):
                t = approx_tokens(p)
                if overlap_tokens + t > OVERLAP_TOKENS:
                    break
                overlap_buf.insert(0, p)
                overlap_tokens += t
            buffer = overlap_buf
            buffer_tokens = overlap_tokens
        buffer.append(para)
        buffer_tokens += para_tokens

    flush()
    return records


SPECIAL_CHUNKERS = {
    "ninety_five_theses": chunk_theses,
    "small_catechism": chunk_catechism,
    "table_talk": chunk_table_talk,
}


def main():
    all_records = []
    for clean_path in sorted(CLEAN_DIR.glob("*.txt")):
        slug = clean_path.stem
        if slug not in WORK_META:
            print(f"Skipping unrecognized file: {clean_path.name}")
            continue
        text = clean_path.read_text(encoding="utf-8")
        chunker = SPECIAL_CHUNKERS.get(slug, chunk_paragraphs)
        records = chunker(slug, text)
        print(f"{slug}: {len(records)} chunks ({chunker.__name__})")
        all_records.extend(records)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_records)} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
