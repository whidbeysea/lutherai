"""
Clean raw fetched files into plain text per work, under corpus/clean/<slug>.txt.

Three input shapes to handle:
  - Project Wittenberg HTML pages: strip tags/nav via BeautifulSoup, drop the
    boilerplate header/footer lines (site nav, "Project Wittenberg" banners, etc.)
  - CCEL HTML pages (Table Talk, Bondage of the Will): real text lives inside
    <div class="book-content"> -- clean hand-formatted text, no OCR noise.
  - archive.org "stream" pages: the actual OCR text lives inside a single <pre> tag,
    wrapped in a Google Books scan disclaimer we strip off, then OCR line-wrap noise
    (double-spaced words from the goog OCR) is normalized. (No longer used as of the
    CCEL switch, but kept for any future archive.org source.)
"""
import re
from pathlib import Path

from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw"
CLEAN_DIR = Path(__file__).resolve().parent.parent / "corpus" / "clean"

# Lines that are Project Wittenberg site chrome, not Luther's text -- dropped if a
# cleaned line matches one of these (case-insensitive, substring match).
NAV_NOISE = [
    "project wittenberg",
    "previous page",
    "next page",
    "to:",
    "return to",
    "table of contents",
    "book of concord",
    "triglot concordia",
    "martin luther",  # only drops standalone byline/nav occurrences, not body text
]


def is_nav_noise(line: str) -> bool:
    stripped = line.strip().lower()
    if not stripped:
        return True
    if len(stripped) < 60 and any(marker in stripped for marker in NAV_NOISE):
        return True
    return False


def clean_wittenberg_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not is_nav_noise(ln)]
    return "\n".join(lines)


def clean_ccel_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("div.book-content")
    if content is None:
        return ""  # e.g. an index/login page slipped in -- contributes nothing
    for tag in content(["script", "style"]):
        tag.decompose()
    text = content.get_text("\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines)


GOOGLE_DISCLAIMER_END_MARKERS = [
    "usage guidelines",
    "about google book search",
]


def clean_archive_org_stream(raw: str) -> str:
    match = re.search(r"<pre[^>]*>(.*?)</pre>", raw, re.S)
    body = match.group(1) if match else raw
    body = BeautifulSoup(body, "lxml").get_text()

    # OCR artifact: words are often double/triple-spaced ("T h i s  i s  a"). Collapse
    # runs of 2+ spaces to a single space *before* disclaimer search, since the
    # disclaimer text itself is subject to the same OCR spacing noise.
    lines = body.split("\n")
    cleaned_lines = [re.sub(r" {2,}", " ", ln).strip() for ln in lines]
    cleaned_lines = [ln for ln in cleaned_lines if ln]
    body = "\n".join(cleaned_lines)

    # Drop the Google Books scan disclaimer block at the top, if present.
    lower = body.lower()
    cut = 0
    for marker in GOOGLE_DISCLAIMER_END_MARKERS:
        idx = lower.find(marker)
        if idx != -1:
            cut = max(cut, idx + len(marker))
    return body[cut:].strip()


def clean_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".txt" and raw.lstrip().startswith("<!DOCTYPE html"):
        return clean_archive_org_stream(raw)
    if path.suffix == ".txt":
        return raw  # plain wittenberg .txt file, already clean
    if 'class="book-content"' in raw or "class='book-content'" in raw:
        return clean_ccel_html(raw)
    return clean_wittenberg_html(raw)


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for work_dir in sorted(RAW_DIR.iterdir()):
        if not work_dir.is_dir():
            continue
        parts = []
        for part_path in sorted(work_dir.iterdir()):
            parts.append(clean_file(part_path))
        combined = "\n\n".join(parts)
        out_path = CLEAN_DIR / f"{work_dir.name}.txt"
        out_path.write_text(combined, encoding="utf-8")
        print(f"{work_dir.name}: {len(combined):,} chars -> {out_path}")


if __name__ == "__main__":
    main()
