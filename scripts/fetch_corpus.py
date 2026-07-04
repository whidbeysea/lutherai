"""
Fetch the priority Luther texts into corpus/raw/<work_slug>/.

Sources are public domain: Project Wittenberg (HTML) and Internet Archive plain-text
dumps of pre-1923 English translations. Deliberately excludes the copyrighted 55-volume
American Edition (Luther's Works) and any antisemitic tracts (e.g. "On the Jews and
Their Lies") -- those must never be fetched into this corpus.

Each entry in SOURCES is a work with one or more part URLs, fetched in order and saved
as 00.html, 01.html, ... under corpus/raw/<slug>/ so clean_corpus.py can reassemble them.
"""
import time
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "corpus" / "raw"

HEADERS = {"User-Agent": "luther-bot-corpus-fetch/1.0 (seminary research project)"}

WITTENBERG_BASE = "https://www.projectwittenberg.org/pub/resources/text/wittenberg/"
CCEL_BASE = "https://ccel.org/ccel/luther/"


def _to_roman(n: int) -> str:
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    result = ""
    for v, s in vals:
        while n >= v:
            result += s
            n -= v
    return result


# Table Talk chapters are numbered i..xlv in CCEL's TOC; chapter xlii ("Of the Jews")
# is deliberately excluded -- see the guardrail note on the table_talk source below.
TABLETALK_PARTS = ["i", "ii", "iii", "iv"] + [
    f"v.{_to_roman(n)}" for n in range(1, 46) if n != 42
]

# Bondage of the Will section structure -- full ordered list scraped directly from
# https://ccel.org/ccel/luther/bondage.toc.html (185 parts, front matter + nested
# section numbering i, iii-vi, vi.i, vii.i-v, ... through xvii.ii).
BONDAGE_PARTS = [
    "i", "iii", "iv", "v", "vi", "vi.i", "vii", "vii.i", "vii.ii", "vii.iii", "vii.iv",
    "vii.v", "viii", "viii.i", "viii.ii", "ix", "ix.i", "ix.ii", "ix.iii", "ix.iv",
    "ix.v", "ix.vi", "ix.vii", "ix.viii", "ix.ix", "ix.x", "ix.xi", "ix.xii", "ix.xiii",
    "ix.xiv", "ix.xv", "ix.xvi", "ix.xvii", "ix.xviii", "ix.xix", "x", "x.i", "x.ii",
    "x.iii", "x.iv", "x.v", "x.vi", "x.vii", "x.viii", "x.ix", "x.x", "x.xi", "x.xii",
    "x.xiii", "xi", "xi.i", "xi.ii", "xi.iii", "xi.iv", "xi.v", "xi.vi", "xi.vii",
    "xi.viii", "xi.ix", "xi.x", "xi.xi", "xi.xii", "xi.xiii", "xi.xiv", "xi.xv",
    "xi.xvi", "xi.xvii", "xi.xviii", "xi.xix", "xi.xx", "xi.xxi", "xi.xxii", "xi.xxiii",
    "xi.xxiv", "xi.xxv", "xi.xxvi", "xi.xxvii", "xi.xxviii", "xi.xxix", "xi.xxx",
    "xi.xxxi", "xi.xxxii", "xi.xxxiii", "xi.xxxiv", "xi.xxxv", "xii", "xii.i", "xii.ii",
    "xii.iii", "xii.iv", "xii.v", "xii.vi", "xii.vii", "xii.viii", "xii.ix", "xii.x",
    "xii.xi", "xii.xii", "xii.xiii", "xii.xiv", "xii.xv", "xii.xvi", "xii.xvii",
    "xii.xviii", "xii.xix", "xii.xx", "xii.xxi", "xii.xxii", "xii.xxiii", "xii.xxiv",
    "xii.xxv", "xii.xxvi", "xii.xxvii", "xii.xxviii", "xii.xxix", "xii.xxx", "xii.xxxi",
    "xii.xxxii", "xii.xxxiii", "xii.xxxiv", "xii.xxxv", "xii.xxxvi", "xii.xxxvii",
    "xii.xxxviii", "xii.xxxix", "xii.xl", "xii.xli", "xii.xlii", "xii.xliii",
    "xii.xliv", "xii.xlv", "xii.xlvi", "xii.xlvii", "xii.xlviii", "xii.xlix", "xii.l",
    "xii.li", "xii.lii", "xii.liii", "xii.liv", "xii.lv", "xii.lvi", "xii.lvii",
    "xii.lviii", "xiii", "xiii.i", "xiii.ii", "xiii.iii", "xiii.iv", "xiii.v",
    "xiii.vi", "xiii.vii", "xiii.viii", "xiii.ix", "xiii.x", "xiii.xi", "xiii.xii",
    "xiii.xiii", "xiii.xiv", "xiii.xv", "xiii.xvi", "xiii.xvii", "xiii.xviii",
    "xiii.xix", "xiii.xx", "xiii.xxi", "xiii.xxii", "xiii.xxiii", "xiii.xxiv",
    "xiii.xxv", "xiii.xxvi", "xiii.xxvii", "xiii.xxviii", "xiii.xxix", "xiii.xxx",
    "xiii.xxxi", "xiii.xxxii", "xiv", "xiv.i", "xiv.ii", "xv", "xvi", "xvii", "xvii.i",
    "xvii.ii",
]

SOURCES = {
    "ninety_five_theses": {
        "category": "theology",
        "year": 1517,
        "urls": [WITTENBERG_BASE + "luther/web/ninetyfive.html"],
    },
    "freedom_of_a_christian": {
        "category": "theology",
        "year": 1520,
        "urls": [
            WITTENBERG_BASE + "luther/web/cclib-1.html",
            WITTENBERG_BASE + "luther/web/cclib-2.html",
            WITTENBERG_BASE + "luther/web/cclib-3.html",
        ],
    },
    "address_to_christian_nobility": {
        "category": "political",
        "year": 1520,
        "urls": [
            WITTENBERG_BASE + "luther/web/nblty-01.html",
            WITTENBERG_BASE + "luther/web/nblty-02.html",
            WITTENBERG_BASE + "luther/web/nblty-03.html",
            WITTENBERG_BASE + "luther/web/nblty-04.html",
            WITTENBERG_BASE + "luther/web/nblty-05.html",
            WITTENBERG_BASE + "luther/web/nblty-06.html",
            WITTENBERG_BASE + "luther/web/nblty-07.html",
        ],
    },
    "treatise_on_good_works": {
        "category": "pastoral",
        "year": 1520,
        "urls": [
            WITTENBERG_BASE + f"luther/work-{n}.txt"
            for n in ["01", "02", "02a", "03", "04", "05", "06"]
        ],
    },
    "commentary_on_galatians": {
        "category": "theology",
        "year": 1535,
        "urls": [
            WITTENBERG_BASE + "luther/gal/web/" + name
            for name in [
                "gal0-00.html", "gal1-01.html", "gal1-04.html", "gal1-07.html",
                "gal2-01.html", "gal2-04.html", "gal2-14.html", "gal2-17.html",
                "gal3-01.html", "gal3-10.html", "gal3-20.html",
                "gal4-01.html", "gal4-10.html",
                "gal5-01.html", "gal5-14.html",
                "gal6-01.html",
            ]
        ],
    },
    "large_catechism": {
        "category": "catechetical",
        "year": 1529,
        "urls": [
            WITTENBERG_BASE + f"luther/catechism/web/cat-{n:02d}.html"
            for n in range(1, 16)
        ],
    },
    "small_catechism": {
        "category": "catechetical",
        "year": 1529,
        "urls": [
            WITTENBERG_BASE + "luther/little.book/web/book-1.html",
            WITTENBERG_BASE + "luther/little.book/web/book-2.html",
            WITTENBERG_BASE + "luther/little.book/web/book-3.html",
            WITTENBERG_BASE + "luther/little.book/web/book-4.html",
            WITTENBERG_BASE + "luther/little.book/web/book-5.html",
            WITTENBERG_BASE + "luther/little.book/web/book-6.html",
        ],
    },
    "selected_sermons": {
        "category": "pastoral",
        "year": 1518,
        "urls": [WITTENBERG_BASE + "luther/web/3formsrt.html"],
    },
    # Not hosted on Project Wittenberg -- pulled from Internet Archive plain-text
    # transcriptions of pre-1923 (public domain) English editions.
    "heidelberg_disputation": {
        "category": "theology",
        "year": 1518,
        "urls": ["https://www.catchpenny.org/heidel.html"],
    },
    # Switched from Internet Archive Google Books OCR scans (heavy OCR noise, e.g.
    # "OfFree-wiU") to CCEL's hand-formatted HTML edition -- clean text, real quotes,
    # no scan artifacts. CCEL uses a per-page <div class="book-content"> container.
    "bondage_of_the_will": {
        "category": "theology",
        "year": 1525,
        "source_type": "ccel",
        "urls": [
            CCEL_BASE + f"bondage/bondage.{part}.html" for part in BONDAGE_PARTS
        ],
    },
    "table_talk": {
        "category": "pastoral",
        "year": 1566,  # compiled posthumously from Luther's conversations
        "source_type": "ccel",
        # Excludes CCEL's "Of the Jews" chapter (tabletalk.v.xlii) -- antisemitic
        # material is never ingested into this corpus, per project guardrail policy.
        "urls": [
            CCEL_BASE + f"tabletalk/tabletalk.{part}.html" for part in TABLETALK_PARTS
        ],
    },
}


def fetch_url(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"    FAILED: {url} ({e})")
        return None


def fetch_work(slug: str, spec: dict) -> None:
    out_dir = RAW_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = spec["urls"]
    ext = "txt" if urls[0].endswith(".txt") else "html"

    print(f"Fetching {slug} ({len(urls)} part(s))...")
    fetched_any = False
    for i, url in enumerate(urls):
        out_path = out_dir / f"{i:02d}.{ext}"
        if out_path.exists():
            fetched_any = True
            continue
        content = fetch_url(url)
        if content is None and "fallback_urls" in spec:
            for fb in spec["fallback_urls"]:
                content = fetch_url(fb)
                if content is not None:
                    break
        if content is not None:
            out_path.write_text(content, encoding="utf-8", errors="replace")
            fetched_any = True
            time.sleep(1)  # be polite to free public-domain hosts
        else:
            print(f"    Could not fetch part {i} of {slug} -- skipping")

    if not fetched_any:
        print(f"    WARNING: no parts fetched for {slug}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for slug, spec in SOURCES.items():
        fetch_work(slug, spec)
    print(f"\nDone. {len(SOURCES)} works targeted -> {RAW_DIR}")


if __name__ == "__main__":
    main()
