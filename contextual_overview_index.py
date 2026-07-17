import os
import sqlite3

import chromadb

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH  = os.path.join(_HERE, "chroma_db")
CONTENT_DB   = os.path.join(_HERE, "subgraphs", "content.db")
COLLECTION   = "movie_overviews"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build context-enriched overview strings from SQLite
# ──────────────────────────────────────────────────────────────────────────────
def build_contextual_overviews() -> list[dict]:
    """
    Read every movie row from content.db and construct a contextualised
    overview document:

        <Title> (<Year>) — Genres: <genre>.
        <overview text>

    Returns a list of dicts with keys: id, doc, meta.
    Rows with empty overviews are skipped.
    """
    if not os.path.exists(CONTENT_DB):
        raise FileNotFoundError(
            f"Content DB not found: {CONTENT_DB}\n"
            "Run `python build_db.py` first."
        )

    print(f"  Connecting to SQLite: {CONTENT_DB} …")
    conn = sqlite3.connect(CONTENT_DB)
    rows = conn.execute(
        """
        SELECT id, title, release_year, genre, overview
        FROM   movies
        WHERE  overview IS NOT NULL AND TRIM(overview) != ''
        ORDER  BY id
        """
    ).fetchall()
    conn.close()

    print(f"  Found {len(rows)} movie(s) with a non-empty overview.")

    records = []
    for movie_id, title, release_year, genre, overview in rows:
        # ── Build deterministic context prefix from structured metadata ─────
        year_str  = str(release_year) if release_year else "Unknown year"
        genre_str = genre.strip()     if genre        else "Unknown genre"
        title_str = title.strip()     if title        else "Unknown title"

        context = f"{title_str} ({year_str}) — Genres: {genre_str}."

        # Contextualised document = context header + blank line + overview
        contextual_doc = f"{context}\n{overview.strip()}"

        records.append({
            "id":   f"overview_{movie_id}",
            "doc":  contextual_doc,
            "meta": {
                "movie_id": str(movie_id),
                "title":    title_str,
            },
        })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Upsert into ChromaDB
# ──────────────────────────────────────────────────────────────────────────────
def upsert_overviews(records: list[dict]) -> None:
    """
    Upsert all contextualised overview documents into the ChromaDB
    'movie_overviews' collection (cosine similarity space).
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB upsert in one batch (handles duplicates gracefully)
    col.upsert(
        ids       =[r["id"]   for r in records],
        documents =[r["doc"]  for r in records],
        metadatas =[r["meta"] for r in records],
    )

    print(f"  ✅  '{COLLECTION}' → {col.count()} document(s) in ChromaDB.")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Quick sanity-check: print a few examples
# ──────────────────────────────────────────────────────────────────────────────
def print_samples(records: list[dict], n: int = 3) -> None:
    print(f"\n{'─'*60}")
    print(f"Sample contextualised overviews (first {n}):")
    print(f"{'─'*60}")
    for rec in records[:n]:
        print(f"\n[{rec['id']}]")
        # Show first 200 chars of the contextualised doc
        preview = rec["doc"][:220].replace("\n", " | ")
        print(f"  {preview}…")
    print(f"{'─'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🎬  Contextual Overview Indexer")
    print("=" * 60)

    # ── Step 1: Build contextualised overview records ────────────────────────
    print("\n📂  [1/2] Building contextual overviews from SQLite …")
    try:
        records = build_contextual_overviews()
    except FileNotFoundError as exc:
        print(f"  ❌  {exc}")
        raise SystemExit(1)

    print_samples(records)

    # ── Step 2: Upsert into ChromaDB ─────────────────────────────────────────
    print("📦  [2/2] Upserting into ChromaDB …")
    upsert_overviews(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✨  Done!")
    print(f"   Collection : {COLLECTION}")
    print(f"   Documents  : {len(records)}")
    print(
        "\nContext format:\n"
        "   <Title> (<Year>) — Genres: <genres>.\n"
        "   <original overview text>"
    )
    print("=" * 60)
    print("\nRun `python rag_query.py` to start querying with overview context.")
