"""
build_overview_collections.py
──────────────────────────────────────────────────────────────────────────────
Creates TWO ChromaDB collections from the movie content database:

  Collection A ── multi_movie_chunks
      Groups movies into windows of 5 and merges their contextualised
      overviews into ONE massive string → embedded as a single vector.

  Collection B ── single_movie_chunks
      Each movie's contextualised overview is embedded as its own vector

Run:
    python build_overview_collections.py
"""

import os
import sqlite3

import chromadb

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_HERE, "chroma_db")
CONTENT_DB  = os.path.join(_HERE, "subgraphs", "content.db")

COLLECTION_A = "multi_movie_chunks"   # 5 movies merged → 1 vector
COLLECTION_B = "single_movie_chunks"  # 1 movie         → 1 vector

CHUNK_SIZE   = 5                      # movies per multi-chunk


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load all movies from SQLite
# ──────────────────────────────────────────────────────────────────────────────
def load_movies() -> list[dict]:
    """
    Fetch every movie that has a non-empty overview from content.db.
    Returns a list of dicts with keys: id, title, release_year, genre, overview.
    """
    if not os.path.exists(CONTENT_DB):
        raise FileNotFoundError(
            f"Content DB not found: {CONTENT_DB}\n"
            "Run `python build_db.py` first."
        )

    print(f"  Connecting to SQLite: {CONTENT_DB} ...")
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

    print(f"  Loaded {len(rows)} movie(s) with a non-empty overview.")
    return [
        {
            "id":           movie_id,
            "title":        (title or "").strip(),
            "release_year": release_year or 0,
            "genre":        (genre or "").strip(),
            "overview":     (overview or "").strip(),
        }
        for movie_id, title, release_year, genre, overview in rows
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Build contextualised overview string for a single movie
#     (same format already used in movie_overviews / contextual_overview_index.py)
# ──────────────────────────────────────────────────────────────────────────────
def contextualise(movie: dict) -> str:
    """
    Returns the standard contextual overview string:

        <Title> (<Year>) - Genres: <genre>.
        <overview text>
    """
    year_str  = str(movie["release_year"]) if movie["release_year"] else "Unknown year"
    genre_str = movie["genre"]  or "Unknown genre"
    title_str = movie["title"] or "Unknown title"

    header = f"{title_str} ({year_str}) - Genres: {genre_str}."
    return f"{header}\n{movie['overview']}"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Collection B records  (single_movie_chunks)
#     One contextualised document per movie - identical format to the existing
#     movie_overviews collection.
# ──────────────────────────────────────────────────────────────────────────────
def build_single_chunks(movies: list[dict]) -> list[dict]:
    """
    Returns one ChromaDB record per movie:
        id       -> "overview_<movie_id>"
        doc      -> contextualised overview string
        meta     -> {movie_id, title, release_year, genre}
    """
    records = []
    for m in movies:
        records.append({
            "id":  f"overview_{m['id']}",
            "doc": contextualise(m),
            "meta": {
                "movie_id":     str(m["id"]),
                "title":        m["title"],
                "release_year": str(m["release_year"]),
                "genre":        m["genre"],
            },
        })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Collection A records  (multi_movie_chunks)
#     Every 5 consecutive movies are merged into one massive string -> 1 vector.
# ──────────────────────────────────────────────────────────────────────────────
def build_multi_chunks(movies: list[dict], chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """
    Slides a window of `chunk_size` movies across the list and merges their
    contextualised overviews into a single document separated by dividers.

    id   -> "multi_chunk_<start_idx>_<end_idx>"
    doc  -> all chunk_size contextualised overviews joined by a divider
    meta -> comma-separated movie_ids, titles, and the chunk index
    """
    records  = []
    total    = len(movies)

    for start in range(0, total, chunk_size):
        batch = movies[start : start + chunk_size]
        end   = start + len(batch) - 1

        divider = "\n" + "-" * 60 + "\n"
        merged_doc = divider.join(contextualise(m) for m in batch)

        # Annotate the chunk with a header listing all titles
        titles_line = " | ".join(
            f"{m['title']} ({m['release_year'] or '?'})" for m in batch
        )
        full_doc = f"[CHUNK {start//chunk_size + 1}] Movies: {titles_line}\n\n{merged_doc}"

        records.append({
            "id":  f"multi_chunk_{start}_{end}",
            "doc": full_doc,
            "meta": {
                "chunk_index":  str(start // chunk_size + 1),
                "movie_ids":    ",".join(str(m["id"]) for m in batch),
                "titles":       " | ".join(m["title"] for m in batch),
                "start_idx":    str(start),
                "end_idx":      str(end),
                "chunk_size":   str(len(batch)),
            },
        })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Upsert into ChromaDB
# ──────────────────────────────────────────────────────────────────────────────
def upsert_collection(
    client: chromadb.PersistentClient,
    name:   str,
    records: list[dict],
) -> None:
    col = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    # Upsert in batches of 500 to avoid memory spikes on large datasets
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        col.upsert(
            ids       =[r["id"]   for r in batch],
            documents =[r["doc"]  for r in batch],
            metadatas =[r["meta"] for r in batch],
        )
    print(f"  [OK]  '{name}' -> {col.count()} document(s) stored in ChromaDB.")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Pretty-print samples
# ──────────────────────────────────────────────────────────────────────────────
def print_samples(collection_name: str, records: list[dict], n: int = 2) -> None:
    bar = "-" * 60
    print(f"\n{bar}")
    print(f"  Sample(s) from  [{collection_name}]  (first {min(n, len(records))})")
    print(bar)
    for rec in records[:n]:
        print(f"\n  ID  : {rec['id']}")
        print(f"  META: {rec['meta']}")
        preview = rec["doc"][:300].replace("\n", " | ")
        print(f"  DOC : {preview}...")
    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Building ChromaDB Overview Collections")
    print("=" * 60)

    # Load source data
    print("\n[1/4] Loading movies from SQLite ...")
    try:
        movies = load_movies()
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        raise SystemExit(1)

    # Build records
    print(f"\n[2/4] Building Collection B records  (single_movie_chunks) ...")
    single_records = build_single_chunks(movies)
    print(f"      -> {len(single_records)} single-movie documents")

    print(f"\n[3/4] Building Collection A records  (multi_movie_chunks, window={CHUNK_SIZE}) ...")
    multi_records = build_multi_chunks(movies, chunk_size=CHUNK_SIZE)
    print(f"      -> {len(multi_records)} multi-movie chunk documents")

    # Samples
    print_samples(COLLECTION_B, single_records, n=2)
    print_samples(COLLECTION_A, multi_records,  n=2)

    # Upsert into ChromaDB
    print("\n[4/4] Upserting into ChromaDB ...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    print(f"\n  [A] {COLLECTION_A}  ...")
    upsert_collection(chroma_client, COLLECTION_A, multi_records)

    print(f"\n  [B] {COLLECTION_B}  ...")
    upsert_collection(chroma_client, COLLECTION_B, single_records)

    # Summary
    print("\n" + "=" * 60)
    print("Done!")
    print(f"   {COLLECTION_A:<25} {len(multi_records):>6} docs  (5 movies -> 1 vector)")
    print(f"   {COLLECTION_B:<25} {len(single_records):>6} docs  (1 movie  -> 1 vector)")
    print("=" * 60)
    print("\nCollections are ready in ChromaDB.")
