"""
setup_chroma_db.py
──────────────────
Dynamically builds the ChromaDB vector store from REAL data sources:

  schema_rules  ← introspected live from the Elasticsearch 'movies' index mapping
  vocab_rules   ← extracted from all unique genre values in subgraphs/content.db

No rules are hardcoded. Every document in ChromaDB reflects what actually
exists in the database and index.

Run once (or whenever the schema / vocabulary changes):
    python setup_chroma_db.py
"""

import os
import sqlite3
import chromadb
from elasticsearch import Elasticsearch

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
CHROMA_PATH  = "./chroma_db"
ES_HOST      = "http://localhost:9200"
ES_INDEX     = "movies"
CONTENT_DB   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subgraphs", "content.db")

# ──────────────────────────────────────────────────────────────────────────────
# Helper: natural-language synonyms for each ES field type
#   These are GENERIC DSL hints that apply to any field of that type.
#   The field-specific docs are auto-generated below.
# ──────────────────────────────────────────────────────────────────────────────
OPERATOR_RULES = [
    {
        "id": "op_01",
        "doc": "more than, greater than, over, at least, above, minimum → ES: range query with gte or gt operator",
        "meta": {"field": "range", "type": "operator"},
    },
    {
        "id": "op_02",
        "doc": "less than, under, below, at most, maximum, fewer than → ES: range query with lte or lt operator",
        "meta": {"field": "range", "type": "operator"},
    },
    {
        "id": "op_03",
        "doc": "and, both, also, as well as, combined, together → ES: bool query with must clauses",
        "meta": {"field": "bool.must", "type": "operator"},
    },
    {
        "id": "op_04",
        "doc": "or, either, alternatively, any of → ES: bool query with should clauses",
        "meta": {"field": "bool.should", "type": "operator"},
    },
    {
        "id": "op_05",
        "doc": "not, exclude, without, except, no → ES: bool query with must_not clauses",
        "meta": {"field": "bool.must_not", "type": "operator"},
    },
    {
        "id": "op_06",
        "doc": "top, highest, most, best, sorted by descending, ranked by → ES: sort descending",
        "meta": {"field": "sort", "type": "operator"},
    },
    {
        "id": "op_07",
        "doc": "lowest, least, cheapest, smallest, sorted by ascending → ES: sort ascending",
        "meta": {"field": "sort", "type": "operator"},
    },
    {
        "id": "op_08",
        "doc": "limit, show me N, top N, first N, how many → ES: size parameter",
        "meta": {"field": "size", "type": "operator"},
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# ES type → human-readable synonyms used when building schema documents
# ──────────────────────────────────────────────────────────────────────────────
FIELD_NL_SYNONYMS = {
    "budget":       "budget, production cost, how much it cost, cost to make, film budget",
    "genres":       "genre, category, type of movie, film type, kind of film",
    "id":           "movie id, identifier, film id, database id",
    "overview":     "description, plot, synopsis, about, story, what is it about, summary",
    "release_date": "year, release year, when was it released, year made, decade, release date",
    "revenue":      "money made, earnings, box office, gross, income, profit, total revenue",
    "title":        "title, name, movie name, film title, called, named",
}

FIELD_QUERY_HINTS = {
    "budget":       "use range query for numeric comparisons; sort for ranking",
    "genres":       "use match query; must match EXACTLY the controlled vocabulary below",
    "id":           "use term query for exact match",
    "overview":     "use match query for full-text search",
    "release_date": "use range query with gte/lte on YYYY-MM-DD; extract year with date_histogram",
    "revenue":      "use range query for numeric comparisons; sort for ranking",
    "title":        "use match for full-text; use title.keyword for exact/sort",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Pull schema from live Elasticsearch mapping
# ──────────────────────────────────────────────────────────────────────────────
def fetch_schema_rules_from_es() -> list[dict]:
    """
    Connect to the live ES index, read the field mapping, and return
    a list of ChromaDB-ready documents — one per field + operator rules.
    """
    print(f"  Connecting to Elasticsearch at {ES_HOST} …")
    es = Elasticsearch([ES_HOST])

    if not es.ping():
        raise ConnectionError(f"Cannot reach Elasticsearch at {ES_HOST}")

    mapping   = es.indices.get_mapping(index=ES_INDEX)
    properties = mapping[ES_INDEX]["mappings"]["properties"]

    print(f"  Found {len(properties)} field(s) in index '{ES_INDEX}': {list(properties.keys())}")

    records = []
    for i, (field_name, field_def) in enumerate(sorted(properties.items()), start=1):
        es_type    = field_def.get("type", "object")
        has_kw     = "keyword" in field_def.get("fields", {})
        nl_synonyms = FIELD_NL_SYNONYMS.get(field_name, field_name)
        query_hint  = FIELD_QUERY_HINTS.get(field_name, f"use appropriate ES query for type '{es_type}'")

        # Build the natural-language document for this field
        doc = (
            f"{nl_synonyms} "
            f"→ ES field: {field_name} (type: {es_type}"
            + (", keyword sub-field: " + field_name + ".keyword available for exact/sort" if has_kw else "")
            + f"); {query_hint}"
        )

        records.append({
            "id":   f"schema_{field_name}",
            "doc":  doc,
            "meta": {
                "field":    field_name,
                "es_type":  es_type,
                "has_keyword": str(has_kw),
            },
        })

    # Add the generic operator / DSL-pattern rules
    records.extend(OPERATOR_RULES)
    print(f"  ✔  {len(records)} schema rule documents built "
          f"({len(properties)} fields + {len(OPERATOR_RULES)} operator rules).")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Pull vocabulary from the real SQLite content database
# ──────────────────────────────────────────────────────────────────────────────
def fetch_vocab_rules_from_db() -> list[dict]:
    """
    Read all UNIQUE genre values from subgraphs/content.db → movies.genre
    (comma-separated in the DB) and return ChromaDB-ready documents.
    """
    print(f"  Connecting to SQLite: {CONTENT_DB} …")
    if not os.path.exists(CONTENT_DB):
        raise FileNotFoundError(f"Content DB not found: {CONTENT_DB}. Run build_db.py first.")

    conn = sqlite3.connect(CONTENT_DB)
    rows = conn.execute(
        "SELECT genre FROM movies WHERE genre IS NOT NULL AND genre != ''"
    ).fetchall()
    conn.close()

    # Each row's genre is a comma-separated string like "Action, Adventure"
    genres: set[str] = set()
    for (raw,) in rows:
        for g in raw.split(","):
            g = g.strip()
            if g:
                genres.add(g)

    print(f"  Found {len(genres)} unique genre(s) in the database.")

    records = []
    for i, genre in enumerate(sorted(genres), start=1):
        # Build synonyms automatically from the genre name itself
        lower = genre.lower()
        doc = (
            f"{lower}, {genre}"
            # Add common alias patterns
            + (", sci-fi, scifi, space, futuristic" if "science fiction" in lower else "")
            + (", animated, cartoon" if "animation" in lower else "")
            + (", films" if lower not in ("foreign", "tv movie") else "")
            + f" → Use EXACTLY: {genre}"
        )
        records.append({
            "id":   f"vocab_{i:02d}_{genre.replace(' ', '_')}",
            "doc":  doc,
            "meta": {"field": "genres", "value": genre},
        })

    print(f"  ✔  {len(records)} vocab rule documents built.")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Upsert into ChromaDB
# ──────────────────────────────────────────────────────────────────────────────
def upsert_collection(client: chromadb.PersistentClient, name: str, records: list[dict]) -> None:
    col = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    col.upsert(
        ids=[r["id"] for r in records],
        documents=[r["doc"] for r in records],
        metadatas=[r["meta"] for r in records],
    )
    print(f"  ✅  '{name}' → {col.count()} document(s) in ChromaDB.")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🔧  Building ChromaDB from REAL data sources")
    print("=" * 60)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # ── Schema rules from ES mapping ─────────────────────────────────────────
    print("\n📡  [1/2] Fetching schema rules from Elasticsearch …")
    try:
        schema_records = fetch_schema_rules_from_es()
    except (ConnectionError, Exception) as exc:
        print(f"  ❌  {exc}")
        raise SystemExit(1)

    upsert_collection(chroma_client, "schema_rules", schema_records)

    # ── Vocab rules from SQLite ───────────────────────────────────────────────
    print("\n📂  [2/2] Fetching vocabulary rules from content database …")
    try:
        vocab_records = fetch_vocab_rules_from_db()
    except (FileNotFoundError, Exception) as exc:
        print(f"  ❌  {exc}")
        raise SystemExit(1)

    upsert_collection(chroma_client, "vocab_rules", vocab_records)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✨  ChromaDB is ready!")
    print(f"   schema_rules : {len(schema_records)} docs  (ES fields + DSL operators)")
    print(f"   vocab_rules  : {len(vocab_records)} docs  (unique genres from DB)")
    print("=" * 60)
    print("\nRun `python rag_query.py` to start querying.")
