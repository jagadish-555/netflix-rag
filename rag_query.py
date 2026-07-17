"""
rag_query.py
────────────
Netflix-style Natural Language → Elasticsearch DSL query generator.

Workflow:
  1. Accept a natural-language query from the user (CLI arg or interactive prompt)
  2. Retrieve top-3 schema rules  from ChromaDB  (field mappings)
  3. Retrieve top-3 vocab  rules  from ChromaDB  (genre controlled vocabulary)
  4. Assemble a strict LLM prompt with the retrieved context
  5. Send prompt to Groq (llama-3.3-70b-versatile) and generate the ES DSL JSON
  6. Execute the DSL query against the live Elasticsearch index
  7. Display the matching results in a formatted table

Prerequisites:
  - Run `python setup_chroma_db.py` once to seed the vector store
  - Set the GROQ_API_KEY environment variable

Usage:
  python rag_query.py "Show me sci-fi movies that made over 100 million"
  python rag_query.py          # interactive prompt mode
"""

import os
import sys
import json

from dotenv import load_dotenv
import chromadb
from groq import Groq
from elasticsearch import Elasticsearch

# Load .env file automatically (works from any directory)
load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CHROMA_PATH       = "./chroma_db"
GROQ_MODEL        = "llama-3.3-70b-versatile"
TOP_K             = 3          # how many rules to retrieve per collection
TOP_K_OVERVIEW    = 3          # how many contextual movie overviews to retrieve
TOP_K_SEMANTIC    = 10         # results returned in --semantic mode
ES_INDEX          = "movies"   # target Elasticsearch index name
ES_HOST           = "http://localhost:9200"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  ChromaDB – connect and retrieve context
# ──────────────────────────────────────────────────────────────────────────────
def retrieve_context(user_query: str) -> tuple[list[str], list[str], list[str]]:
    """
    Query all three ChromaDB collections and return retrieved documents.

    Returns
    -------
    schema_docs   : list[str]  – top-K schema-rule documents
    vocab_docs    : list[str]  – top-K vocab-rule documents
    overview_docs : list[str]  – top-K contextual movie overview documents
                                 (empty list if collection not yet built)
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open ChromaDB at '{CHROMA_PATH}'. "
            "Did you run `python setup_chroma_db.py` first?"
        ) from exc

    # Retrieve schema rules
    try:
        schema_col  = client.get_collection("schema_rules")
        schema_res  = schema_col.query(query_texts=[user_query], n_results=TOP_K)
        schema_docs = schema_res["documents"][0]        # list of strings
    except Exception as exc:
        raise RuntimeError(f"Error querying 'schema_rules' collection: {exc}") from exc

    # Retrieve vocab rules
    try:
        vocab_col   = client.get_collection("vocab_rules")
        vocab_res   = vocab_col.query(query_texts=[user_query], n_results=TOP_K)
        vocab_docs  = vocab_res["documents"][0]         # list of strings
    except Exception as exc:
        raise RuntimeError(f"Error querying 'vocab_rules' collection: {exc}") from exc

    # Retrieve contextual movie overviews (Contextual RAG)
    # Gracefully skip if the collection has not been built yet.
    overview_docs: list[str] = []
    try:
        overview_col  = client.get_collection("movie_overviews")
        overview_res  = overview_col.query(query_texts=[user_query], n_results=TOP_K_OVERVIEW)
        overview_docs = overview_res["documents"][0]    # list of strings
    except Exception:
        # Collection not found — advise user but don't crash
        print(
            "  ⚠️  'movie_overviews' collection not found. "
            "Run `python contextual_overview_index.py` to enable overview-based retrieval."
        )

    return schema_docs, vocab_docs, overview_docs


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Prompt assembly
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(
    user_query: str,
    schema_docs: list[str],
    vocab_docs: list[str],
    overview_docs: list[str],
) -> str:
    """
    Construct a deterministic, zero-shot prompt for the LLM.

    The prompt enforces:
      • Elasticsearch 8.x DSL JSON only
      • No markdown, no prose – raw JSON output
      • Strict adherence to the retrieved schema and vocabulary
      • Contextual movie overview snippets (Contextual RAG) to enrich
        plot-based and semantic queries
    """
    schema_cheat_sheet = "\n".join(f"  {i+1}. {doc}" for i, doc in enumerate(schema_docs))
    vocab_cheat_sheet  = "\n".join(f"  {i+1}. {doc}" for i, doc in enumerate(vocab_docs))

    # ── Contextual overview section (Contextual RAG) ─────────────────────────
    # Each doc is already formatted as:
    #   "<Title> (<Year>) — Genres: <genres>.\n<overview>"
    # We indent it for readability inside the prompt.
    if overview_docs:
        overview_lines = []
        for i, doc in enumerate(overview_docs, 1):
            # Replace newline between context header and plot with " | "
            compact = doc.replace("\n", " | ", 1)
            overview_lines.append(f"  {i}. {compact}")
        overview_section = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "RELEVANT MOVIE PLOTS  (contextual overview RAG — use these to understand\n"
            "  what the user is looking for and craft a precise 'overview' match query)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + "\n".join(overview_lines)
            + "\n"
        )
    else:
        overview_section = ""

    prompt = f"""You are a precise Elasticsearch 8.x query-generation engine for a Netflix-style movie database.

Your ONLY output must be a single, valid Elasticsearch JSON query (DSL). Do NOT include:
  - markdown code fences (no ```json or ```)
  - explanations, comments, or prose
  - any text before or after the JSON object

The target index is: "{ES_INDEX}"
Available fields: title (text), overview (text), genres (text), revenue (integer),
                  budget (integer), release_date (date, format: yyyy-MM-dd)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA CHEAT SHEET  (field mappings from vector store)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schema_cheat_sheet}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALLOWED TERMINOLOGY  (controlled vocabulary – genres must match EXACTLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{vocab_cheat_sheet}

{overview_section}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Use only the field names listed in the schema cheat sheet.
2. For genre filtering, use ONLY the exact vocabulary values shown above.
3. Wrap multiple conditions in a bool query (must / should / must_not / filter).
4. For numeric comparisons use a range query.
5. For genre matching use a match query on the "genres" field.
6. The "size" field MUST be at the TOP LEVEL of the JSON object, NOT inside the "query" key.
7. Output ONLY the raw JSON – no markdown, no prose.
8. If the user describes a movie by its plot, extract key plot terms from the RELEVANT MOVIE
   PLOTS section above and use them in a match query on the "overview" field.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_query}

Generate the Elasticsearch DSL JSON query now:"""

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Groq – generate the query
# ──────────────────────────────────────────────────────────────────────────────
def generate_es_query(prompt: str) -> str:
    """
    Send the assembled prompt to Groq and return the raw LLM response text.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Export it with:  export GROQ_API_KEY='gsk_...'"
        )

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Elasticsearch DSL generator. "
                    "Output ONLY valid JSON. No markdown. No explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,    # deterministic – critical for query generation
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Validation helper  (best-effort JSON check)
# ──────────────────────────────────────────────────────────────────────────────
def validate_json(raw: str) -> dict:
    """
    Attempt to parse the LLM output as JSON.
    Strip accidental markdown fences if present.
    Raises ValueError on invalid JSON.
    """
    # Strip common markdown wrappers the model might slip in despite instructions
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.splitlines()
        # Remove first and last fence lines
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON. Raw output:\n{raw}\n\nError: {exc}"
        ) from exc

    # Post-process: hoist 'size' to top-level if the LLM buried it inside 'query'
    if "size" not in parsed and "query" in parsed and "size" in parsed["query"]:
        parsed["size"] = parsed["query"].pop("size")

    # Ensure a default size exists
    if "size" not in parsed:
        parsed["size"] = 10

    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Execute the DSL query against Elasticsearch
# ──────────────────────────────────────────────────────────────────────────────
def execute_query(dsl: dict) -> list[dict]:
    """
    Run the generated DSL against the live Elasticsearch 'movies' index.

    Returns
    -------
    hits : list[dict]  – each hit's _source document
    """
    try:
        es = Elasticsearch([ES_HOST])
        if not es.ping():
            raise ConnectionError(f"Cannot reach Elasticsearch at {ES_HOST}")
    except Exception as exc:
        raise ConnectionError(f"Elasticsearch connection failed: {exc}") from exc

    try:
        response = es.search(index=ES_INDEX, body=dsl)
    except Exception as exc:
        raise RuntimeError(f"Elasticsearch query failed: {exc}") from exc

    hits      = response["hits"]["hits"]
    total     = response["hits"]["total"]["value"]
    return hits, total


def display_results(hits: list[dict], total: int) -> None:
    """
    Print a clean, readable table of the Elasticsearch results.
    """
    if not hits:
        print("\n⚠️   No results found. Try a different query.")
        return

    # Column widths
    W_NUM   =  3
    W_TITLE = 38
    W_GENRE = 20
    W_REV   = 14
    W_BUD   = 14
    W_YEAR  =  6

    DIVIDER = (
        f"+{'─'*(W_NUM+2)}+{'─'*(W_TITLE+2)}+{'─'*(W_GENRE+2)}"
        f"+{'─'*(W_REV+2)}+{'─'*(W_BUD+2)}+{'─'*(W_YEAR+2)}+"
    )
    HEADER = (
        f"| {'#':>{W_NUM}} "
        f"| {'Title':<{W_TITLE}} "
        f"| {'Genres':<{W_GENRE}} "
        f"| {'Revenue (USD)':>{W_REV}} "
        f"| {'Budget (USD)':>{W_BUD}} "
        f"| {'Year':>{W_YEAR}} |"
    )

    print(f"\n🎬  Found {total} total match(es) — showing {len(hits)}:\n")
    print(DIVIDER)
    print(HEADER)
    print(DIVIDER)

    for i, hit in enumerate(hits, 1):
        src     = hit["_source"]
        title   = str(src.get("title",   "—"))[:W_TITLE]
        genres  = str(src.get("genres",  "—"))[:W_GENRE]
        revenue = src.get("revenue", 0) or 0
        budget  = src.get("budget",  0) or 0
        date    = str(src.get("release_date", "—"))[:4]   # just year

        rev_str = f"${revenue:,}"  if revenue else "N/A"
        bud_str = f"${budget:,}"   if budget  else "N/A"

        print(
            f"| {i:>{W_NUM}} "
            f"| {title:<{W_TITLE}} "
            f"| {genres:<{W_GENRE}} "
            f"| {rev_str:>{W_REV}} "
            f"| {bud_str:>{W_BUD}} "
            f"| {date:>{W_YEAR}} |"
        )

    print(DIVIDER)


# ──────────────────────────────────────────────────────────────────────────────
# 6a. Semantic search — ChromaDB ranks by plot similarity, ES enriches with data
# ──────────────────────────────────────────────────────────────────────────────
def semantic_search(user_query: str, n: int = TOP_K_SEMANTIC) -> tuple[list[str], list[dict]]:
    """
    Query the 'movie_overviews' ChromaDB collection directly.
    ChromaDB vector similarity determines the ranking order.
    The returned movie_ids are then used to fetch full data from ES.

    Returns
    -------
    docs      : list[str]   – contextualised overview strings (in ranked order)
    metadatas : list[dict]  – {movie_id, title} per result   (in ranked order)
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col    = client.get_collection("movie_overviews")
    except Exception as exc:
        raise RuntimeError(
            "'movie_overviews' collection not found. "
            "Run `python contextual_overview_index.py` first."
        ) from exc

    res       = col.query(query_texts=[user_query], n_results=n)
    docs      = res["documents"][0]
    metadatas = res["metadatas"][0]
    return docs, metadatas


def fetch_by_ids(movie_ids: list[str]) -> tuple[dict, dict[str, dict]]:
    """
    Fetch full movie documents from Elasticsearch by their numeric IDs.

    Returns
    -------
    dsl     : dict            – the ES query that was executed (for display)
    es_data : dict[str, dict] – lookup keyed by movie_id string
    """
    try:
        es = Elasticsearch([ES_HOST])
        if not es.ping():
            raise ConnectionError(f"Cannot reach Elasticsearch at {ES_HOST}")
    except Exception as exc:
        raise ConnectionError(f"Elasticsearch connection failed: {exc}") from exc

    # terms query on the 'id' field — fetches all matching docs in one request
    dsl = {
        "size": len(movie_ids),
        "query": {
            "terms": {"id": [int(mid) for mid in movie_ids]}
        },
    }
    try:
        response = es.search(index=ES_INDEX, body=dsl)
    except Exception as exc:
        raise RuntimeError(f"Elasticsearch fetch failed: {exc}") from exc

    # Build lookup: movie_id (str) → _source dict
    es_data = {
        str(hit["_source"]["id"]): hit["_source"]
        for hit in response["hits"]["hits"]
    }
    return dsl, es_data


def display_semantic_results(
    docs: list[str],
    metadatas: list[dict],
    es_data: dict[str, dict],
    query: str,
) -> None:
    """
    Print semantic results ranked by ChromaDB vector similarity,
    enriched with revenue / budget / year from Elasticsearch.
    """
    if not docs:
        print("\n⚠️   No results found. Try a different query.")
        return

    W_NUM   =  3
    W_TITLE = 38
    W_GENRE = 20
    W_REV   = 14
    W_BUD   = 14
    W_YEAR  =  6

    DIVIDER = (
        f"+{'─'*(W_NUM+2)}+{'─'*(W_TITLE+2)}+{'─'*(W_GENRE+2)}"
        f"+{'─'*(W_REV+2)}+{'─'*(W_BUD+2)}+{'─'*(W_YEAR+2)}+"
    )
    HEADER = (
        f"| {'#':>{W_NUM}} "
        f"| {'Title':<{W_TITLE}} "
        f"| {'Genres':<{W_GENRE}} "
        f"| {'Revenue (USD)':>{W_REV}} "
        f"| {'Budget (USD)':>{W_BUD}} "
        f"| {'Year':>{W_YEAR}} |"
    )

    print(f"\n🔎  Semantic search results for: '{query}'")
    print(f"    Ranked by plot similarity (ChromaDB) · enriched by Elasticsearch\n")
    print(DIVIDER)
    print(HEADER)
    print(DIVIDER)

    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        movie_id  = meta.get("movie_id", "")
        src       = es_data.get(movie_id, {})

        # Fall back to ChromaDB context header if ES data is missing
        lines     = doc.split("\n", 1)
        header    = lines[0]   # "Title (Year) — Genres: ..."

        if src:
            title   = str(src.get("title",   meta.get("title", "—")))[:W_TITLE]
            genres  = str(src.get("genres",  "—"))[:W_GENRE]
            revenue = src.get("revenue", 0) or 0
            budget  = src.get("budget",  0) or 0
            year    = str(src.get("release_date", "—"))[:4]
        else:
            # ES not reachable — parse from ChromaDB context header
            try:
                title_year, genres_part = header.split(" — Genres: ", 1)
                genres  = genres_part.rstrip(".")[:W_GENRE]
                parts   = title_year.rsplit(" (", 1)
                title   = parts[0][:W_TITLE]
                year    = parts[1].rstrip(")") if len(parts) > 1 else "—"
            except (ValueError, IndexError):
                title, genres, year = meta.get("title", "—")[:W_TITLE], "—", "—"
            revenue = budget = 0

        rev_str = f"${revenue:,}" if revenue else "N/A"
        bud_str = f"${budget:,}"  if budget  else "N/A"

        print(
            f"| {i:>{W_NUM}} "
            f"| {title:<{W_TITLE}} "
            f"| {genres:<{W_GENRE}} "
            f"| {rev_str:>{W_REV}} "
            f"| {bud_str:>{W_BUD}} "
            f"| {year:>{W_YEAR}} |"
        )

    print(DIVIDER)
    print("\n💡  Tip: run without --semantic to use structured filters (genre, budget, year).")


def main() -> None:
    # ── Parse --semantic flag ─────────────────────────────────────────────────
    args       = sys.argv[1:]
    semantic   = "--semantic" in args
    query_args = [a for a in args if a != "--semantic"]

    # ── Obtain user query ─────────────────────────────────────────────────────
    if query_args:
        user_query = " ".join(query_args)
    else:
        mode_label = "Semantic Search" if semantic else "RAG Query Generator"
        print(f"🎬  Netflix {mode_label}")
        if semantic:
            print("    Mode: direct vector search on movie overviews (no LLM / ES)")
        print("─" * 50)
        user_query = input("Enter your movie query: ").strip()
        if not user_query:
            print("❌  No query provided. Exiting.")
            sys.exit(1)

    print(f"\n📝  User query  : {user_query}")
    print("─" * 60)

    # ══════════════════════════════════════════════════════════════════════════
    # SEMANTIC MODE — bypass Groq + Elasticsearch entirely
    # Usage:  python rag_query.py --semantic "movie about a hero fighting a clown"
    # ══════════════════════════════════════════════════════════════════════════
    if semantic:
        print("🔎  Mode: semantic vector search (ChromaDB ranks · ES enriches) …")
        try:
            docs, metadatas = semantic_search(user_query)
        except RuntimeError as exc:
            print(f"❌  {exc}")
            sys.exit(1)

        # Fetch full movie data from ES using the ranked movie IDs
        movie_ids = [m["movie_id"] for m in metadatas]
        print(f"    Retrieved {len(movie_ids)} movie(s) from ChromaDB, fetching ES data …")
        try:
            es_dsl, es_data = fetch_by_ids(movie_ids)
            print(f"    ✅  ES enrichment: {len(es_data)}/{len(movie_ids)} movies found.")
            # Print the ES query in the same format as standard mode
            print("\n" + "═" * 60)
            print("✅  Elasticsearch DSL Query (semantic id lookup):")
            print("═" * 60)
            print(json.dumps(es_dsl, indent=2))
            print("═" * 60)
        except (ConnectionError, RuntimeError) as exc:
            print(f"    ⚠️  ES unavailable ({exc}) — showing ChromaDB data only.")
            es_data = {}

        display_semantic_results(docs, metadatas, es_data, user_query)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # STANDARD MODE — ChromaDB RAG → Groq LLM → Elasticsearch
    # ══════════════════════════════════════════════════════════════════════════

    # ── Step 1: Retrieve context from ChromaDB ───────────────────────────────
    print("🔍  Retrieving context from ChromaDB …")
    try:
        schema_docs, vocab_docs, overview_docs = retrieve_context(user_query)
    except RuntimeError as exc:
        print(f"❌  ChromaDB error: {exc}")
        sys.exit(1)

    print(f"\n📚  Schema rules retrieved  ({len(schema_docs)}):")
    for doc in schema_docs:
        print(f"    • {doc}")

    print(f"\n📖  Vocab rules retrieved   ({len(vocab_docs)}):")
    for doc in vocab_docs:
        print(f"    • {doc}")

    print(f"\n🎬  Overview docs retrieved ({len(overview_docs)}) [Contextual RAG]:")
    for doc in overview_docs:
        print(f"    • {doc.splitlines()[0]}")

    # ── Step 2: Assemble prompt ──────────────────────────────────────────────
    print("\n🛠   Assembling prompt …")
    prompt = build_prompt(user_query, schema_docs, vocab_docs, overview_docs)

    # ── Step 3: Call Groq ────────────────────────────────────────────────────
    print(f"🤖  Sending to Groq ({GROQ_MODEL}) …")
    try:
        raw_output = generate_es_query(prompt)
    except EnvironmentError as exc:
        print(f"❌  {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"❌  Groq API error: {exc}")
        sys.exit(1)

    # ── Step 4: Validate & pretty-print the DSL ─────────────────────────────
    print("\n" + "═" * 60)
    print("✅  Generated Elasticsearch DSL Query:")
    print("═" * 60)

    try:
        dsl = validate_json(raw_output)
        print(json.dumps(dsl, indent=2))
    except ValueError:
        print("⚠️   Note: output may not be valid JSON – printing raw response:")
        print(raw_output)
        sys.exit(1)

    print("═" * 60)

    # ── Step 5: Execute against Elasticsearch ───────────────────────────────
    print("\n⚡  Executing query against Elasticsearch …")
    try:
        hits, total = execute_query(dsl)
    except ConnectionError as exc:
        print(f"❌  {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"❌  {exc}")
        sys.exit(1)

    # ── Step 6: Display results ──────────────────────────────────────────────
    display_results(hits, total)


if __name__ == "__main__":
    main()
