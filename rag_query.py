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

# Overview collection options (set via --chunks flag at runtime)
CHUNKS_SINGLE     = "single_movie_chunks"   # 1 movie  → 1 vector (default)
CHUNKS_MULTI      = "multi_movie_chunks"    # 5 movies → 1 vector
CHUNKS_DEFAULT    = CHUNKS_SINGLE

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Shared Utilities
# ──────────────────────────────────────────────────────────────────────────────
def get_groq_client() -> Groq:
    """Returns an authenticated Groq client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Export it with:  export GROQ_API_KEY='gsk_...'"
        )
    return Groq(api_key=api_key)


def get_es_client() -> Elasticsearch:
    """Returns an authenticated Elasticsearch client."""
    try:
        es = Elasticsearch([ES_HOST])
        if not es.ping():
            raise ConnectionError(f"Cannot reach Elasticsearch at {ES_HOST}")
        return es
    except Exception as exc:
        raise ConnectionError(f"Elasticsearch connection failed: {exc}") from exc


def extract_json_from_markdown(raw: str) -> str:
    """Strips markdown code fences from LLM output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return cleaned.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 1.  ChromaDB – connect and retrieve context
# ──────────────────────────────────────────────────────────────────────────────
def retrieve_context(
    user_query: str,
    collection_name: str = CHUNKS_DEFAULT,
) -> tuple[list[str], list[str], list[str]]:
    """
    Query all three ChromaDB collections and return retrieved documents.

    Parameters
    ----------
    user_query      : the natural-language query string
    collection_name : which overview collection to use
                      CHUNKS_SINGLE  → 'single_movie_chunks'  (1 movie  / vector)
                      CHUNKS_MULTI   → 'multi_movie_chunks'   (5 movies / vector)

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
    # Gracefully skip if the chosen collection has not been built yet.
    overview_docs: list[str] = []
    try:
        overview_col  = client.get_collection(collection_name)
        overview_res  = overview_col.query(query_texts=[user_query], n_results=TOP_K_OVERVIEW)
        overview_docs = overview_res["documents"][0]    # list of strings
    except Exception:
        # Collection not found — advise user but don't crash
        print(
            f"  ⚠️  '{collection_name}' collection not found. "
            "Run `python build_overview_collections.py` to enable overview-based retrieval."
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

    if overview_docs:
        overview_lines = []
        for i, doc in enumerate(overview_docs, 1):
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
    client = get_groq_client()
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
# 3b. Groq – generate semantic filter (for --semantic mode)
# ──────────────────────────────────────────────────────────────────────────────
def build_semantic_filter_prompt(user_query: str, candidates: list[dict]) -> str:
    """
    Builds a prompt that asks the LLM to filter a list of candidate movies based on
    the user's query, outputting only a JSON list of matching movie IDs.
    """
    candidates_json = json.dumps(candidates, indent=2)
    return f"""You are a movie recommendation filtering engine.

The user asked: "{user_query}"

Below is a JSON list of candidate movies retrieved from a vector search. Some might be irrelevant due to "context bleed" from chunking, or they might not meet the specific filters the user asked for (like budget, revenue, or specific characters).

Your task is to analyze each candidate and pick ONLY the ones that truly match the user's request.

Candidates:
{candidates_json}

Output ONLY a raw JSON array of the matching integer IDs. Do not include markdown formatting, explanations, or any text other than the JSON array.
If none match, output an empty array: []

Example output:
[12, 45, 120]
"""

def generate_semantic_filter(prompt: str) -> list[int]:
    """
    Calls Groq with the filter prompt and parses the JSON array response.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You output strictly raw JSON arrays of integers. No markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()
    raw = extract_json_from_markdown(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON array. Raw output:\n{raw}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Validation helper  (best-effort JSON check)
# ──────────────────────────────────────────────────────────────────────────────
def validate_json(raw: str) -> dict:
    """
    Attempt to parse the LLM output as JSON.
    Strip accidental markdown fences if present.
    Raises ValueError on invalid JSON.
    """
    cleaned = extract_json_from_markdown(raw)
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
    es = get_es_client()
    try:
        response = es.search(index=ES_INDEX, body=dsl)
    except Exception as exc:
        raise RuntimeError(f"Elasticsearch query failed: {exc}") from exc

    hits      = response["hits"]["hits"]
    total     = response["hits"]["total"]["value"]
    return hits, total


def print_movie_table(movies: list[dict], title: str) -> None:
    """
    Print a clean, readable table of movie results.
    `movies` is a list of dictionaries with keys:
    - title (str)
    - genres (str)
    - revenue (int)
    - budget (int)
    - release_year (str)
    """
    if not movies:
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

    print(f"\n{title}\n")
    print(DIVIDER)
    print(HEADER)
    print(DIVIDER)

    for i, src in enumerate(movies, 1):
        m_title = str(src.get("title", "—"))[:W_TITLE]
        genres  = str(src.get("genres", "—"))[:W_GENRE]
        revenue = src.get("revenue", 0) or 0
        budget  = src.get("budget", 0) or 0
        year    = str(src.get("release_year", "—"))[:W_YEAR]

        rev_str = f"${revenue:,}" if revenue else "N/A"
        bud_str = f"${budget:,}"  if budget  else "N/A"

        print(
            f"| {i:>{W_NUM}} "
            f"| {m_title:<{W_TITLE}} "
            f"| {genres:<{W_GENRE}} "
            f"| {rev_str:>{W_REV}} "
            f"| {bud_str:>{W_BUD}} "
            f"| {year:>{W_YEAR}} |"
        )

    print(DIVIDER)


# ──────────────────────────────────────────────────────────────────────────────
# 6a. Semantic search — ChromaDB ranks by plot similarity, ES enriches with data
# ──────────────────────────────────────────────────────────────────────────────
def semantic_search(
    user_query: str,
    n: int = TOP_K_SEMANTIC,
    collection_name: str = CHUNKS_DEFAULT,
) -> tuple[list[str], list[dict]]:
    """
    Query a ChromaDB overview collection directly.
    ChromaDB vector similarity determines the ranking order.
    The returned movie_ids are then used to fetch full data from ES.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col    = client.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"'{collection_name}' collection not found. "
            "Run `python build_overview_collections.py` first."
        ) from exc

    res       = col.query(query_texts=[user_query], n_results=n)
    docs      = res["documents"][0]
    metadatas = res["metadatas"][0]
    return docs, metadatas


def fetch_by_ids(movie_ids: list[str]) -> tuple[dict, dict[str, dict]]:
    """
    Fetch full movie documents from Elasticsearch by their numeric IDs.
    """
    es = get_es_client()

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

    es_data = {
        str(hit["_source"]["id"]): hit["_source"]
        for hit in response["hits"]["hits"]
    }
    return dsl, es_data


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Routing (Auto-mode)
# ──────────────────────────────────────────────────────────────────────────────
def build_router_prompt(user_query: str) -> str:
    """
    Builds a prompt asking the LLM to classify the query as either
    SEMANTIC (plot/vibe based) or STRUCTURED (budget/revenue/year based).
    """
    return f"""You are a Netflix search query router.

Analyze the following user query: "{user_query}"

Classify it into exactly one of these two categories:
1. "SEMANTIC"
   - Use this if the user is asking about the plot, story, vibes, characters, or concepts.
   - Examples: "a hero fighting a villain in space", "a sad movie about dogs", "find movies with joker as villan"
2. "STRUCTURED"
   - Use this if the user is asking for strict database filters like revenue, budget, release year, or exact genres.
   - Examples: "movies that made over 100 million", "sci-fi movies released after 2010", "budget under 40 million"

Your output MUST be a strict JSON object with a single key "route" whose value is either "SEMANTIC" or "STRUCTURED".
Do not include any other text or markdown fences.

Example output:
{{"route": "SEMANTIC"}}
"""

def route_query(user_query: str) -> str:
    """
    Calls Groq to route the query. Returns 'SEMANTIC' or 'STRUCTURED'.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You output strictly raw JSON objects. No markdown. No explanations.",
            },
            {"role": "user", "content": build_router_prompt(user_query)},
        ],
        temperature=0.0,
        max_tokens=64,
    )

    raw = response.choices[0].message.content.strip()
    raw = extract_json_from_markdown(raw)

    try:
        parsed = json.loads(raw)
        route = parsed.get("route", "STRUCTURED").upper()
        return route if route in ["SEMANTIC", "STRUCTURED"] else "STRUCTURED"
    except json.JSONDecodeError:
        # Fallback to structured if the LLM fails to output JSON
        return "STRUCTURED"


def main() -> None:
    # ── Parse flags ───────────────────────────────────────────────────────────
    args = sys.argv[1:]
    
    force_semantic   = "--semantic" in args
    force_structured = "--dsl" in args
    
    args = [a for a in args if a not in ("--semantic", "--dsl")]

    # --chunks single  (default) or --chunks multi
    collection_name = CHUNKS_DEFAULT
    if "--chunks" in args:
        idx = args.index("--chunks")
        if idx + 1 < len(args):
            choice = args[idx + 1].lower()
            if choice == "multi":
                collection_name = CHUNKS_MULTI
            elif choice == "single":
                collection_name = CHUNKS_SINGLE
            else:
                print(f"❌  Unknown --chunks value '{choice}'. Choose 'single' or 'multi'.")
                sys.exit(1)
            args = args[:idx] + args[idx + 2:]   # remove --chunks <value> from remaining args
        else:
            print("❌  --chunks requires a value: 'single' or 'multi'.")
            sys.exit(1)

    query_args = args   # everything left is the query

    # ── Obtain user query ─────────────────────────────────────────────────────
    if query_args:
        user_query = " ".join(query_args)
    else:
        print("🎬  Netflix Search Engine")
        print("    Modes available:")
        print("      • Auto (Router chooses engine based on query intent)")
        print("      • --semantic (Forced Vector/Plot Search)")
        print("      • --dsl      (Forced Structured DB Search)")
        print("─" * 50)
        user_query = input("Enter your movie query: ").strip()
        if not user_query:
            print("❌  No query provided. Exiting.")
            sys.exit(1)

    print(f"\n📝  User query  : {user_query}")
    print(f"📦  Chunk mode  : {'multi (5 movies/vector)' if collection_name == CHUNKS_MULTI else 'single (1 movie/vector)'}")
    
    # ── Routing Decision ──────────────────────────────────────────────────────
    if force_semantic:
        semantic = True
        print("🚥  Routing     : FORCED Semantic Mode")
    elif force_structured:
        semantic = False
        print("🚥  Routing     : FORCED Structured (DSL) Mode")
    else:
        print("🧠  Analyzing query intent ...")
        route = route_query(user_query)
        semantic = (route == "SEMANTIC")
        print(f"🚥  Routing     : AUTO-DETECTED → {route} Mode")
        
    print("─" * 60)

    # ══════════════════════════════════════════════════════════════════════════
    # SEMANTIC MODE — bypass Groq + Elasticsearch entirely
    # Usage:  python rag_query.py --semantic "movie about a hero fighting a clown"
    # ══════════════════════════════════════════════════════════════════════════
    if semantic:
        print(f"🔎  Mode: semantic vector search · collection: '{collection_name}' …")
        try:
            docs, metadatas = semantic_search(user_query, collection_name=collection_name)
        except RuntimeError as exc:
            print(f"❌  {exc}")
            sys.exit(1)

        # Fetch full movie data from ES using the ranked movie IDs
        movie_ids = []
        for m in metadatas:
            if "movie_ids" in m:
                movie_ids.extend(m["movie_ids"].split(","))
            elif "movie_id" in m:
                movie_ids.append(m["movie_id"])

        print(f"    Retrieved {len(movie_ids)} movie(s) from ChromaDB, fetching ES data …")
        try:
            es_dsl, es_data = fetch_by_ids(movie_ids)
            print(f"    ✅  ES enrichment: {len(es_data)}/{len(movie_ids)} movies found.")
            
            # --- NEW: LLM Filtering Step ---
            print("🤖  Sending candidates to Groq for strict filtering …")
            
            # Prepare candidates for the LLM
            candidates = []
            for doc, meta in zip(docs, metadatas):
                m_ids = meta.get("movie_ids", meta.get("movie_id", "")).split(",")
                for m_id in m_ids:
                    m_id = m_id.strip()
                    if m_id and m_id in es_data:
                        src = es_data[m_id]
                        candidates.append({
                            "id": int(m_id),
                            "title": src.get("title", ""),
                            "genres": src.get("genres", ""),
                            "release_year": src.get("release_date", "")[:4] if src.get("release_date") else "",
                            "revenue": src.get("revenue", 0),
                            "budget": src.get("budget", 0),
                            "overview": src.get("overview", "")
                        })

            if not candidates:
                print("    ⚠️  No valid ES candidates to filter.")
                return

            filter_prompt = build_semantic_filter_prompt(user_query, candidates)
            try:
                filtered_ids = generate_semantic_filter(filter_prompt)
                print(f"    🎯  LLM kept {len(filtered_ids)} movie(s) out of {len(candidates)}.")
                
                movies = []
                for fid in filtered_ids:
                    fid_str = str(fid)
                    if fid_str in es_data:
                        src = es_data[fid_str]
                        movies.append({
                            "title": src.get("title", ""),
                            "genres": src.get("genres", ""),
                            "revenue": src.get("revenue", 0),
                            "budget": src.get("budget", 0),
                            "release_year": src.get("release_date", "")[:4] if src.get("release_date") else ""
                        })

                title = f"🔎  Semantic search results for: '{user_query}'\n    Ranked by plot similarity (ChromaDB) · enriched by Elasticsearch"
                print_movie_table(movies, title=title)

            except Exception as exc:
                print(f"    ❌  LLM filtering failed: {exc}. Displaying unfiltered results.")
                title = f"🔎  Semantic search results for: '{user_query}'\n    Ranked by plot similarity (ChromaDB) · enriched by Elasticsearch"
                print_movie_table(candidates, title=title)

        except (ConnectionError, RuntimeError) as exc:
            print(f"    ❌  ES unavailable ({exc}). Semantic search requires Elasticsearch for metadata.")

        print("\n💡  Tip: run without --semantic to use structured filters (genre, budget, year).")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # STANDARD MODE — ChromaDB RAG → Groq LLM → Elasticsearch
    # ══════════════════════════════════════════════════════════════════════════

    # ── Step 1: Retrieve context from ChromaDB ───────────────────────────────
    print(f"🔍  Retrieving context from ChromaDB (collection: '{collection_name}') …")
    try:
        schema_docs, vocab_docs, overview_docs = retrieve_context(user_query, collection_name)
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
    movies = []
    for hit in hits:
        src = hit["_source"]
        movies.append({
            "title": src.get("title", ""),
            "genres": src.get("genres", ""),
            "revenue": src.get("revenue", 0),
            "budget": src.get("budget", 0),
            "release_year": src.get("release_date", "")[:4] if src.get("release_date") else ""
        })
    print_movie_table(movies, title=f"🎬  Found {total} total match(es) — showing {len(movies)}:")


if __name__ == "__main__":
    main()
