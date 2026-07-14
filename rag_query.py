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
CHROMA_PATH   = "./chroma_db"
GROQ_MODEL    = "llama-3.3-70b-versatile"
TOP_K         = 3          # how many rules to retrieve per collection
ES_INDEX      = "movies"   # target Elasticsearch index name
ES_HOST       = "http://localhost:9200"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  ChromaDB – connect and retrieve context
# ──────────────────────────────────────────────────────────────────────────────
def retrieve_context(user_query: str) -> tuple[list[str], list[str]]:
    """
    Query both ChromaDB collections and return retrieved documents.

    Returns
    -------
    schema_docs : list[str]  – top-K schema-rule documents
    vocab_docs  : list[str]  – top-K vocab-rule documents
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

    return schema_docs, vocab_docs


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Prompt assembly
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(user_query: str, schema_docs: list[str], vocab_docs: list[str]) -> str:
    """
    Construct a deterministic, zero-shot prompt for the LLM.

    The prompt enforces:
      • Elasticsearch 8.x DSL JSON only
      • No markdown, no prose – raw JSON output
      • Strict adherence to the retrieved schema and vocabulary
    """
    schema_cheat_sheet = "\n".join(f"  {i+1}. {doc}" for i, doc in enumerate(schema_docs))
    vocab_cheat_sheet  = "\n".join(f"  {i+1}. {doc}" for i, doc in enumerate(vocab_docs))

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Use only the field names listed in the schema cheat sheet.
2. For genre filtering, use ONLY the exact vocabulary values shown above.
3. Wrap multiple conditions in a bool query (must / should / must_not / filter).
4. For numeric comparisons use a range query.
5. For genre matching use a match query on the "genres" field.
6. The "size" field MUST be at the TOP LEVEL of the JSON object, NOT inside the "query" key.
7. Output ONLY the raw JSON – no markdown, no prose.

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
# 6.  Main orchestration
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Obtain user query ────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        print("🎬  Netflix RAG Query Generator")
        print("─" * 40)
        user_query = input("Enter your movie query: ").strip()
        if not user_query:
            print("❌  No query provided. Exiting.")
            sys.exit(1)

    print(f"\n📝  User query  : {user_query}")
    print("─" * 60)

    # ── Step 1: Retrieve context from ChromaDB ───────────────────────────────
    print("🔍  Retrieving context from ChromaDB …")
    try:
        schema_docs, vocab_docs = retrieve_context(user_query)
    except RuntimeError as exc:
        print(f"❌  ChromaDB error: {exc}")
        sys.exit(1)

    print(f"\n📚  Schema rules retrieved  ({len(schema_docs)}):")
    for doc in schema_docs:
        print(f"    • {doc}")

    print(f"\n📖  Vocab rules retrieved   ({len(vocab_docs)}):")
    for doc in vocab_docs:
        print(f"    • {doc}")

    # ── Step 2: Assemble prompt ──────────────────────────────────────────────
    print("\n🛠   Assembling prompt …")
    prompt = build_prompt(user_query, schema_docs, vocab_docs)

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

