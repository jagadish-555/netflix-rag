# 🎬 Netflix RAG — Natural Language to Elasticsearch Query System

A Netflix-style pipeline that converts plain English questions into Elasticsearch DSL queries using **Contextual RAG** (ChromaDB) + **Groq LLM**, backed by a **GraphQL federation** layer over SQLite.

Two query modes are supported:

| Mode | How it works | Best for |
|---|---|---|
| **Standard** | ChromaDB RAG → Groq LLM → Elasticsearch DSL | Structured filters (genre, budget, year, revenue) |
| **`--semantic`** | ChromaDB vector search → ES id lookup | "Find me a movie about…" plot descriptions |

---

## 🗂️ Project Structure

```
netflix_rag/
├── build_db.py                    # Builds SQLite databases from the TMDB CSV
├── elastic_search_index.py        # Indexes movies from SQLite → Elasticsearch
├── setup_chroma_db.py             # Seeds schema_rules + vocab_rules into ChromaDB
├── contextual_overview_index.py   # Contextual RAG: indexes all movie overviews into ChromaDB
├── rag_query.py                   # Main pipeline — standard + semantic modes
├── tmdb_5000_movies.csv           # Source dataset (TMDB 5000 Movies)
├── supergraph.yaml                # Apollo Federation supergraph config
├── supergraph.graphql             # Composed supergraph schema
├── router.yaml                    # Apollo Router runtime config
├── run_router.sh                  # Starts the Apollo Router in Docker (port 4000)
├── requirements.txt               # Python dependencies
├── .env                           # Your API keys (not committed)
└── subgraphs/
    ├── content.db                 # SQLite: title, genre, overview, release_date, runtime
    ├── finance.db                 # SQLite: budget, revenue
    ├── content_subgraph.py        # GraphQL subgraph — Content data (port 8001)
    ├── finance_subgraph.py        # GraphQL subgraph — Finance data (port 8002)
    ├── content.graphql            # Content subgraph schema
    └── finance.graphql            # Finance subgraph schema
```

---

## ✅ Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.10+ | `python3 --version` |
| Docker Desktop | Latest | `docker --version` |
| Groq API Key | — | [console.groq.com](https://console.groq.com) |

---

## 🚀 Setup (Run Once)

### Step 1 — Clone & create virtual environment

```bash
cd ~/Desktop/netflix_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Add your Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY="gsk_your_key_here"
```

### Step 3 — Build the SQLite databases from CSV

> ⚠️ Skip if `subgraphs/content.db` and `subgraphs/finance.db` already exist.

```bash
python build_db.py tmdb_5000_movies.csv
```

This creates:
- `subgraphs/content.db` — movie metadata (title, genre, overview, year, runtime)
- `subgraphs/finance.db` — financial data (budget, revenue)

---

## 🐳 Running the Full Stack (4 Separate Terminals)

There is no Docker Compose yet — each service must be started in its **own terminal**.
Open 4 terminal windows/tabs inside the `netflix_rag/` folder.

---

### Terminal 1 — Elasticsearch

```bash
docker run --rm \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0
```

Wait until you see:
```
"message": "started"
```
Verify: open `http://localhost:9200` in your browser — you should see Elasticsearch JSON info.

---

### Terminal 2 — Content Subgraph (GraphQL on port 8001)

```bash
cd subgraphs
source ../.venv/bin/activate
uvicorn content_subgraph:app --host 0.0.0.0 --port 8001 --reload
```

Verify: open `http://localhost:8001/graphql` — GraphiQL playground should load.

---

### Terminal 3 — Finance Subgraph (GraphQL on port 8002)

```bash
cd subgraphs
source ../.venv/bin/activate
uvicorn finance_subgraph:app --host 0.0.0.0 --port 8002 --reload
```

Verify: open `http://localhost:8002/graphql` — GraphiQL playground should load.

---

### Terminal 4 — Apollo Router (port 4000)

> ⚠️ Run this from the **project root** (`netflix_rag/`), not from `subgraphs/`.

```bash
bash run_router.sh
```

This starts the Apollo Router via Docker, mounting `router.yaml` and `supergraph.graphql`.

Verify: open `http://localhost:4000` — Apollo Sandbox should load.

---

### Service Port Map

| Service | Port | Terminal |
|---|---|---|
| Elasticsearch | 9200 | 1 |
| Content Subgraph | 8001 | 2 |
| Finance Subgraph | 8002 | 3 |
| Apollo Router | 4000 | 4 |

---

### Stop all services

Press `Ctrl+C` in each terminal to stop the respective service.
To stop the Apollo Router Docker container:
```bash
docker ps        # find the container name
docker stop <container_name>
```

---

## 🔍 Step 4 — Index movies into Elasticsearch

Run this in a **new terminal** (5th terminal) **after Elasticsearch is healthy** (Terminal 1 shows `"started"`):

```bash
source .venv/bin/activate
python elastic_search_index.py
```

This reads both SQLite databases, joins them, and bulk-indexes ~4,800 movies into Elasticsearch.

---

## 🧠 Step 5 — Seed ChromaDB (RAG Vector Store)

Run this **after** Elasticsearch is running and the movies are indexed:

```bash
source .venv/bin/activate
python setup_chroma_db.py
```

This introspects the **live Elasticsearch mapping** and the **real genre vocabulary** from SQLite to build two ChromaDB collections:

| Collection | Contents | Source |
|---|---|---|
| `schema_rules` | ES field mappings + DSL operator hints | ES `/movies/_mapping` |
| `vocab_rules` | 20 exact genre terms | `subgraphs/content.db` |

> ✅ Safe to re-run — uses upsert. Re-run if the ES schema or genres change.

---

## 🎯 Step 5a — Build Contextual Overview Index (Contextual RAG)

This is the core of the **Contextual RAG** implementation, inspired by [Anthropic's Contextual Retrieval blog](https://www.anthropic.com/engineering/contextual-retrieval).

```bash
source .venv/bin/activate
python contextual_overview_index.py
```

### What it does

Each movie's `overview` is embedded **with a structured context prefix** built from its metadata:

```
# Raw overview (bad for semantic search alone):
"A cryptic message from Bond's past sends him on a trail..."

# Contextualised overview (what gets embedded):
"Spectre (2015) — Genres: Action, Adventure, Thriller.
A cryptic message from Bond's past sends him on a trail..."
```

The context prefix (`Title (Year) — Genres: ...`) tells the embedding model **who, what, and when** — so queries like *"Bond spy film"* or *"2015 action thriller"* correctly surface the right movie even if those words aren't in the overview text.

**Key design decisions:**
- ✅ No LLM calls — context is built deterministically from SQLite columns
- ✅ No chunking — each overview is already atomic and self-contained
- ✅ Runs in seconds — all 4,799 movies processed instantly
- ✅ Safe to re-run — uses ChromaDB `upsert`

This creates a third ChromaDB collection:

| Collection | Contents | Source |
|---|---|---|
| `movie_overviews` | 4,799 contextualised plot embeddings | `subgraphs/content.db` |

---

## 🎬 Step 6 — Run the RAG Query Pipeline

> Make sure Elasticsearch (Terminal 1) is still running before querying.

```bash
source .venv/bin/activate
```

### Mode 1 — Standard (NL → ES DSL query)

Best for structured filters: genre, budget, revenue, year ranges.

```bash
# CLI mode
python rag_query.py "Show me sci-fi movies that made over 100 million"

# Interactive mode
python rag_query.py
# → Enter your movie query: <type here>
```

### Mode 2 — Semantic (`--semantic` flag)

Best for plot descriptions — no LLM needed, just vector similarity.

```bash
# CLI mode
python rag_query.py --semantic "movie about a hero fighting a clown villain in a city"

# Interactive mode
python rag_query.py --semantic
# → Enter your movie query: <type here>
```

---

## 🔁 How the Two Modes Work

### Standard mode

```
User Query (natural language)
        │
        ▼
  ChromaDB (RAG) — 3 collections queried in parallel
  ├── schema_rules    → top-3 ES field mappings
  ├── vocab_rules     → top-3 exact genre terms
  └── movie_overviews → top-3 contextualised plots (hint to LLM)
        │
        ▼
  Groq LLM (llama-3.3-70b-versatile)
  → generates Elasticsearch DSL JSON
        │
        ▼
  Elasticsearch (localhost:9200)
  → executes the query against 4,800 movies
        │
        ▼
  Formatted Results Table (Title, Genres, Revenue, Budget, Year)
```

### `--semantic` mode

```
User Query (plot description)
        │
        ▼
  ChromaDB movie_overviews
  → ranks top-10 movies by vector similarity   ← Contextual RAG IS the ranking
        │  returns ranked movie_ids
        ▼
  Elasticsearch (terms query by id)
  → fetches full data in one request            ← enriches with revenue, budget, year
        │
        ▼
  Formatted Results Table (same columns, ChromaDB rank order)
```

> **Why two modes?** The LLM is great at translating "budget over $100M" or "released after 2015" into structured ES queries. But for vague plot descriptions like "movie where fish looks for his son", ChromaDB's vector similarity directly gives better results without the LLM overhead.

---

## 📌 Example Queries

### Standard mode

```bash
python rag_query.py "find me movie that has generated highest revenue"
python rag_query.py "top 5 action movies with budget under 50 million"
python rag_query.py "recent horror or thriller films released after 2015"
python rag_query.py "comedy movies released between 2010 and 2020"
python rag_query.py "animated family movies that made more than 500 million"
python rag_query.py "find movies in which joker is a villain"
```

### Semantic mode

```bash
python rag_query.py --semantic "movie about a hero fighting a clown villain in a city"
python rag_query.py --semantic "animated movie about a fish looking for his son"
python rag_query.py --semantic "spy who uncovers a secret organisation"
python rag_query.py --semantic "robot learns to feel emotions"
```

---

## 📄 Example Output

### Standard mode

```
📝  User query  : find movies in which joker is a villain
────────────────────────────────────────────────────────────
🔍  Retrieving context from ChromaDB …

🎬  Overview docs retrieved (3) [Contextual RAG]:
    • The Dark Knight (2008) — Genres: Drama, Action, Crime, Thriller.
    • Batman (1989) — Genres: Fantasy, Action.
    • Batman (1966) — Genres: Family, Adventure, Comedy, Science Fiction, Crime.

🛠   Assembling prompt …
🤖  Sending to Groq (llama-3.3-70b-versatile) …

════════════════════════════════════════════════════════════
✅  Generated Elasticsearch DSL Query:
════════════════════════════════════════════════════════════
{
  "size": 100,
  "query": {
    "bool": {
      "must": [{ "match": { "overview": "Joker" } }]
    }
  }
}
════════════════════════════════════════════════════════════

🎬  Found 3 total match(es) — showing 3:

+─────+────────────────────────────────────────+──────────────────────+────────────────+────────────────+────────+
|   # | Title                                  | Genres               |  Revenue (USD) |   Budget (USD) |   Year |
+─────+────────────────────────────────────────+──────────────────────+────────────────+────────────────+────────+
|   1 | Batman                                 | Fantasy, Action      |   $411,348,924 |    $35,000,000 |   1989 |
|   2 | The Dark Knight                        | Drama, Action, Crime | $1,004,558,444 |   $185,000,000 |   2008 |
|   3 | Batman Returns                         | Action, Fantasy      |   $280,000,000 |    $80,000,000 |   1992 |
+─────+────────────────────────────────────────+──────────────────────+────────────────+────────────────+────────+
```

### Semantic mode

```
📝  User query  : movie about a hero fighting a clown villain in a city
────────────────────────────────────────────────────────────
🔎  Mode: semantic vector search (ChromaDB ranks · ES enriches) …
    Retrieved 10 movie(s) from ChromaDB, fetching ES data …
    ✅  ES enrichment: 10/10 movies found.

════════════════════════════════════════════════════════════
✅  Elasticsearch DSL Query (semantic id lookup):
════════════════════════════════════════════════════════════
{
  "size": 10,
  "query": { "terms": { "id": [155, 364, 272, ...] } }
}
════════════════════════════════════════════════════════════

🔎  Semantic search results for: 'movie about a hero fighting a clown villain in a city'
    Ranked by plot similarity (ChromaDB) · enriched by Elasticsearch

+─────+────────────────────────────────────────+──────────────────────+────────────────+────────────────+────────+
|   # | Title                                  | Genres               |  Revenue (USD) |   Budget (USD) |   Year |
+─────+────────────────────────────────────────+──────────────────────+────────────────+────────────────+────────+
|   1 | Defendor                               | Drama, Action, Comedy|            N/A |            N/A |   2009 |
|   2 | Batman                                 | Fantasy, Action      |   $411,348,924 |    $35,000,000 |   1989 |
|   3 | Batman v Superman: Dawn of Justice     | Action, Adventure    | $873,634,919   |   $250,000,000 |   2016 |
+─────+────────────────────────────────────────+──────────────────────+────────────────+────────────────+────────+
```

---

## 🌐 GraphQL Federation API (Optional)

Once the Docker stack is running, you can query the federated GraphQL API at:

**Apollo Router Sandbox:** `http://localhost:4000`

Example federated query (combines Content + Finance subgraphs in one request):

```graphql
query {
  movieContent(id: 19995) {
    title
    genre
    releaseDate
  }
}
```

---

## 🔄 Full Restart from Scratch

```bash
# Terminal 1 — Start Elasticsearch
docker run --rm -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.13.0

# Terminal 2 — Content Subgraph
cd subgraphs && uvicorn content_subgraph:app --host 0.0.0.0 --port 8001

# Terminal 3 — Finance Subgraph
cd subgraphs && uvicorn finance_subgraph:app --host 0.0.0.0 --port 8002

# Terminal 4 — Apollo Router
bash run_router.sh

# Terminal 5 (or any new terminal) — Index + Seed + Query
python elastic_search_index.py        # index movies into ES
python setup_chroma_db.py             # seed schema + vocab rules
python contextual_overview_index.py   # build contextual overview embeddings
python rag_query.py "your question here"
python rag_query.py --semantic "plot description here"
```

---

## 🛠️ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Port 4000 already allocated` | `run_router.sh` ran while another instance is active | `docker ps` to find it, then `docker stop <name>` |
| `GROQ_API_KEY not set` | `.env` file missing or malformed | Create `.env` with `GROQ_API_KEY="gsk_..."` |
| `Cannot open ChromaDB` | `setup_chroma_db.py` not run yet | Run `python setup_chroma_db.py` |
| `'movie_overviews' collection not found` | `contextual_overview_index.py` not run yet | Run `python contextual_overview_index.py` |
| `Elasticsearch connection failed` | Terminal 1 (ES) not running | Start Elasticsearch in Terminal 1 |
| `no such table: movies` | Wrong working directory for subgraph | Run `uvicorn` from inside `subgraphs/` folder |
| LLM returns wrong genre name | RAG vocabulary not seeded | Re-run `python setup_chroma_db.py` |
| `--semantic` returns unrelated movies | `movie_overviews` collection is stale | Re-run `python contextual_overview_index.py` |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Dataset | TMDB 5000 Movies CSV |
| Storage | SQLite (content + finance silos) |
| Search Engine | Elasticsearch 8.x |
| Vector Store | ChromaDB (`all-MiniLM-L6-v2` embeddings) |
| Contextual RAG | `contextual_overview_index.py` — deterministic context prefix from metadata |
| LLM | Groq — `llama-3.3-70b-versatile` |
| GraphQL | Strawberry + FastAPI + Apollo Router |
| Environment | Python 3.10+, Docker |
