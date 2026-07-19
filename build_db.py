import sqlite3
import sys
import os
import json
import pandas as pd

# Resolve paths relative to this script, not the caller's CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_SUBGRAPHS = os.path.join(_HERE, "subgraphs")

def parse_json_list(raw: str, key: str = "name") -> str:
    """
    TMDB stores fields like genres as JSON strings: [{"id": 28, "name": "Action"}]
    This safely extracts them into a clean, comma-separated string: "Action, Adventure".
    """
    if pd.isna(raw) or not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        items = json.loads(raw)
        return ", ".join(str(item[key]) for item in items if key in item)
    except (json.JSONDecodeError, TypeError):
        return ""

def build(csv_path: str,
          content_db: str = None,
          finance_db: str = None):
    if content_db is None:
        content_db = os.path.join(_SUBGRAPHS, "content.db")
    if finance_db is None:
        finance_db = os.path.join(_SUBGRAPHS, "finance.db")
    os.makedirs(os.path.dirname(content_db), exist_ok=True)
    os.makedirs(os.path.dirname(finance_db), exist_ok=True)

    print(f"Loading data from {csv_path}...")
    # Load CSV
    df = pd.read_csv(csv_path)

    # ==========================================
    # DATA CLEANING & STANDARDIZATION
    # ==========================================
    print("Cleaning and standardizing data...")
    
    # 1. Deduplication and primary key safety
    df = df.dropna(subset=["id", "title"]).copy()
    df = df.drop_duplicates(subset=["id"])

    # 2. Clean textual fields
    df["overview"] = df["overview"].fillna("").str.strip()
    df["title"] = df["title"].fillna("").str.strip()
    df["genre"] = df["genres"].apply(parse_json_list)

    # 3. Clean dates and derive an integer release_year for math queries (e.g. > 2010)
    df["release_date"] = df["release_date"].fillna("")
    df["release_year"] = df["release_date"].apply(
        lambda d: int(d[:4]) if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit() else 0
    )

    # 4. Enforce strict integer types to prevent GraphQL schema crashes
    df["runtime"] = df["runtime"].fillna(0).astype(int)
    df["budget"] = df["budget"].fillna(0).astype(int)
    df["revenue"] = df["revenue"].fillna(0).astype(int)

    # 5. Pre-compute contextualized overview for RAG
    df["contextualized_overview"] = df.apply(
        lambda row: f"{row['title']} ({row['release_year']}) - Genres: {row['genre']}.\n{row['overview']}",
        axis=1
    )


    # ==========================================
    # SILO 1: CONTENT DEPARTMENT
    # ==========================================
    print(f"Building Content Silo ({content_db})...")
    with sqlite3.connect(content_db) as conn:
        content_df = df[["id", "title", "release_date", "release_year", "genre", "runtime", "overview", "contextualized_overview"]]
        content_df.to_sql("movies", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_content_id ON movies(id)")

    # ==========================================
    # SILO 2: FINANCE DEPARTMENT
    # ==========================================
    print(f"Building Finance Silo ({finance_db})...")
    with sqlite3.connect(finance_db) as conn:
        finance_df = df[["id", "budget", "revenue"]]
        finance_df.to_sql("financials", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_finance_id ON financials(id)")

    print(f"✅ Success! Data silos created.")
    print(f"Content DB: {len(content_df)} rows written.")
    print(f"Finance DB: {len(finance_df)} rows written.")

if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "tmdb_5000_movies.csv"
    build(csv_arg)