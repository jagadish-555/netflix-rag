import os
import sqlite3
from elasticsearch import Elasticsearch, helpers

def index_from_databases(content_db=None, finance_db=None, index_name="movies"):
    # Resolve DB paths relative to this script's location so it works from any CWD
    _here = os.path.dirname(os.path.abspath(__file__))
    if content_db is None:
        content_db = os.path.join(_here, "subgraphs", "content.db")
    if finance_db is None:
        finance_db = os.path.join(_here, "subgraphs", "finance.db")

    es = Elasticsearch(["http://localhost:9200"])

    # 1. Open a single connection to content.db and ATTACH finance.db
    #    so both tables are visible in one SQLite session.
    conn = sqlite3.connect(content_db)
    conn.execute(f"ATTACH DATABASE '{finance_db}' AS finance")

    # 2. Cross-database JOIN: content.movies LEFT JOIN finance.financials
    query = """
    SELECT
        m.id, m.title, m.overview, m.genre, m.release_date,
        f.budget, f.revenue
    FROM movies m
    LEFT JOIN finance.financials f ON m.id = f.id
    """

    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # 3. Generator for bulk indexing
    def generate_actions():
        for row in rows:
            source = {
                "id": row[0],
                "title": row[1],
                "overview": row[2],
                "genres": row[3],
                "budget": row[5],
                "revenue": row[6],
            }
            # Skip empty release_date — Elasticsearch rejects empty strings on date fields
            if row[4]:
                source["release_date"] = row[4]
            yield {
                "_index": index_name,
                "_id": row[0],
                "_source": source,
            }

    print(f"Indexing {len(rows)} movies from databases...")
    success, failed = helpers.bulk(es, generate_actions(), raise_on_error=False)
    print(f"✅ Indexed {success} documents successfully.")
    if failed:
        print(f"⚠️  {len(failed)} document(s) failed to index:")
        for err in failed:
            print(" -", err)

    conn.close()

if __name__ == "__main__":
    index_from_databases()