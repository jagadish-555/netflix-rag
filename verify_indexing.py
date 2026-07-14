from elasticsearch import Elasticsearch

def verify_index(index_name="movies"):
    # Connect to the local Elasticsearch instance
    es = Elasticsearch(["http://localhost:9200"])
    
    if not es.ping():
        print("❌ Cannot connect to Elasticsearch.")
        return

    # 1. Check if index exists
    if not es.indices.exists(index=index_name):
        print(f"❌ Index '{index_name}' does not exist.")
        return

    # 2. Get total document count
    count = es.count(index=index_name)['count']
    print(f"✅ Index '{index_name}' exists.")
    print(f"📊 Total movies indexed: {count}")

    # 3. Retrieve one sample document to verify structure
    res = es.search(index=index_name, body={"size": 1, "query": {"match_all": {}}})
    if res['hits']['hits']:
        sample = res['hits']['hits'][0]['_source']
        print("\n🔍 Sample Document Structure:")
        for key, value in sample.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️ Index is empty.")

if __name__ == "__main__":
    verify_index()