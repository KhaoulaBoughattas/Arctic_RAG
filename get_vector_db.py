import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Load data with embeddings
input_file = "data/chunks_with_embeddings.json"
with open(input_file, "r", encoding='utf-8') as f:
    data = json.load(f)

# Connect to Qdrant
client = QdrantClient(host='localhost', port=6333, timeout=600)

# Collection name
collection_name = "psybot-embedding"

# Delete existing collection if it exists
if client.collection_exists(collection_name):
    print(f"⚡ Collection {collection_name} already exists. Deleting...")
    client.delete_collection(collection_name)

# Create collection
print(f"🟣 Creating collection {collection_name}")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# Insert points in batches
batch_size = 500
points = []
for idx, item in enumerate(data):
    points.append(
        PointStruct(
            id=idx,
            vector=item['embedding'],
            payload={"page_content": item['page_content']}
        )
    )

print(f"➡ Inserting {len(points)} points into Qdrant")
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    try:
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"✅ {len(batch)} points inserted successfully.")
    except Exception as e:
        print(f"❌ Error inserting points: {e}")

print("✅ Qdrant is ready and operational.")