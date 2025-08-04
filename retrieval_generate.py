from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest, SearchParams
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import subprocess
import time

def retrieve_and_rerank(
    query: str,
    collection_name: str = "psybot-embedding",
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
    top_k: int = 5,
    search_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Hybrid dense + BM25 retrieval followed by reranking.
    """
    try:
        # Connect to Qdrant with longer timeout
        print(f"🔌 Connecting to Qdrant at {qdrant_url}:{qdrant_port}...")
        client = QdrantClient(host=qdrant_url, port=qdrant_port, timeout=300)
        
        # Test connection
        collections = client.get_collections()
        print("✅ Connected to Qdrant successfully")
        
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return []

    # Models for embeddings and reranking
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    # Encode query
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]

    # Dense search in Qdrant (using the correct method)
    try:
        print("🔍 Searching in Qdrant...")
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=search_k,
            with_payload=True,
        )
        print(f"✅ Found {len(search_result)} results")
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return []

    # Extract documents and scores
    docs = [item.payload['page_content'] for item in search_result]
    dense_scores = [item.score for item in search_result]

    if not docs:
        print("⚠️ No documents found.")
        return []

    # BM25 on retrieved documents
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Combine dense and BM25 scores (weighted)
    alpha = 0.6  # Weight for dense scores
    beta = 0.4   # Weight for BM25 scores
    combined_scores = [alpha * d + beta * b for d, b in zip(dense_scores, bm25_scores)]

    # Select top 2 * top_k candidates for reranking
    top_candidates_idx = sorted(
        range(len(combined_scores)),
        key=lambda i: combined_scores[i],
        reverse=True
    )[:2 * top_k]

    # Prepare pairs for reranking
    pairs = [(query, docs[i]) for i in top_candidates_idx]

    # Rerank using CrossEncoder
    rerank_scores = reranker.predict(pairs)

    # Final sorting based on rerank scores
    reranked_idx = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)[:top_k]

    # Final results
    results = [
        {"score": rerank_scores[i], "page_content": docs[top_candidates_idx[i]]}
        for i in reranked_idx
    ]
    return results

def generate(prompt: str, documents: List[Dict[str, Any]], model='qwen2.5:1.5b') -> str:
    """
    Generate a coherent response using Qwen 2.5:7B with the given documents.
    """
    context = "DOCUMENTS PERTINENTS:\n" + \
              "\n".join([f"- {d['page_content']} (score: {d['score']:.4f})" for d in documents])
    full_prompt = f"""{context}
Question: {prompt}
Can you provide a coherent, gentle, and professional representation of the best answer using the relevant documents?
    """

    try:
        print("🤖 Generating response with Qwen...")
        result = subprocess.run(
            ['ollama', 'run', model],
            input=full_prompt,
            text=True,
            capture_output=True,
            encoding='utf-8',
            timeout=120  # Add timeout for generation
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error during generation: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "❌ Generation timed out. Please try again."
    except Exception as e:
        return f"Exception during generation: {e}"

def main():
    print("🔍 Hybrid dense + BM25 + CrossEncoder reranking")
    query = input("❓ Enter your question: ").strip()
    if not query:
        print("⚡ Empty question, exiting.")
        return

    # Step 1: Retrieve and rerank documents
    results = retrieve_and_rerank(query)

    if not results:
        print("❌ No relevant results found.")
        return

    # Step 2: Display retrieved documents
    print("\n📄 Results:")
    for i, res in enumerate(results, 1):
        print(f"{i}. Score: {res['score']:.4f}\nContent: {res['page_content']}\n{'-'*40}")

    # Step 3: Generate a response with context
    print("\n🤖 Generating response with context...")
    response = generate(query, results)
    print("\n🔹 Generated Response 🔹")
    print(response)

if __name__ == "__main__":
    main()