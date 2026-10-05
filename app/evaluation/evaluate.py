import sys
import os
import json

# Ensure we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.retrieval.vector_store import VectorStore

def evaluate_retrieval(top_k: int = 3):
    """
    A lightweight evaluation script that computes Recall@K.
    Requires a pre-populated FAISS index and a matching evaluation dataset.
    """
    print("--- RAG Retrieval Evaluation ---")
    
    # 1. Initialize vector store
    store = VectorStore(index_dir="data")
    if store.index.ntotal == 0:
        print("Error: The vector index is empty. Please run the application and upload documents first.")
        return

    # 2. Define or load evaluation dataset
    # In a real scenario, this would be loaded from a CSV containing query <-> doc_name/chunk pairs
    # For demonstration, we use a simple hardcoded dataset format
    eval_dataset = [
        # Example format: {"query": "What is X?", "expected_doc": "document_1.pdf"}
    ]
    
    if not eval_dataset:
        print("Note: The evaluation dataset is empty. To run a real evaluation:")
        print("1. Upload a specific document (e.g., 'attention_is_all_you_need.pdf') via the app.")
        print("2. Add queries and expected document names to the 'eval_dataset' list in this script.")
        print("Example: {\"query\": \"What is self-attention?\", \"expected_doc\": \"attention_is_all_you_need.pdf\"}")
        return

    hits = 0
    total = len(eval_dataset)
    
    print(f"Evaluating {total} queries with Top-K={top_k}...\n")
    
    for item in eval_dataset:
        query = item["query"]
        expected_doc = item["expected_doc"]
        
        # Retrieve chunks
        results = store.search(query, top_k=top_k)
        
        # Check if the expected document is in the retrieved chunks
        hit = any(res["doc_name"] == expected_doc for res in results)
        if hit:
            hits += 1
            
        print(f"Q: {query}")
        print(f"Expected Doc: {expected_doc}")
        print(f"Hit: {'YES' if hit else 'NO'}\n")

    recall = hits / total if total > 0 else 0
    print(f"--- Results ---")
    print(f"Recall@{top_k}: {recall:.2f} ({hits}/{total})")
    print(f"---------------")

if __name__ == "__main__":
    evaluate_retrieval(top_k=3)
