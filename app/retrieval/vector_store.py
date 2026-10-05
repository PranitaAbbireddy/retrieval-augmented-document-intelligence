import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, index_dir: str = "data"):
        """
        Initializes the VectorStore with a CPU-friendly embedding model and FAISS.
        """
        # Load a highly efficient, CPU-friendly embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_dir = index_dir
        
        # Ensure the data directory exists
        os.makedirs(self.index_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(self.index_dir, "metadata.json")
        
        # The embedding dimension for all-MiniLM-L6-v2 is 384
        self.dimension = 384
        
        # Initialize or load FAISS index and metadata
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._load_or_create_index()

    def _load_or_create_index(self):
        """
        Loads an existing FAISS index and metadata from disk, or creates new ones.
        """
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            # Create a simple flat L2 index suitable for small-medium datasets
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    def save_index(self):
        """
        Persists the FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Generates embeddings for the provided chunks and adds them to the vector store.
        """
        if not chunks:
            return
            
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings as a numpy array
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Append metadata
        self.metadata.extend(chunks)
        
        # Save to disk immediately for persistence
        self.save_index()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar chunks for a given query.
        """
        if self.index.ntotal == 0:
            return []
            
        # Generate embedding for the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):  # Check for valid index
                result = self.metadata[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)
                
        return results

    def clear_index(self):
        """
        Removes all documents from the index and deletes local files.
        """
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
