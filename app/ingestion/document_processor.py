import os
import fitz  # PyMuPDF
from typing import List, Dict, Any

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF file page by page.
    Returns a list of dictionaries containing page metadata and text.
    """
    # Load the document and preserve page metadata for source attribution
    doc = fitz.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Clean basic whitespace issues
        text = " ".join(text.split())
        
        if text.strip():
            pages_data.append({
                "page": page_num + 1,
                "text": text
            })
            
    doc.close()
    return pages_data

def chunk_text(pages_data: List[Dict[str, Any]], doc_name: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Splits page text into smaller chunks for embedding generation.
    Uses a simple word-based splitting approach suitable for CPU processing.
    """
    chunks = []
    chunk_id_counter = 0
    
    for page_data in pages_data:
        text = page_data["text"]
        words = text.split()
        
        # Iterate over words and create overlapping chunks
        start_idx = 0
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "chunk_id": f"{doc_name}_chunk_{chunk_id_counter}",
                "doc_name": doc_name,
                "page": page_data["page"],
                "text": chunk_text
            })
            
            chunk_id_counter += 1
            
            # Break if we reached the end of the text
            if end_idx == len(words):
                break
                
            # Advance start_idx, but account for overlap
            start_idx = end_idx - overlap
            
    return chunks
