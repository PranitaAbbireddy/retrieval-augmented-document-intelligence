import os
from typing import List, Dict, Any
from google import genai
from google.genai import types
import time

class LLMClient:
    def __init__(self):
        """
        Initializes the LLM client using the free Gemini API.
        Fails gracefully if the API key is not present.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
            
        # Use the fast, free tier model
        self.model_name = "gemini-3.8-flash"

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates a grounded answer based ONLY on the provided context.
        """
        if not self.client:
            return "Error: GEMINI_API_KEY environment variable is not set. Please set it to use the LLM."
            
        if not context_chunks:
            return "I cannot answer this question because no relevant context was found in the uploaded documents."

        # Construct the context block
        context_texts = []
        for i, chunk in enumerate(context_chunks):
            doc = chunk.get('doc_name', 'Unknown')
            page = chunk.get('page', 'Unknown')
            text = chunk.get('text', '')
            context_texts.append(f"[Source {i+1}: {doc}, Page {page}]\n{text}")
            
        context_block = "\n\n".join(context_texts)
        
        # Strict grounding prompt
        prompt = f"""You are a helpful and precise document intelligence assistant.
Your task is to answer the user's question based strictly on the context provided below.

Context:
{context_block}

Question:
{query}

Instructions:
1. Answer the question using ONLY the provided context.
2. If the context does not contain the answer, explicitly state: "The provided documents do not contain enough information to answer this question." Do NOT invent or hallucinate information.
3. Be concise and professional.
4. When you state a fact, you may reference the source number (e.g., [Source 1]).
"""
        
        max_retries = 3
        backoff_time = 1
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0, # Keep deterministic
                    )
                )
                return response.text
            except Exception as e:
                error_msg = str(e)
                # Check for transient errors like 503 or UNAVAILABLE
                is_transient = "503" in error_msg or "UNAVAILABLE" in error_msg or "high demand" in error_msg
                
                if is_transient and attempt < max_retries:
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff (1s, 2s, 4s)
                elif is_transient and attempt == max_retries:
                    return "I'm sorry, the AI service is currently experiencing high demand and is temporarily unavailable. Please try your question again in a few moments."
                else:
                    # Non-transient error or unknown error, fail immediately
                    return f"Error during generation: {error_msg}"
