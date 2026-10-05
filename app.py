import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.ingestion.document_processor import extract_text_from_pdf, chunk_text
from app.retrieval.vector_store import VectorStore
from app.generation.llm_client import LLMClient

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Polished Pastel UI ---
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #fdfbf7; /* Very light cream/off-white */
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers and Text */
    h1, h2, h3 {
        color: #4a4a4a;
        font-weight: 600;
    }
    p, span, div {
        color: #5c5c5c;
    }

    /* Citation Cards */
    .citation-card {
        background-color: #f4f7fa; /* Pale blue/grey */
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .citation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.05);
    }
    .citation-header {
        font-weight: 600;
        color: #6c72cb; /* Soft lavender blue */
        margin-bottom: 8px;
        font-size: 0.95em;
    }
    .citation-text {
        font-size: 0.9em;
        color: #64748b;
        line-height: 1.6;
    }

    /* Error Cards */
    .error-card {
        background-color: #fef1f2; /* Pastel pink/red */
        border: 1px solid #fecdd3;
        border-radius: 12px;
        padding: 16px;
        color: #be123c;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Warning Cards */
    .warning-card {
        background-color: #fffbeb; /* Light peach/yellow */
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 16px;
        color: #b45309;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Success Cards */
    .success-card {
        background-color: #f0fdf4; /* Pastel green */
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 12px 16px;
        color: #15803d;
        margin-bottom: 16px;
        font-size: 0.95em;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: transparent !important;
    }
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #ffffff !important;
        border: 1px solid #f1f5f9;
        border-radius: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton>button[kind="primary"] {
        background-color: #8b93ff; /* Soft lavender */
        color: white;
        border: none;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #7a82f5;
        box-shadow: 0 4px 12px rgba(139, 147, 255, 0.3);
    }
    
    /* Hide Streamlit basic elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Initialize Global Resources ---
@st.cache_resource
def get_vector_store():
    return VectorStore()

@st.cache_resource
def get_llm_client():
    return LLMClient()

vector_store = get_vector_store()
llm_client = get_llm_client()

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.markdown("### ✨ Document AI")
    st.markdown("Upload documents to your secure local knowledge base.")
    
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents", type="primary", use_container_width=True) and uploaded_files:
        with st.spinner("Extracting and processing documents..."):
            all_chunks = []
            for file in uploaded_files:
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    pages_data = extract_text_from_pdf(tmp_path)
                    chunks = chunk_text(pages_data, doc_name=file.name)
                    all_chunks.extend(chunks)
                finally:
                    os.remove(tmp_path)
            
            # Add to local FAISS store
            vector_store.add_chunks(all_chunks)
            st.markdown(f'<div class="success-card">✨ Successfully processed {len(uploaded_files)} document(s).</div>', unsafe_allow_html=True)
            
    st.divider()
    
    # Advanced/Developer Settings (Hide clutter)
    with st.expander("⚙️ Advanced Settings"):
        st.markdown("Configure retrieval parameters and view index health.")
        top_k = st.slider("Retrieval Results (Top-K)", min_value=1, max_value=10, value=3)
        
        if vector_store.index:
            st.caption(f"Index Size: {vector_store.index.ntotal} chunks")
        else:
            st.caption("Index Size: 0 chunks")
            
        if st.button("Clear Vector Index"):
            vector_store.clear_index()
            st.success("Knowledge base cleared.")

# --- Main Area: Header and Chat Interface ---
st.title("AI Document Assistant")
st.markdown("Ask natural language questions about your uploaded documents using semantic retrieval.")
st.write("") # Spacer

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to render custom UI cards
def render_error(message):
    st.markdown(f'<div class="error-card">⚠️ <span>{message}</span></div>', unsafe_allow_html=True)
    
def render_warning(message):
    st.markdown(f'<div class="warning-card">💡 <span>{message}</span></div>', unsafe_allow_html=True)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📄 View Source Citations"):
                for idx, cit in enumerate(message["citations"]):
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">Source {idx+1}: {cit['doc_name']} (Page {cit['page']})</div>
                        <div class="citation-text">...{cit['text']}...</div>
                    </div>
                    """, unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Pre-flight checks with custom UI cards
    if vector_store.index is None or vector_store.index.ntotal == 0:
        render_warning("Your knowledge base is empty. Please upload some PDF documents first.")
    elif not os.environ.get("GEMINI_API_KEY"):
        render_error("GEMINI_API_KEY environment variable is missing. Please check your .env configuration.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        with st.spinner("Searching your documents & generating an answer..."):
            # Ensure top_k is retrieved from the advanced expander (defaults to 3 if not set)
            try:
                k_val = top_k
            except NameError:
                k_val = 3
                
            # Retrieve relevant chunks
            context_chunks = vector_store.search(prompt, top_k=k_val)
            
            # Generate answer
            answer = llm_client.generate_answer(prompt, context_chunks)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # If the answer contains "Error during generation:", we can optionally format it,
            # but since the prompt asked us not to change the LLM client, we'll just display it.
            # However, we can catch it here if we want to format it as an error card.
            if answer.startswith("Error during generation:"):
                render_error(answer)
            else:
                st.markdown(answer)
                
            if context_chunks:
                with st.expander("📄 View Source Citations"):
                    for idx, cit in enumerate(context_chunks):
                        st.markdown(f"""
                        <div class="citation-card">
                            <div class="citation-header">Source {idx+1}: {cit['doc_name']} (Page {cit['page']})</div>
                            <div class="citation-text">...{cit['text']}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Add assistant response and citations to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "citations": context_chunks
        })
