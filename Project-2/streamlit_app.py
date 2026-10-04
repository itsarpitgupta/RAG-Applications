"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Ensure Python can locate the 'src' package from the current file location
sys.path.append(str(Path(__file__).parent))

# Import our core components
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Set up browser tab title, favicon, and screen layout
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Apply custom styling for a clean, modern button appearance
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """
    Initializes Streamlit's session memory variables so that user search history
    and the compiled RAG pipeline stay persisted across re-runs.
    """
    # Holds our compiled LangGraph runner
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    # Flag to check if we've already initialized documents and vector embeddings
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    # List of past user questions, answers, and execution durations
    if 'history' not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def initialize_rag():
    """
    Performs the one-time startup sequence for our RAG system:
    1. Loads the LLM from Config (e.g. ChatGroq or ChatOpenAI).
    2. Initializes DocumentProcessor with configured chunk size & overlap.
    3. Loads source articles from data/url.txt and slices them into chunks.
    4. Builds the local FAISS vector store with HuggingFace embeddings.
    5. Assembles and compiles the LangGraph workflow.
    
    The @st.cache_resource decorator ensures this heavy setup only happens once,
    keeping app interactions fast.
    """
    try:
        # Step 1: Create our LLM instance based on environment keys
        llm = Config.get_llm()

        # Step 2: Initialize our document processor with chunk parameters
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        # Step 3: Initialize the vector store using local HuggingFace embeddings
        vector_store = VectorStore()

        # Step 4: Gather URLs from our configuration or data/url.txt
        urls = Config.DEFAULT_URLS

        # Step 5: Download the web pages and split them into chunks
        documents = doc_processor.process_urls(urls)

        # Step 6: Convert the chunks into vector embeddings and index them in FAISS
        vector_store.create_vectorstore(documents)

        # Step 7: Construct our LangGraph workflow tying the retriever to the LLM
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0


def main():
    """
    The main user interface function that renders:
    - Page headers and description.
    - System initialization trigger with spinner.
    - Input form for asking questions.
    - Result display with the LLM answer and expandable source documents.
    - Recent search history timeline.
    """
    # Make sure session memory variables are initialized
    init_session_state()

    # Render main title and description
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")

    # One-time initialization on first launch
    if not st.session_state.initialized:
        with st.spinner("Loading system and indexing documents..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    # Question submission form
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")

    # Handle question submission
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching documents and formulating answer..."):
                start_time = time.time()

                # Run the LangGraph RAG workflow with the user's question
                result = st.session_state.rag_system.run(question)

                elapsed_time = time.time() - start_time

                # Save search query and answer into session history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })

                # Display the AI's answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])

                # Render an expandable viewer showing the exact source text chunks used
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )

                # Show response generation time
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # Display recent search history if questions have been asked
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")

        # Show the most recent 3 searches in reverse chronological order
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")


if __name__ == "__main__":
    main()