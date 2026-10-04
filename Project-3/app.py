import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import uuid
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from config.settings import load_settings, BASE_DIR
from src.core.database import DatabaseManager
from src.core.models import DocumentRecord, DocumentStatus
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.vectorstore.embedder import EmbeddingManager
from src.vectorstore.faiss_manager import FAISSManager
from src.ingestion.delta_worker import DeltaIngestionWorker
from src.scheduler.job_runner import IngestionScheduler
from src.llm.grok_client import GrokClient
from src.pipeline.rag_engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="Incremental RAG System with Grok",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize System Resources (Cached across user interactions)
@st.cache_resource
def get_system_components():
    settings = load_settings()
    db = DatabaseManager(settings.storage.db_path)
    embedder = EmbeddingManager(
        model_name=settings.embeddings.model_name,
        device=settings.embeddings.device,
        normalize=settings.embeddings.normalize
    )
    faiss_mgr = FAISSManager(
        index_dir=settings.storage.faiss_index_dir,
        dimension=embedder.dimension,
        metric=settings.vectorstore.metric
    )
    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        separators=settings.chunking.separators
    )
    delta_worker = DeltaIngestionWorker(
        db=db,
        faiss_manager=faiss_mgr,
        embedder=embedder,
        chunker=chunker,
        max_batch_size=settings.scheduler.max_batch_size
    )
    scheduler = IngestionScheduler(
        delta_worker=delta_worker,
        interval_hours=settings.scheduler.interval_hours
    )
    if settings.scheduler.enabled:
        scheduler.start()

    grok_client = GrokClient(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens
    )
    rag_engine = RAGEngine(
        db=db,
        faiss_manager=faiss_mgr,
        embedder=embedder,
        grok_client=grok_client
    )

    return {
        "settings": settings,
        "db": db,
        "faiss_mgr": faiss_mgr,
        "embedder": embedder,
        "chunker": chunker,
        "delta_worker": delta_worker,
        "scheduler": scheduler,
        "grok_client": grok_client,
        "rag_engine": rag_engine
    }

components = get_system_components()
settings = components["settings"]
db = components["db"]
faiss_mgr = components["faiss_mgr"]
scheduler = components["scheduler"]
rag_engine = components["rag_engine"]
grok_client = components["grok_client"]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("⚡ Incremental RAG")
    st.caption("Scheduled Delta Ingestion & Grok LLM")
    st.divider()

    # System Health Metrics
    stats = db.get_stats()
    doc_counts = stats.get("documents", {})
    pending_count = doc_counts.get("PENDING", 0)
    indexed_count = doc_counts.get("INDEXED", 0)
    total_vectors = faiss_mgr.total_vectors

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Indexed Docs", indexed_count)
        st.metric("Pending Docs", pending_count)
    with col2:
        st.metric("FAISS Vectors", total_vectors)
        st.metric("Schedule", f"{settings.scheduler.interval_hours}h Interval")

    st.divider()

    # Manual Ingestion Trigger
    st.subheader("Delta Scheduler")
    sched_status = scheduler.get_status()
    st.write(f"**Status:** {'🟢 Active' if sched_status['scheduler_active'] else '🔴 Stopped'}")
    if sched_status.get("next_run_time"):
        st.caption(f"Next hourly run: `{sched_status['next_run_time']}`")

    if st.button("🚀 Trigger Ingestion Now", use_container_width=True, type="primary"):
        with st.spinner("Processing pending documents..."):
            res = scheduler.trigger_now()
            if res.get("status") == "in_progress":
                st.warning("Job is already running.")
            else:
                st.success(f"Indexed {res.get('processed_docs', 0)} doc(s), +{res.get('total_new_chunks', 0)} chunks!")
                st.rerun()

    st.divider()
    st.subheader("Model Configuration")
    st.write(f"**Embedding:** `{settings.embeddings.model_name.split('/')[-1]}`")
    st.write(f"**Vector Store:** `FAISS ({faiss_mgr.metric})`")
    grok_status = "🟢 Configured" if grok_client.is_configured else "🟡 Demo Mode (Add key in .env)"
    st.write(f"**Grok LLM:** {grok_status}")


# ----------------- MAIN CONTENT AREA -----------------
st.title("Document Intelligence & Incremental Search")
st.markdown(
    "Upload new documents anytime. The automated scheduler isolates and chunks **only newly added files**, "
    "preserving existing FAISS vectors with zero reprocessing."
)

tab_search, tab_docs, tab_arch = st.tabs(["🔍 Search & Ask Grok", "📁 Document Management", "📐 Architecture & Design"])

# ----------------- TAB 1: RAG SEARCH -----------------
with tab_search:
    st.subheader("Ask Questions Across All Indexed Documents")
    
    col_q, col_k = st.columns([4, 1])
    with col_q:
        user_query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What are the terms of the loan policy?",
            label_visibility="collapsed"
        )
    with col_k:
        top_k = st.slider("Top K Sources", min_value=1, max_value=8, value=settings.llm.top_k_retrieval)

    if st.button("Ask Grok", type="primary") and user_query:
        with st.spinner("Retrieving nearest chunks and generating response with Grok..."):
            response = rag_engine.answer_query(user_query, top_k=top_k)

        st.markdown("### Answer")
        st.markdown(response.answer)
        st.caption(f"Generated via: `{response.model_used}` | Retrieved Chunks: {response.total_retrieved}")

        # Sources & Citations
        if response.citations:
            st.divider()
            st.subheader("📚 Source Citations & References")
            
            for cite in response.citations:
                with st.expander(
                    f"[Source {cite.source_index}] {cite.document_name} — Page {cite.page_number} (Similarity: {cite.similarity_score:.4f})"
                ):
                    st.write(f"**Document ID:** `{cite.doc_id}` | **Chunk #:** `{cite.chunk_index}`")
                    st.info(cite.snippet)

# ----------------- TAB 2: DOCUMENT MANAGEMENT -----------------
with tab_docs:
    st.subheader("Upload Documents")
    st.info(
        "💡 **Incremental Guarantee:** Uploaded documents are saved as `PENDING`. When the scheduler triggers "
        "(or when you click 'Trigger Ingestion Now'), only these new files will be chunked and added to FAISS. "
        "Your existing documents are never re-chunked."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        upload_btn = st.button(f"Upload {len(uploaded_files)} File(s) to Ingestion Queue", type="primary")
        if upload_btn:
            saved_count = 0
            dup_count = 0
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.read()
                file_size = len(file_bytes)
                
                # Save to raw_docs
                dest_path = settings.storage.raw_docs_dir / uploaded_file.name
                with open(dest_path, "wb") as f:
                    f.write(file_bytes)

                # Compute hash
                file_hash = DocumentParser.compute_sha256(dest_path)

                # Create document record
                doc_record = DocumentRecord(
                    doc_id=str(uuid.uuid4()),
                    filename=uploaded_file.name,
                    filepath=str(dest_path),
                    file_size_bytes=file_size,
                    file_hash_sha256=file_hash,
                    status=DocumentStatus.PENDING
                )

                registered = db.register_document(doc_record)
                if registered:
                    saved_count += 1
                else:
                    dup_count += 1

            if saved_count > 0:
                st.success(f"Successfully queued {saved_count} document(s) with status `PENDING`!")
            if dup_count > 0:
                st.warning(f"{dup_count} document(s) were skipped as duplicates (identical SHA-256 hash).")
            st.rerun()

    st.divider()
    st.subheader("Document Registry Status")
    
    docs = db.get_all_documents()
    if docs:
        table_data = []
        for d in docs:
            table_data.append({
                "Filename": d.filename,
                "Status": d.status.value,
                "Chunks": d.chunk_count,
                "Size (KB)": round(d.file_size_bytes / 1024, 1),
                "Uploaded At": d.uploaded_at.strftime("%Y-%m-%d %H:%M"),
                "Indexed At": d.indexed_at.strftime("%Y-%m-%d %H:%M") if d.indexed_at else "—",
                "Doc ID": d.doc_id[:8] + "..."
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No documents uploaded yet.")

# ----------------- TAB 3: ARCHITECTURE -----------------
with tab_arch:
    st.subheader("How Delta Chunking Works")
    st.markdown("""
    1. **Upload**: Documents are written to disk and recorded in SQLite with state `PENDING` and a SHA-256 content checksum.
    2. **Hourly Scheduler**: APScheduler triggers an hourly worker that isolates **only** `PENDING` documents (`WHERE status = 'PENDING'`).
    3. **Chunking & Embedding**: Only new documents are chunked and converted to vectors using `sentence-transformers/all-MiniLM-L6-v2`.
    4. **Incremental FAISS Append**: `FAISS.add()` appends new vectors onto the existing in-memory index starting at index offset `N`.
    5. **Metadata Sync**: SQLite records chunk texts and metadata corresponding to the new vector IDs.
    6. **Zero Re-indexing Overhead**: Historical vectors remain intact. Ingestion time for file 101 is $O(\\text{len}(\\text{file}_{101}))$, not $O(\\sum_{i=1}^{101} \\text{len}(\\text{file}_i))$.
    """)
