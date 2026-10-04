import logging
from pathlib import Path
from typing import Dict, Any
from src.core.database import DatabaseManager
from src.core.models import DocumentStatus
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.vectorstore.embedder import EmbeddingManager
from src.vectorstore.faiss_manager import FAISSManager

logger = logging.getLogger("DeltaIngestionWorker")

class DeltaIngestionWorker:
    def __init__(
        self,
        db: DatabaseManager,
        faiss_manager: FAISSManager,
        embedder: EmbeddingManager,
        chunker: TextChunker,
        max_batch_size: int = 50
    ):
        self.db = db
        self.faiss_manager = faiss_manager
        self.embedder = embedder
        self.chunker = chunker
        self.max_batch_size = max_batch_size

    def run_sync(self) -> Dict[str, Any]:
        """
        Executes a single delta ingestion cycle.
        Scans ONLY unindexed (PENDING) documents, chunks and embeds them,
        and appends them to FAISS. Existing documents are NOT re-processed.
        """
        pending_docs = self.db.get_pending_documents(limit=self.max_batch_size)
        if not pending_docs:
            logger.info("No pending documents found for delta ingestion.")
            return {"processed_docs": 0, "total_new_chunks": 0, "status": "idle"}

        logger.info(f"Discovered {len(pending_docs)} pending document(s) for delta ingestion.")
        total_new_chunks = 0
        successful_docs = 0
        failed_docs = 0

        for doc in pending_docs:
            doc_path = Path(doc.filepath)
            if not doc_path.exists():
                logger.error(f"File not found on disk: {doc.filepath}")
                self.db.update_document_status(
                    doc.doc_id,
                    status=DocumentStatus.FAILED,
                    error_message=f"File not found: {doc.filepath}"
                )
                failed_docs += 1
                continue

            try:
                # Mark document as PROCESSING
                self.db.update_document_status(doc.doc_id, status=DocumentStatus.PROCESSING)
                logger.info(f"Processing '{doc.filename}' (doc_id={doc.doc_id})...")

                # 1. Parse Document text and page structure
                sections = DocumentParser.parse(doc_path)
                if not sections or not any(s["text"].strip() for s in sections):
                    logger.warning(f"File '{doc.filename}' contains no extractable text.")
                    self.db.update_document_status(
                        doc.doc_id,
                        status=DocumentStatus.INDEXED,
                        chunk_count=0
                    )
                    successful_docs += 1
                    continue

                # 2. Chunk text while tracking FAISS ID continuity
                start_faiss_id = self.faiss_manager.total_vectors
                chunks = self.chunker.chunk_document(
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    sections=sections,
                    start_faiss_id=start_faiss_id
                )

                if not chunks:
                    logger.warning(f"No chunks generated for '{doc.filename}'.")
                    self.db.update_document_status(
                        doc.doc_id,
                        status=DocumentStatus.INDEXED,
                        chunk_count=0
                    )
                    successful_docs += 1
                    continue

                # 3. Generate dense vector embeddings for this document's chunks
                chunk_texts = [c.chunk_text for c in chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)

                # 4. Incrementally append new vectors to FAISS
                added_start_id = self.faiss_manager.add_vectors(embeddings)
                assert added_start_id == start_faiss_id, f"FAISS ID offset mismatch: expected {start_faiss_id}, got {added_start_id}"

                # 5. Persist chunk text and source metadata in SQLite Docstore
                self.db.add_chunks(chunks)

                # 6. Mark document as INDEXED
                self.db.update_document_status(
                    doc.doc_id,
                    status=DocumentStatus.INDEXED,
                    chunk_count=len(chunks)
                )

                total_new_chunks += len(chunks)
                successful_docs += 1
                logger.info(f"Successfully indexed '{doc.filename}' with {len(chunks)} chunks.")

            except Exception as e:
                logger.exception(f"Failed to process '{doc.filename}': {e}")
                self.db.update_document_status(
                    doc.doc_id,
                    status=DocumentStatus.FAILED,
                    error_message=str(e)
                )
                failed_docs += 1

        return {
            "processed_docs": successful_docs,
            "failed_docs": failed_docs,
            "total_new_chunks": total_new_chunks,
            "total_vectors_in_faiss": self.faiss_manager.total_vectors,
            "status": "completed"
        }
