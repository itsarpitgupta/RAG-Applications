import sqlite3
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
from src.core.models import DocumentRecord, DocumentStatus, ChunkRecord

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_tables(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    file_hash_sha256 TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    chunk_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    uploaded_at TEXT NOT NULL,
                    indexed_at TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash_sha256);")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    faiss_id INTEGER PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER,
                    chunk_text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);")

    def register_document(self, doc: DocumentRecord) -> bool:
        """Registers a new uploaded document. Returns False if duplicate hash exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_id, status FROM documents WHERE file_hash_sha256 = ?", (doc.file_hash_sha256,))
            existing = cursor.fetchone()
            if existing:
                return False

            cursor.execute("""
                INSERT INTO documents (
                    doc_id, filename, filepath, file_size_bytes, file_hash_sha256,
                    status, chunk_count, error_message, retry_count, uploaded_at, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.doc_id, doc.filename, doc.filepath, doc.file_size_bytes,
                doc.file_hash_sha256, doc.status.value, doc.chunk_count,
                doc.error_message, doc.retry_count,
                doc.uploaded_at.isoformat(),
                doc.indexed_at.isoformat() if doc.indexed_at else None
            ))
            return True

    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE file_hash_sha256 = ?", (file_hash,))
            row = cursor.fetchone()
            if row:
                return self._row_to_doc(row)
            return None

    def get_pending_documents(self, limit: int = 50) -> List[DocumentRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE status = ? ORDER BY uploaded_at ASC LIMIT ?",
                (DocumentStatus.PENDING.value, limit)
            )
            rows = cursor.fetchall()
            return [self._row_to_doc(r) for r in rows]

    def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: Optional[str] = None
    ):
        with self._get_connection() as conn:
            indexed_at_val = datetime.now(timezone.utc).isoformat() if status == DocumentStatus.INDEXED else None
            conn.execute("""
                UPDATE documents
                SET status = ?, chunk_count = ?, error_message = ?, indexed_at = COALESCE(?, indexed_at)
                WHERE doc_id = ?
            """, (status.value, chunk_count, error_message, indexed_at_val, doc_id))

    def get_all_documents(self) -> List[DocumentRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_doc(r) for r in rows]

    def add_chunks(self, chunks: List[ChunkRecord]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO chunks (
                    faiss_id, doc_id, filename, chunk_index, page_number,
                    chunk_text, token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    c.faiss_id, c.doc_id, c.filename, c.chunk_index,
                    c.page_number, c.chunk_text, c.token_count,
                    c.created_at.isoformat()
                )
                for c in chunks
            ])

    def get_chunks_by_faiss_ids(self, faiss_ids: List[int]) -> List[ChunkRecord]:
        if not faiss_ids:
            return []
        placeholders = ",".join("?" for _ in faiss_ids)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM chunks WHERE faiss_id IN ({placeholders})", faiss_ids)
            rows = cursor.fetchall()
            id_to_chunk = {r["faiss_id"]: self._row_to_chunk(r) for r in rows}
            # Return ordered by request sequence
            return [id_to_chunk[fid] for fid in faiss_ids if fid in id_to_chunk]

    def get_max_faiss_id(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(faiss_id), -1) as max_id FROM chunks")
            row = cursor.fetchone()
            return row["max_id"] if row else -1

    def get_stats(self) -> dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, count(*) as count FROM documents GROUP BY status")
            doc_counts = {r["status"]: r["count"] for r in cursor.fetchall()}
            cursor.execute("SELECT count(*) as total_chunks FROM chunks")
            total_chunks = cursor.fetchone()["total_chunks"]
            return {
                "documents": doc_counts,
                "total_chunks": total_chunks
            }

    def _row_to_doc(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            filename=row["filename"],
            filepath=row["filepath"],
            file_size_bytes=row["file_size_bytes"],
            file_hash_sha256=row["file_hash_sha256"],
            status=DocumentStatus(row["status"]),
            chunk_count=row["chunk_count"],
            error_message=row["error_message"],
            retry_count=row["retry_count"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else None
        )

    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            faiss_id=row["faiss_id"],
            doc_id=row["doc_id"],
            filename=row["filename"],
            chunk_index=row["chunk_index"],
            page_number=row["page_number"],
            chunk_text=row["chunk_text"],
            token_count=row["token_count"],
            created_at=datetime.fromisoformat(row["created_at"])
        )
