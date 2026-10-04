from enum import Enum
from typing import Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    DUPLICATE_SKIPPED = "DUPLICATE_SKIPPED"

class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    filepath: str
    file_size_bytes: int
    file_hash_sha256: str
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    indexed_at: Optional[datetime] = None

class ChunkRecord(BaseModel):
    faiss_id: int
    doc_id: str
    filename: str
    chunk_index: int
    page_number: Optional[int] = 1
    chunk_text: str
    token_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Citation(BaseModel):
    source_index: int
    document_name: str
    doc_id: str
    page_number: Optional[int]
    chunk_index: int
    similarity_score: float
    snippet: str

class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    total_retrieved: int
    model_used: str
