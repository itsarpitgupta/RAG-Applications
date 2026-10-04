import shutil
import tempfile
import numpy as np
from pathlib import Path
import pytest

from src.core.database import DatabaseManager
from src.core.models import DocumentRecord, DocumentStatus
from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.vectorstore.faiss_manager import FAISSManager
from src.vectorstore.embedder import EmbeddingManager
from src.ingestion.delta_worker import DeltaIngestionWorker

class MockEmbedder:
    """Fast deterministic embedder for unit testing."""
    def __init__(self, dimension=8):
        self.dimension = dimension
        self.normalize = True

    def embed_texts(self, texts, batch_size=32):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        # Generate deterministic normalized pseudo-embeddings
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 1000
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dimension).astype(np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / (norm if norm > 0 else 1.0))
        return np.array(vecs, dtype=np.float32)

    def embed_query(self, query: str):
        return self.embed_texts([query])

@pytest.fixture
def temp_env():
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "test.db"
    faiss_dir = temp_dir / "faiss"
    raw_docs = temp_dir / "raw_docs"
    raw_docs.mkdir()

    db = DatabaseManager(db_path)
    faiss_mgr = FAISSManager(index_dir=faiss_dir, dimension=8, metric="inner_product")
    embedder = MockEmbedder(dimension=8)
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    worker = DeltaIngestionWorker(db, faiss_mgr, embedder, chunker)

    yield {
        "temp_dir": temp_dir,
        "db": db,
        "faiss_mgr": faiss_mgr,
        "worker": worker,
        "raw_docs": raw_docs
    }

    shutil.rmtree(temp_dir, ignore_errors=True)

def test_incremental_ingestion_invariant(temp_env):
    db = temp_env["db"]
    faiss_mgr = temp_env["faiss_mgr"]
    worker = temp_env["worker"]
    raw_docs = temp_env["raw_docs"]

    # 1. Create first document
    file1 = raw_docs / "policy_v1.txt"
    file1.write_text("This is the first document with loan eligibility criteria. Customers must be over 21 years old.")
    
    doc1 = DocumentRecord(
        doc_id="doc-1",
        filename="policy_v1.txt",
        filepath=str(file1),
        file_size_bytes=file1.stat().st_size,
        file_hash_sha256=DocumentParser.compute_sha256(file1),
        status=DocumentStatus.PENDING
    )
    assert db.register_document(doc1) is True

    # Run delta ingestion
    res1 = worker.run_sync()
    assert res1["processed_docs"] == 1
    initial_chunks = res1["total_new_chunks"]
    assert faiss_mgr.total_vectors == initial_chunks
    assert db.get_document_by_hash(doc1.file_hash_sha256).status == DocumentStatus.INDEXED

    # 2. Add second document (Simulate adding 101st document)
    file2 = raw_docs / "interest_rates.txt"
    file2.write_text("Interest rate for personal loans is 10.5 percent per annum. Repayment tenure is up to 5 years.")
    
    doc2 = DocumentRecord(
        doc_id="doc-2",
        filename="interest_rates.txt",
        filepath=str(file2),
        file_size_bytes=file2.stat().st_size,
        file_hash_sha256=DocumentParser.compute_sha256(file2),
        status=DocumentStatus.PENDING
    )
    assert db.register_document(doc2) is True

    # Run delta ingestion again
    res2 = worker.run_sync()
    assert res2["processed_docs"] == 1
    doc2_chunks = res2["total_new_chunks"]
    
    # Invariant: Total vectors = initial_chunks + doc2_chunks
    assert faiss_mgr.total_vectors == initial_chunks + doc2_chunks

    # Verify doc1 was NOT re-processed
    doc1_refreshed = db.get_document_by_hash(doc1.file_hash_sha256)
    assert doc1_refreshed.chunk_count == initial_chunks

def test_deduplication(temp_env):
    db = temp_env["db"]
    raw_docs = temp_env["raw_docs"]

    file1 = raw_docs / "sample.txt"
    file1.write_text("Hello world text for testing hash deduplication.")
    file_hash = DocumentParser.compute_sha256(file1)

    doc1 = DocumentRecord(
        doc_id="doc-a",
        filename="sample.txt",
        filepath=str(file1),
        file_size_bytes=file1.stat().st_size,
        file_hash_sha256=file_hash,
        status=DocumentStatus.PENDING
    )
    assert db.register_document(doc1) is True

    # Duplicate registration attempt with same file hash
    doc2 = DocumentRecord(
        doc_id="doc-b",
        filename="sample_copy.txt",
        filepath=str(file1),
        file_size_bytes=file1.stat().st_size,
        file_hash_sha256=file_hash,
        status=DocumentStatus.PENDING
    )
    assert db.register_document(doc2) is False
