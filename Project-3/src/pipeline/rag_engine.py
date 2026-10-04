import logging
from typing import Optional
from src.core.database import DatabaseManager
from src.core.models import RAGResponse, Citation
from src.vectorstore.embedder import EmbeddingManager
from src.vectorstore.faiss_manager import FAISSManager
from src.llm.grok_client import GrokClient

logger = logging.getLogger("RAGEngine")

class RAGEngine:
    def __init__(
        self,
        db: DatabaseManager,
        faiss_manager: FAISSManager,
        embedder: EmbeddingManager,
        grok_client: GrokClient
    ):
        self.db = db
        self.faiss_manager = faiss_manager
        self.embedder = embedder
        self.grok_client = grok_client

    def answer_query(self, query_text: str, top_k: int = 4) -> RAGResponse:
        """
        Executes full RAG retrieval and synthesis:
        1. Embeds query text.
        2. Searches FAISS for top-k nearest chunk vectors.
        3. Retrieves chunk metadata and source information from SQLite.
        4. Synthesizes response via Grok LLM with strict grounding.
        5. Returns structured answer with source citations.
        """
        clean_query = query_text.strip()
        if not clean_query:
            return RAGResponse(
                query=query_text,
                answer="Please enter a valid search query.",
                citations=[],
                total_retrieved=0,
                model_used=self.grok_client.model
            )

        if self.faiss_manager.total_vectors == 0:
            return RAGResponse(
                query=clean_query,
                answer="No documents are currently indexed in the vector database. Please upload and index documents first.",
                citations=[],
                total_retrieved=0,
                model_used=self.grok_client.model
            )

        # 1. Embed query
        query_vec = self.embedder.embed_query(clean_query)

        # 2. Similarity search in FAISS
        faiss_ids, scores = self.faiss_manager.search(query_vec, top_k=top_k)
        if not faiss_ids:
            return RAGResponse(
                query=clean_query,
                answer="No matching information could be found in the indexed documents.",
                citations=[],
                total_retrieved=0,
                model_used=self.grok_client.model
            )

        # 3. Retrieve chunk records
        chunks = self.db.get_chunks_by_faiss_ids(faiss_ids)

        # Build citations mapping FAISS IDs to scores
        id_to_score = dict(zip(faiss_ids, scores))
        citations = []
        for idx, chunk in enumerate(chunks, 1):
            score = id_to_score.get(chunk.faiss_id, 0.0)
            citations.append(
                Citation(
                    source_index=idx,
                    document_name=chunk.filename,
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    similarity_score=round(float(score), 4),
                    snippet=chunk.chunk_text[:280] + ("..." if len(chunk.chunk_text) > 280 else "")
                )
            )

        # 4. Generate grounded synthesis with Grok
        answer = self.grok_client.generate_rag_answer(clean_query, chunks)

        return RAGResponse(
            query=clean_query,
            answer=answer,
            citations=citations,
            total_retrieved=len(citations),
            model_used=self.grok_client.model if self.grok_client.is_configured else f"{self.grok_client.model} (mock/demo)"
        )
