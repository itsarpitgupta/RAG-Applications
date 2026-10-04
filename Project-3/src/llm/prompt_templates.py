from typing import List
from src.core.models import ChunkRecord

RAG_SYSTEM_PROMPT = """You are an intelligent, grounded RAG assistant powered by Grok.
Your role is to answer user questions strictly based on the provided reference contexts.

Follow these strict rules:
1. Base your answer ONLY on the provided sources. Do not assume or extrapolate facts not present in the text.
2. For EVERY key assertion or fact in your response, cite the source using inline tags such as [Source 1], [Source 2].
3. At the end of your response, list a 'Sources Used' section with the document name and page number.
4. If the provided sources do not contain the answer, explicitly state:
   "Based on the provided documents, I could not find information regarding this question."
"""

def format_context_for_llm(chunks: List[ChunkRecord]) -> str:
    """Formats retrieved chunks with clear source indices and metadata headers."""
    formatted_sections = []
    for idx, chunk in enumerate(chunks, 1):
        header = f"--- [Source {idx}] Document: '{chunk.filename}' (Page {chunk.page_number}, Chunk #{chunk.chunk_index}) ---"
        body = chunk.chunk_text.strip()
        formatted_sections.append(f"{header}\n{body}")
    
    return "\n\n".join(formatted_sections)

def build_rag_user_prompt(query: str, chunks: List[ChunkRecord]) -> str:
    context_str = format_context_for_llm(chunks)
    return f"""Context Information:
---------------------
{context_str}
---------------------

User Query: {query}

Please provide a detailed, accurate answer citing the relevant [Source X] references inline."""
