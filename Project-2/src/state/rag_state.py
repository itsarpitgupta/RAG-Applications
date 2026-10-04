"""RAG state definition for LangGraph"""

from langchain_core.documents import Document
from typing import List
from pydantic import BaseModel


class RAGState(BaseModel):
    """
    This class defines the shared 'memory notebook' that gets passed from one node
    to the next in our LangGraph workflow.
    
    As the question moves through the system, each step records its findings here:
    1. 'question' stores what the user originally asked.
    2. 'retrieved_docs' holds the relevant document chunks found in our vector database.
    3. 'answer' contains the final response composed by the Large Language Model.
    """

    # The original query or question submitted by the user
    question: str

    # The list of document chunks that the retriever found most relevant to the question
    retrieved_docs: List[Document] = []

    # The final synthesized answer produced by the LLM
    answer: str = ""