from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Try importing HuggingFace embeddings from the dedicated langchain_huggingface package first,
# with a fallback to langchain_community if needed.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStore:
    """
    This class manages converting text chunks into mathematical numbers (embeddings)
    and storing them inside a FAISS vector database so we can find relevant passages
    almost instantaneously when a user asks a question.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the vector store and loads the local HuggingFace embedding model.
        
        Args:
            model_name: The name of the transformer model to use. By default, we use
                        'all-MiniLM-L6-v2' because it is lightweight, fast, runs locally,
                        and does not require any paid API keys.
        """
        # Load the sentence transformer embedding model
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        
        # We start with empty references for the vector index and retriever until documents are indexed
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Takes a list of sliced text chunks, converts their words into vector embeddings
        using our HuggingFace model, and builds a FAISS index in memory.
        
        Args:
            documents: The list of chunked documents created by DocumentProcessor.
        """
        # Compute embeddings for all text chunks and assemble the FAISS vector index
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        
        # Build a retriever interface on top of the vector index for easy querying
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Returns a LangChain retriever object that other components (like LangGraph) can use
        to fetch matching chunks.
        
        Args:
            search_kwargs: Optional dictionary of search settings, such as {'k': 4}
                           to decide how many top matching documents to return.
        """
        # Make sure the user created the vector database before asking for a retriever
        if self.vectorstore is None:
            raise ValueError(
                "Vector store is not initialized yet. Please call create_vectorstore() with documents first."
            )
        
        # Default to retrieving the 4 most relevant chunks if no custom number was specified
        kwargs = search_kwargs or {"k": 4}
        return self.vectorstore.as_retriever(search_kwargs=kwargs)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Searches the FAISS index to find the top 'k' most relevant text chunks
        for a given user question or query string.
        
        Args:
            query: The user's question or search phrase.
            k: The maximum number of relevant chunks to return.
        """
        # Verify the vector store is ready
        if self.vectorstore is None:
            raise ValueError(
                "Vector store is not initialized yet. Please call create_vectorstore() first."
            )
        
        # Perform similarity search and return the closest matching documents
        return self.vectorstore.similarity_search(query, k=k)
