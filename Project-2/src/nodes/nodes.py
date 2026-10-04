"""LangGraph nodes for RAG workflow"""

# Import the shared state schema that tracks questions, retrieved documents, and answers
try:
    from src.state.rag_state import RAGState
except ImportError:
    from state.rag_state import RAGState


class RAGNodes:
    """
    This class contains the individual 'worker functions' (called nodes in LangGraph)
    that perform the two main jobs of our RAG pipeline:
    Job 1: Search and retrieve relevant document snippets from the vector store.
    Job 2: Hand those snippets to the LLM and generate a grounded, accurate answer.
    """

    def __init__(self, retriever, llm):
        """
        Stores the retriever and language model so every node has access to them.
        
        Args:
            retriever: The search tool that looks up matching chunks in the vector database.
            llm: The Large Language Model that reads the context and writes the response.
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Node 1: Document Retrieval
        
        This method takes the user's question from the incoming state, passes it
        to the vector retriever, collects the best matching text chunks, and returns
        an updated state containing both the original question and the retrieved documents.
        """
        # Look up documents matching the user's question
        docs = self.retriever.invoke(state.question)

        # Return a new state holding the question and the documents we just found
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Node 2: Answer Generation
        
        This method takes the retrieved text snippets, combines them into a neat
        context block, builds a prompt for the LLM instructing it to answer strictly
        based on that context, and saves the generated answer back into the state.
        """
        # Join together all the text contents of the retrieved chunks with line breaks
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Formulate a clear prompt showing the AI the exact context to base its answer on
        prompt = f"""Answer the question based on the context.
                    Context:
                    {context}
                    Question: {state.question}"""

        # Ask the LLM to generate the answer based on the prompt
        response = self.llm.invoke(prompt)

        # Return the completed state with the question, source documents, and the new answer
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )