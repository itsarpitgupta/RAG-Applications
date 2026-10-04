"""Graph builder for LangGraph workflow"""

# Try importing RAGNodes from src.nodes.nodes first, with a fallback if run from a different root
try:
    from src.nodes.nodes import RAGNodes
except ImportError:
    from nodes.nodes import RAGNodes

from langgraph.graph import StateGraph, END

# Import the shared state schema
try:
    from src.state.rag_state import RAGState
except ImportError:
    from state.rag_state import RAGState


class GraphBuilder:
    """
    This class connects our individual pieces into a coordinated workflow using LangGraph.
    It builds a computational state graph:
    1. First node ('retriever'): Fetches relevant chunks for the user's question.
    2. Second node ('responder'): Uses the LLM to write an answer using those chunks.
    3. End of graph: Delivers the completed answer back to the user.
    """

    def __init__(self, retriever, llm):
        """
        Initializes the graph builder with the necessary components.
        
        Args:
            retriever: The vector database retriever instance.
            llm: The language model instance.
        """
        # Create our node collection with our retriever and LLM
        self.nodes = RAGNodes(retriever, llm)
        # We start with the graph uncompiled until build() is called
        self.graph = None

    def build(self):
        """
        Constructs the LangGraph computational flow:
        - Registers our RAGState as the shared data model.
        - Adds the retrieval node and answer generation node.
        - Wires the connections (edges) between them.
        - Compiles the graph into an executable pipeline.
        
        Returns:
            The compiled, executable LangGraph workflow instance.
        """
        # Initialize a new graph structure tracking our RAGState schema
        builder = StateGraph(RAGState)

        # Register Node 1: 'retriever' runs the retrieve_docs function
        builder.add_node("retriever", self.nodes.retrieve_docs)

        # Register Node 2: 'responder' runs the generate_answer function
        builder.add_node("responder", self.nodes.generate_answer)

        # Designate the 'retriever' as the starting station of the workflow
        builder.set_entry_point("retriever")

        # Wire an arrow from 'retriever' to 'responder'
        builder.add_edge("retriever", "responder")

        # Wire an arrow from 'responder' to END (signifying workflow completion)
        builder.add_edge("responder", END)

        # Compile all the nodes and edges into an optimized, runnable state machine
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        """
        Executes the entire RAG pipeline for a given user question.
        
        Args:
            question: The plain-text question entered by the user.
            
        Returns:
            A dictionary representing the final state, containing:
            - 'question': The query asked.
            - 'retrieved_docs': The documents found.
            - 'answer': The LLM's final response text.
        """
        # If the graph hasn't been built yet, compile it now
        if self.graph is None:
            self.build()

        # Wrap the user's question in our initial RAGState object
        initial_state = RAGState(question=question)

        # Push the state through the graph and return the final output state
        return self.graph.invoke(initial_state)