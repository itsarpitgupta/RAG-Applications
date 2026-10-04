"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
try:
    from src.state.rag_state import RAGState
except ImportError:
    from state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tools for external fallback knowledge
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """
    This class is an advanced alternative node runner that uses an autonomous 'ReAct' Agent.
    Instead of just doing a single static retrieval, this agent has a toolkit:
    - A 'retriever' tool to look up facts in your local indexed documents.
    - A 'wikipedia' tool to look up general knowledge if your local docs don't have the answer.
    The agent dynamically decides which tool to call before answering.
    """

    def __init__(self, retriever, llm):
        """
        Initializes the node with access to the document retriever and the LLM.
        
        Args:
            retriever: The vector store search tool.
            llm: The reasoning model that drives the ReAct agent loop.
        """
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # We will construct the ReAct agent on demand the first time it is needed

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Node 1: Direct document retrieval
        Fetches the most relevant chunks from the local vector database and attaches them to state.
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self) -> List[Tool]:
        """
        Creates the set of tools that the agent is allowed to use.
        
        Tool 1: 'retriever' - fetches custom passages from your indexed corpus.
        Tool 2: 'wikipedia' - queries the public Wikipedia encyclopedia for broader world knowledge.
        """

        # Define the function the agent triggers when it calls the local retriever
        def retriever_tool_fn(query: str) -> str:
            # Query the vector store for matching documents
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            
            # Combine the top matching snippets with titles and numbers
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        # Package the retriever function as a standard LangChain Tool
        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages and specific facts from indexed corpus documents.",
            func=retriever_tool_fn,
        )

        # Set up a Wikipedia query client that grabs the top 3 matching summaries
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge, definitions, and broad concepts.",
            func=wiki.run,
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """
        Assembles the ReAct agent using the LLM and the tools defined in _build_tools.
        Provides a guiding system prompt so the agent knows to check local documents first.
        """
        # Gather our tools
        tools = self._build_tools()

        # Instruct the agent on its role and priority order
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )

        # Build the prebuilt LangGraph ReAct agent
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Node 2: Generates an answer using the ReAct agent.
        The agent reasons about the question, selects the right tool(s), executes them,
        and produces the final synthesized response.
        """
        # Lazily instantiate the agent if it hasn't been created yet
        if self._agent is None:
            self._build_agent()

        # Send the user's question into the agent's message loop
        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        # Extract the final response message from the agent conversation history
        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        # Return the completed state with the agent's answer
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )
