import os
import logging
from typing import List, Optional
from src.core.models import ChunkRecord
from src.llm.prompt_templates import RAG_SYSTEM_PROMPT, build_rag_user_prompt

logger = logging.getLogger("GrokClient")

class GrokClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-beta",
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        self.api_key = api_key or os.getenv("GROK_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        # Clean model prefix if provided (e.g. groq:openai/gpt-oss-120b)
        if self.model and self.model.startswith("groq:"):
            self.model = self.model.replace("groq:", "", 1)

        # Auto-detect Groq keys (gsk_...) vs xAI Grok keys (xai-...)
        if self.api_key and self.api_key.startswith("gsk_"):
            if "api.x.ai" in self.base_url:
                self.base_url = "https://api.groq.com/openai/v1"
            if self.model in ("grok-beta", "grok-2", "grok"):
                self.model = "openai/gpt-oss-120b"

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                logger.info(f"Initialized LLM client with model '{self.model}' at {self.base_url}")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your_") and self._client)

    def generate_rag_answer(self, query: str, chunks: List[ChunkRecord]) -> str:
        """
        Sends grounded RAG prompt to Grok LLM and retrieves response with citations.
        If Grok API key is not configured, provides an extractive synthesis mock.
        """
        if not chunks:
            return "No relevant documents found to answer your query. Please upload and index documents first."

        user_prompt = build_rag_user_prompt(query, chunks)

        if self.is_configured:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Grok API call failed: {e}. Falling back to grounded contextual synthesis.")
                return f"[Grok API Connection Error: {e}]\n\n" + self._fallback_synthesis(query, chunks)

        # Standalone / Offline Grounded Synthesis
        return self._fallback_synthesis(query, chunks)

    def _fallback_synthesis(self, query: str, chunks: List[ChunkRecord]) -> str:
        """Grounded synthesis when Grok API key is not configured yet."""
        lead_chunk = chunks[0]
        summary_lines = [
            f"**Answer (Synthesized from {len(chunks)} source chunks)**:\n",
            f"Based on **{lead_chunk.filename}** [Source 1, Page {lead_chunk.page_number}]:",
            f"> \"{lead_chunk.chunk_text[:300]}...\"",
            "\n**Supporting Evidence**:"
        ]
        for idx, ch in enumerate(chunks[1:], 2):
            summary_lines.append(f"- **[Source {idx}]** *{ch.filename}* (Page {ch.page_number}): \"{ch.chunk_text[:180]}...\"")

        summary_lines.append("\n*(Note: Configure `GROK_API_KEY` in `.env` to enable full generative completions via xAI Grok)*")
        return "\n".join(summary_lines)
