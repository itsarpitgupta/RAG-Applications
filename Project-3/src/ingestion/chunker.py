from typing import List, Dict, Any
from src.core.models import ChunkRecord
from datetime import datetime, timezone

class TextChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_document(
        self,
        doc_id: str,
        filename: str,
        sections: List[Dict[str, Any]],
        start_faiss_id: int
    ) -> List[ChunkRecord]:
        """
        Takes parsed sections [{"page_number": int, "text": str}]
        and splits them into overlapping chunks with metadata.
        """
        all_chunks: List[ChunkRecord] = []
        current_faiss_id = start_faiss_id
        chunk_index = 0

        for section in sections:
            page_num = section.get("page_number", 1)
            raw_text = section.get("text", "")
            if not raw_text.strip():
                continue

            text_splits = self._recursive_split(raw_text, self.chunk_size, self.chunk_overlap)
            for split_text in text_splits:
                cleaned = split_text.strip()
                if not cleaned:
                    continue

                chunk_rec = ChunkRecord(
                    faiss_id=current_faiss_id,
                    doc_id=doc_id,
                    filename=filename,
                    chunk_index=chunk_index,
                    page_number=page_num,
                    chunk_text=cleaned,
                    token_count=len(cleaned.split()),
                    created_at=datetime.now(timezone.utc)
                )
                all_chunks.append(chunk_rec)
                current_faiss_id += 1
                chunk_index += 1

        return all_chunks

    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits text recursively using hierarchical separators."""
        if len(text) <= chunk_size:
            return [text]

        splits = []
        # Find the best separator
        separator = self.separators[-1]
        for sep in self.separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break

        if separator:
            sub_texts = text.split(separator)
        else:
            sub_texts = list(text)

        current_chunk = []
        current_len = 0

        for s in sub_texts:
            s_len = len(s) + (len(separator) if current_chunk else 0)
            if current_len + s_len > chunk_size and current_chunk:
                merged = separator.join(current_chunk)
                splits.append(merged)
                # Keep overlap from the end of the current chunk
                overlap_chars = 0
                overlap_chunk = []
                for prev in reversed(current_chunk):
                    if overlap_chars + len(prev) <= chunk_overlap:
                        overlap_chunk.insert(0, prev)
                        overlap_chars += len(prev)
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = sum(len(x) for x in current_chunk) + (len(separator) * (len(current_chunk) - 1) if current_chunk else 0)

            current_chunk.append(s)
            current_len += len(s)

        if current_chunk:
            splits.append(separator.join(current_chunk))

        return splits
