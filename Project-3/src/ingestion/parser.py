import hashlib
from pathlib import Path
from typing import List, Dict, Any

class DocumentParser:
    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Computes the SHA-256 hash of a file for deduplication."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def parse(cls, filepath: Path) -> List[Dict[str, Any]]:
        """
        Parses a document file and returns a list of page sections:
        [{"page_number": int, "text": str}]
        """
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            return cls._parse_pdf(filepath)
        elif ext == ".docx":
            return cls._parse_docx(filepath)
        elif ext in [".txt", ".md"]:
            return cls._parse_text(filepath)
        else:
            raise ValueError(f"Unsupported document format: {ext}")

    @staticmethod
    def _parse_pdf(filepath: Path) -> List[Dict[str, Any]]:
        from pypdf import PdfReader
        reader = PdfReader(str(filepath))
        pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page_number": idx + 1, "text": text})
        return pages

    @staticmethod
    def _parse_docx(filepath: Path) -> List[Dict[str, Any]]:
        import docx
        doc = docx.Document(str(filepath))
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        
        # Docx doesn't have native pages without rendering, so treat as single page or sectioned
        return [{"page_number": 1, "text": "\n".join(full_text)}]

    @staticmethod
    def _parse_text(filepath: Path) -> List[Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        return [{"page_number": 1, "text": text}]
