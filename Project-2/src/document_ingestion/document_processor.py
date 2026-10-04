from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)


class DocumentProcessor:
    """
    This class is responsible for taking raw data—such as web articles, PDFs,
    or plain text files—loading their content into Python, and then cutting that text
    into smaller, bite-sized chunks so the search system can understand and index them easily.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Sets up the document processor with the desired chunking sizes.
        
        Args:
            chunk_size: Maximum character count for each slice of text.
            chunk_overlap: Number of characters to repeat between neighboring chunks so thoughts aren't cut mid-sentence.
        """
        # Save our desired chunk size and overlap values
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize the LangChain text splitter that smartly splits along paragraphs, sentences, and words
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """
        Fetches an article from the web given its URL, parses the HTML,
        and returns it as a LangChain Document containing the text and metadata.
        """
        # Create a web loader for this specific URL and download its text
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """
        Scans an entire directory on your computer, finds every PDF inside it,
        and loads all of their pages into Document objects.
        """
        # Point the directory loader to the target folder and read all PDFs
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Opens a single plain text (.txt) file using UTF-8 encoding
        and loads its entire text into a Document object.
        """
        # Read the plain text file directly from disk
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Reads a single specific PDF file from your disk and loads each of its pages as Documents.
        """
        # Load pages from this individual PDF file
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """
        A flexible helper method that takes one or more sources—which can be web URLs,
        a folder of PDFs, an individual PDF, or a text file—inspects what kind of source
        each one is, and uses the right loader to bring them all into a combined list of documents.
        """
        # If the caller passed a single string or Path instead of a list, wrap it in a list so we can loop over it
        if isinstance(sources, (str, Path)):
            sources = [sources]

        docs: List[Document] = []
        for src in sources:
            src_str = str(src).strip()

            # Case 1: If the source is an internet URL, load it using our web scraper
            if src_str.startswith("http://") or src_str.startswith("https://"):
                docs.extend(self.load_from_url(src_str))
            else:
                path = Path(src)
                # Case 2: If the source points to a folder, assume it contains PDFs and load all of them
                if path.is_dir():
                    docs.extend(self.load_from_pdf_dir(path))
                # Case 3: If it is an individual PDF file on disk
                elif path.is_file() and path.suffix.lower() == ".pdf":
                    docs.extend(self.load_from_pdf(path))
                # Case 4: If it is a plain text file on disk
                elif path.is_file() and path.suffix.lower() == ".txt":
                    docs.extend(self.load_from_txt(path))
                # If it doesn't match any known format, notify the developer with a clear error
                else:
                    raise ValueError(
                        f"Unsupported source type: {src}. "
                        "Please provide an HTTP(S) URL, a .txt file, a .pdf file, or a directory of PDFs."
                    )
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Takes full loaded documents and slices them into smaller text chunks
        using the configured chunk size and overlap so they fit nicely into embedding models.
        """
        return self.splitter.split_documents(documents)

    def process_urls(self, urls: Union[str, Path, List[str]]) -> List[Document]:
        """
        Runs the full ingestion pipeline:
        1. Checks if you provided a text file with URLs or an actual list of URLs.
        2. Loads all of the web pages into memory.
        3. Chunks them up into smaller pieces ready for the vector database.
        """
        # If the input is a file path pointing to a text file (like data/url.txt), read the URLs from that file
        if isinstance(urls, (str, Path)):
            path = Path(urls)
            if path.is_file():
                # Read each line from the file, skipping empty lines and comments
                with open(path, "r", encoding="utf-8") as f:
                    url_list = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
                # Load the web pages from those URLs
                docs = self.load_documents(url_list)
                # Split the downloaded pages into chunks and return them
                return self.split_documents(docs)
            urls = [str(urls)]

        # If already a list of URLs, load and split directly
        docs = self.load_documents(urls)
        return self.split_documents(docs)


if __name__ == "__main__":
    # Quick self-test to verify document loading and splitting when running this script directly
    processor = DocumentProcessor()
    path = Path("data/url.txt")
    docs = processor.process_urls(path)
    print(f"Processed {len(docs)} documents")
    for doc in docs[:3]:
        print(doc)
    if len(docs) > 3:
        print(f"... and {len(docs) - 3} more chunks.")