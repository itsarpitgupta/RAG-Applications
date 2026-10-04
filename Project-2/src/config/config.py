import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root directory and load environment variables from the .env file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """
    This class holds all the configuration settings for our RAG application,
    including chunk sizes, default document sources, and API settings for language models.
    """

    # How large each piece of text should be (in characters) when splitting long articles
    CHUNK_SIZE: int = 500

    # How many characters should overlap between adjacent chunks to keep context across boundaries
    CHUNK_OVERLAP: int = 50

    # Locate the default text file containing source URLs to ingest
    DEFAULT_URL_FILE = BASE_DIR / "data" / "url.txt"
    if DEFAULT_URL_FILE.exists():
        # Read each line from url.txt, ignoring blank lines and lines starting with '#'
        with open(DEFAULT_URL_FILE, "r", encoding="utf-8") as f:
            DEFAULT_URLS = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    else:
        # Fallback list of default URLs if data/url.txt is missing
        DEFAULT_URLS = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        ]

    # Read API keys and chosen model names from environment variables
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    @classmethod
    def get_llm(cls):
        """
        Picks and returns an active Large Language Model instance based on available keys.
        
        It first checks if you have a GROQ_API_KEY set. If so, it uses Groq's high-speed
        cloud inference. If not, it falls back to OpenAI if an OPENAI_API_KEY is found.
        If neither key is present, it raises a friendly error asking you to provide one in .env.
        """
        # If Groq key is present, use ChatGroq
        if cls.GROQ_API_KEY:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=cls.GROQ_MODEL,
                groq_api_key=cls.GROQ_API_KEY,
                temperature=0.2,
            )
        # If OpenAI key is present, use ChatOpenAI
        elif cls.OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=cls.OPENAI_API_KEY,
                temperature=0.2,
            )
        # If no key was configured, alert the user
        else:
            raise ValueError(
                "No API key found. Please set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
            )