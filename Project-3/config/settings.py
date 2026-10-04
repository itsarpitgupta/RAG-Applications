from pathlib import Path
import os
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig(BaseModel):
    name: str = "Incremental RAG System"
    version: str = "1.0.0"

class StorageConfig(BaseModel):
    raw_docs_dir: Path = BASE_DIR / "storage" / "raw_docs"
    faiss_index_dir: Path = BASE_DIR / "storage" / "faiss_index"
    db_path: Path = BASE_DIR / "storage" / "app.db"

class SchedulerConfig(BaseModel):
    enabled: bool = True
    mode: str = "interval"
    interval_hours: int = 1
    max_batch_size: int = 50
    max_retries: int = 3

class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] = ["\n\n", "\n", ". ", " ", ""]

class EmbeddingsConfig(BaseModel):
    provider: str = "huggingface"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize: bool = True

class VectorstoreConfig(BaseModel):
    type: str = "faiss"
    metric: str = "inner_product"
    index_file: str = "index.bin"
    meta_file: str = "index_meta.json"

class LLMConfig(BaseModel):
    provider: str = "grok"
    api_key_env: str = "GROK_API_KEY"
    api_key: str = ""
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-beta"
    temperature: float = 0.1
    max_tokens: int = 1024
    top_k_retrieval: int = 4

class SystemSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vectorstore: VectorstoreConfig = Field(default_factory=VectorstoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

def load_settings() -> SystemSettings:
    config_path = BASE_DIR / "config" / "config.yaml"
    data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    settings = SystemSettings(**data)
    
    # Override with environment variables if present
    if os.getenv("GROK_API_KEY"):
        settings.llm.api_key = os.getenv("GROK_API_KEY")
    if os.getenv("GROK_MODEL"):
        settings.llm.model = os.getenv("GROK_MODEL")
    if os.getenv("GROK_BASE_URL"):
        settings.llm.base_url = os.getenv("GROK_BASE_URL")
    if os.getenv("EMBEDDING_MODEL_NAME"):
        settings.embeddings.model_name = os.getenv("EMBEDDING_MODEL_NAME")
    if os.getenv("EMBEDDING_DEVICE"):
        settings.embeddings.device = os.getenv("EMBEDDING_DEVICE")
    if os.getenv("SCHEDULER_INTERVAL_HOURS"):
        settings.scheduler.interval_hours = int(os.getenv("SCHEDULER_INTERVAL_HOURS"))

    # Ensure storage directories exist
    settings.storage.raw_docs_dir.mkdir(parents=True, exist_ok=True)
    settings.storage.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)

    return settings
