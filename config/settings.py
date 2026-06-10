"""
Central configuration for Project.
All settings loaded from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"


class Settings:

    # ======== GitHub Models ==================================
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_MODEL: str = "gpt-4o-mini"   # "gpt-4o-mini" or "gpt-4.1-mini" or "mistral-small-2503"
    GITHUB_JUDGE_MODEL: str = "gpt-4.1-mini"   # "gpt-4o-mini" or "gpt-4.1-mini"
    GITHUB_BASE_URL: str = "https://models.inference.ai.azure.com"

    # ======== Embedding Settings ==================================
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ======== ChromaDB Settings ==================================
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")

    # ======== RAG Settings ==================================
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = CHUNK_SIZE // 10
    TOP_K: int = 5

    # ======== Evaluation Thresholds ==================================
    MIN_FAITHFULNESS: float = 0.80
    MIN_CONTEXT_PRECISION: float = 0.75
    MIN_ANSWER_RELEVANCE: float = 0.75

    # ======== Safety Testing ==================================
    CONFIDENCE_THRESHOLD: float = 0.70

    def validate(self) -> None:
        assert self.GITHUB_TOKEN, "GITHUB_TOKEN not set in .env"


settings = Settings()
