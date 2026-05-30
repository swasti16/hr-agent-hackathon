"""
Central configuration for AI Testing Portfolio.
All settings loaded from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"


class Settings:
    
    # ======== LLM Provider ==================================
    DEFAULT_LLM_PROVIDER: str = os.getenv(
        "DEFAULT_LLM_PROVIDER", "groq"
    )

    # ======== Groq Settings ==================================
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_JUDGE_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    # GROQ_JUDGE_MODEL: str = "openai/gpt-oss-120b"

    # ======== Embedding Settings ==================================
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # ======== ChromaDB Settings ==================================
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", "./data/chromadb"
    )

    # ======== RAG Settings ==================================
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = CHUNK_SIZE // 10   # 10% overlap of CHUNK_SIZE
    TOP_K: int = 5

    # ======== Evaluation Thresholds ==================================
    MIN_FAITHFULNESS: float = 0.80
    MIN_CONTEXT_PRECISION: float = 0.75
    MIN_ANSWER_RELEVANCE: float = 0.75

    # ======== Safety Testing ==================================
    CONFIDENCE_THRESHOLD: float = 0.70

    def validate(self) -> None:
        """Validate required settings are present."""
        if self.DEFAULT_LLM_PROVIDER == "groq":
            assert self.GROQ_API_KEY, \
                "GROQ_API_KEY not set in .env"


settings = Settings()