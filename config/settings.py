"""
Central configuration for Project.
All settings loaded from .env file.
"""
import logging
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Mute telemetry warning spam
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)


class Settings:

    # ======== GitHub Models ==================================
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_JUDGE_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "GROQ")
    PROVIDER_API_KEY = GROQ_API_KEY
    GENERATOR_MODEL = GROQ_MODEL
    JUDGE_MODEL = GROQ_JUDGE_MODEL

    # ======== Embedding Settings ==================================
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ======== ChromaDB Settings ==================================
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")

    # ======== RAG Settings ==================================
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = CHUNK_SIZE // 10
    TOP_K: int = 5

    # ======== Logging ==================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "ERROR").upper()

    # ======== Evaluation Thresholds ==================================
    MIN_FAITHFULNESS: float = 0.80
    MIN_CONTEXT_PRECISION: float = 0.75
    MIN_ANSWER_RELEVANCE: float = 0.75

    # ======== Safety Testing ==================================
    CONFIDENCE_THRESHOLD: float = 0.70

    def validate(self) -> None:
        assert self.PROVIDER_API_KEY, "PROVIDER_API_KEY not set in .env"


settings = Settings()
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.ERROR))
