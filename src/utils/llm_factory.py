"""
Factory functions for LLM, embeddings, and vector store.
"""

from config.settings import settings
from langchain_groq import ChatGroq
from src.utils.rate_limited_groq import RateLimitedChatGroq


def get_llm(temperature: float = 0.1):
    """
    Returns LLM based on DEFAULT_LLM_PROVIDER setting.
    Supports: groq
    """
    provider = settings.DEFAULT_LLM_PROVIDER.lower()

    if provider == "groq":
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=temperature
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Use 'groq'"
        )


def get_judge_llm():
    """
    Returns LLM based on DEFAULT_LLM_PROVIDER setting.
    Supports: groq
    """
    return RateLimitedChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_JUDGE_MODEL,
        temperature=0.0,
        delay_seconds=3.0,
        request_timeout=900.0
    )


def get_embeddings():
    """
    Returns local HuggingFace embedding model.
    Always local - free, no API key needed.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_vector_store(collection_name: str | None = None,
                     persist_directory: str | None = None):
    """
    Returns ChromaDB vector store with HuggingFace embeddings.
    """
    from langchain_chroma import Chroma
    if not collection_name:  # handles both None and empty string
        raise ValueError("Must provide collection_name to get_vector_store()")
    if not persist_directory:  # handles both None and empty string
        raise ValueError("Must provide persist_directory to get_vector_store()")
    return Chroma(
        collection_name=collection_name or settings.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory or settings.CHROMA_PERSIST_DIR
    )


def get_llm_provider_name() -> str:
    """Returns current LLM provider name for logging."""
    return settings.DEFAULT_LLM_PROVIDER.upper()
