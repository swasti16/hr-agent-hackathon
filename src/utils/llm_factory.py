"""
Factory functions for LLM, embeddings, and vector store.
Uses GitHub Models (OpenAI-compatible) as the LLM provider.
"""

from langchain_groq import ChatGroq
from src.utils.rate_limited_groq import RateLimitedChatGroq
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        api_key=settings.PROVIDER_API_KEY,
        model=settings.GENERATOR_MODEL,
        temperature=temperature
    )


def get_judge_llm() -> RateLimitedChatGroq:
    logger.info("Initializing judge LLM: %s", settings.JUDGE_MODEL)
    return RateLimitedChatGroq(
        api_key=settings.PROVIDER_API_KEY,
        model=settings.JUDGE_MODEL,
        temperature=0.0,
        delay_seconds=3.0,
        request_timeout=900.0
    )


def get_intent_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        api_key=settings.PROVIDER_API_KEY,
        model=settings.INTENT_MODEL,
        temperature=temperature
    )


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_vector_store(collection_name: str | None = None,
                     persist_directory: str | None = None):
    from langchain_chroma import Chroma
    if not collection_name:
        raise ValueError("Must provide collection_name")
    if not persist_directory:
        raise ValueError("Must provide persist_directory")
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory
    )


def get_llm_provider_name() -> str:
    return settings.LLM_PROVIDER.upper()
