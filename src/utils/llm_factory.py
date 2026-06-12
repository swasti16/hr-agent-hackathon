"""
Factory functions for LLM, embeddings, and vector store.
Uses GitHub Models (OpenAI-compatible) as the LLM provider.
"""

from langchain_openai import ChatOpenAI
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.GITHUB_MODEL,
        api_key=settings.GITHUB_TOKEN,
        base_url=settings.GITHUB_BASE_URL,
        temperature=temperature
    )


def get_judge_llm() -> ChatOpenAI:
    logger.info("Initializing judge LLM: %s", settings.GITHUB_JUDGE_MODEL)
    return ChatOpenAI(
        model=settings.GITHUB_JUDGE_MODEL,
        api_key=settings.GITHUB_TOKEN,
        base_url=settings.GITHUB_BASE_URL,
        temperature=0.0
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
    return "GITHUB"
