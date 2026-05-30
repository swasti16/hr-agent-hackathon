"""
RAGAS Evaluation Suite for HR RAG Pipeline

Tests the RAG pipeline quality using 3 core RAGAS metrics:
  1. Faithfulness     — Did LLM use retrieved context?
  2. Context Precision — Were right chunks retrieved?
  3. Answer Relevance  — Does answer address the question?

"""

import os
import sys
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
import logging
from src.utils.llm_factory import get_judge_llm, get_embeddings
from src.hr_rag_pipeline import HRRagPipeline
from config.settings import settings

# Mute telemetry warning spam
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)


# ======== Golden Test Dataset ==================================
# These are hand-crafted Q&A pairs with known correct answers.
# Ground truth comes directly from our HR documents.

GOLDEN_DATASET = [
    {
        "question": "How many annual leave days do employees get?",
        "ground_truth": "Full-time employees are entitled to 15 days of annual leave per calendar year."
    },
    {
        "question": "What happens after the second disciplinary violation?",
        "ground_truth": "The second violation results in a written warning with a performance improvement plan."
    },
    {
        "question": "How many sick leave days are employees entitled to?",
        "ground_truth": "Employees are entitled to 10 days of paid sick leave per year."
    },
    {
        "question": "How long is maternity leave?",
        "ground_truth": "Female employees are entitled to 26 weeks of paid maternity leave."
    },
    {
        "question": "What is the paternity leave entitlement?",
        "ground_truth": "Male employees are entitled to 5 days of paid paternity leave within 30 days of the child's birth."
    },
    {
        "question": "What is the notice period for employees with less than 2 years of service?",
        "ground_truth": "The notice period is 30 days for employees with less than 2 years of service."
    },
    {
        "question": "What is the notice period for employees with more than 2 years of service?",
        "ground_truth": "The notice period is 60 days for employees with 2 or more years of service."
    },
    {
        "question": "How far in advance should employees apply for leave?",
        "ground_truth": "Employees must apply for leave at least 2 weeks in advance through the HR portal."
    }
]

# ======== Pipeline Setup ==================================

@pytest.fixture(scope="module")
def pipeline():
    """Initialize RAG pipeline once for all tests."""
    p = HRRagPipeline()
    p.load_and_index()
    return p


@pytest.fixture(scope="module", autouse=True)
def configure_ragas_judge():
    """Configure RAGAS to use Groq instead of OpenAI."""
    judge_llm = LangchainLLMWrapper(get_judge_llm())
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    faithfulness.llm = judge_llm
    context_precision.llm = judge_llm
    answer_relevancy.llm = judge_llm
    answer_relevancy.embeddings = judge_embeddings
    
    yield

@pytest.fixture(autouse=True, scope="class")
def setup_scores(request, pipeline):
    """
    Runs evaluate() ONCE.
    All individual tests read from cached results.
    Zero extra API calls.
    """       
    # ONE evaluate() call — all metrics together
    dataset = build_ragas_dataset(pipeline)
    run_config = RunConfig(timeout=900, max_workers=1, max_retries=3)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        run_config=run_config
    )
    
    # Store on class — all tests read from here
    request.cls.scores = {
        "faithfulness": results["faithfulness"],
        "context_precision": results["context_precision"],
        "answer_relevancy": results["answer_relevancy"]
    }

    yield


# ======== Helper: Build RAGAS Dataset ==================================


def build_ragas_dataset(pipeline: HRRagPipeline) -> Dataset:
    """
    Run all golden questions through pipeline and
    build a dataset for RAGAS evaluation.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\nRunning golden dataset through pipeline...")
    for item in GOLDEN_DATASET:
        result = pipeline.ask(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])
        print(f"{item['question'][:60]}...")

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

# ======== RAGAS Evaluation Tests ==================================


class TestRagasMetrics:
    """
    RAGAS evaluation tests for HR RAG pipeline.
    Each test checks one quality dimension.
    """
    # Individual tests read from cached scores — zero extra API calls
    def test_faithfulness_score(self):
        score = self.scores["faithfulness"]
        print(f"\n   Faithfulness: {score:.4f} "
            f"(threshold: {settings.MIN_FAITHFULNESS})")
        assert score >= settings.MIN_FAITHFULNESS, \
            f"Faithfulness {score:.4f} < {settings.MIN_FAITHFULNESS}"

    def test_context_precision_score(self):
        score = self.scores["context_precision"]
        print(f"\n   Context Precision: {score:.4f} "
            f"(threshold: {settings.MIN_CONTEXT_PRECISION})")
        assert score >= settings.MIN_CONTEXT_PRECISION, \
            f"Context Precision {score:.4f} < {settings.MIN_CONTEXT_PRECISION}"

    def test_answer_relevancy_score(self):
        score = self.scores["answer_relevancy"]
        print(f"\n   Answer Relevancy: {score:.4f} "
            f"(threshold: {settings.MIN_ANSWER_RELEVANCE})")
        assert score >= settings.MIN_ANSWER_RELEVANCE, \
            f"Answer Relevancy {score:.4f} < {settings.MIN_ANSWER_RELEVANCE}"
