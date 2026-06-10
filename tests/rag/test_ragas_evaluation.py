"""
RAGAS Evaluation Suite for HR RAG Pipeline

Tests the RAG pipeline quality using 3 core RAGAS metrics:
  1. Faithfulness     — Did LLM use retrieved context?
  2. Context Precision — Were right chunks retrieved?
  3. Answer Relevance  — Does answer address the question?

"""

import pytest
import datetime
import os
import json
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
    # =============== Leave Policy ===============
    {
        "question": "How many earned leaves do employees get per calendar year?",
        "ground_truth": "Employees accrue 21 working days of earned leave per calendar year at a rate of 1.75 days per completed month."
    },
    {
        "question": "What is the maximum carry forward limit for earned leaves after the third year?",
        "ground_truth": "After the third year, the maximum carry forward limit is 30 days, which is the absolute cap."
    },
    {
        "question": "How long is maternity leave for an employee having her first child?",
        "ground_truth": "For the first two surviving children, maternity leave entitlement is 26 weeks of paid leave with a maximum of 8 weeks before delivery."
    },
    {
        "question": "What is the paternity leave entitlement and when must it be used?",
        "ground_truth": "Male employees are entitled to 5 continuous paid working days of paternity leave, which must be availed within 3 months of birth or adoption and cannot be split."
    },
    {
        "question": "What are the rules for comp off accumulation and validity?",
        "ground_truth": "Comp off maximum accumulation is 7 days per calendar year and must be used within 3 months of earning. It cannot be carried into the next calendar year, except comp offs earned in December which can be availed until January 31st."
    },
    {
        "question": "Can an employee on notice period apply for advance leave?",
        "ground_truth": "No. Employees serving their notice period are ineligible for advance leave."
    },
 
    # =============== Notice Period Policy ===============
    {
        "question": "What is the mandatory notice period at ABC Corporation?",
        "ground_truth": "The mandatory notice period is 60 days for all employees and cannot be waived under any circumstances."
    },
    {
        "question": "Can an employee use accumulated leave to shorten the notice period?",
        "ground_truth": "No. Accumulated privilege leave cannot reduce or offset the mandatory notice period."
    },
    {
        "question": "How is notice pay calculated if an employee does not serve the full notice period?",
        "ground_truth": "Notice pay is calculated as Monthly Gross Salary multiplied by 12, divided by 365, multiplied by the number of days not served. Includable components are basic pay, HRA, flexi, conveyance allowance, and standard monthly payouts. PF, ESI, NPS, superannuation, gratuity, and insurance premiums are excluded."
    },
 
    # =============== Code of Conduct ===============
    {
        "question": "What disciplinary action is taken on the first violation of the code of conduct?",
        "ground_truth": "The first violation results in a verbal warning documented in the employee file."
    },
    {
        "question": "What happens in case of gross misconduct?",
        "ground_truth": "Gross misconduct results in immediate termination of employment."
    },
    {
        "question": "What are the rules regarding use of company devices?",
        "ground_truth": "Company devices must be used for business purposes only. Personal use is prohibited and employees must not install unauthorized software on company devices."
    },
 
    # =============== Shift Allowance Policy ===============
    {
        "question": "What is the night shift allowance and what are the eligibility criteria?",
        "ground_truth": "Night shift allowance is INR 450 per day for working a full 8 hours between 9 PM and 9 AM. Employees on the standard general shift from 8:30 AM to 5 PM are not eligible."
    },
    {
        "question": "What is the payout for working on a company declared holiday?",
        "ground_truth": "Employees who work a full 8-hour shift on a company declared holiday receive twice their regular wages for that day."
    },
 
    # =============== Cross-policy questions ===============
    {
        "question": "If an employee resigns and has not served full notice, and also has an approved advance leave during notice, how does it affect their exit?",
        "ground_truth": "Employees on notice period are ineligible for advance leave. Additionally, if emergency medical leave is approved during notice, the final working day extends by the exact number of leaves used. If the employee fails to serve the full notice period, notice pay is calculated as Monthly Gross Salary multiplied by 12, divided by 365, multiplied by days not served."
    },
    {
        "question": "Can an employee working a night shift on a company declared holiday claim both allowances?",
        "ground_truth": "Yes. Concurrent night shift or odd shift allowances may also apply if the hours qualify, in addition to the company declared holiday payout of twice the regular wages."
    }
]


def create_ragas_score_json(faith_score, cprecision_score, relevancy_score):
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ragas_score_json = {
        "recorded_at": datetime.now().isoformat(),
        "judge_model": settings.GROQ_JUDGE_MODEL,
        "pipeline_model": settings.GROQ_MODEL,
        "scores": {
            "faithfulness": faith_score,
            "context_precision": cprecision_score,
            "answer_relevancy": relevancy_score
        },
        "dataset_size": len(GOLDEN_DATASET)
    }
    with open(os.path.join(project_folder, "reports","ragas_score.json"), "w") as f:
        json.dump(ragas_score_json, f, indent=2)


# ======== Pipeline Setup ==================================


@pytest.fixture(scope="module")
def pipeline():
    """Initialize RAG pipeline once for all tests."""
    p = HRRagPipeline()
    p.load_and_index()
    return p


@pytest.fixture(scope="module", autouse=True)
# JUDGE MODEL: gpt-4o-mini (same as pipeline model)
# Phi-4-mini-instruct was tested as an independent judge but produced nan
# for answer_relevancy metric. Root cause: Phi-4 generates synthetic questions
# in a format RAGAS 0.1.21 cannot parse during the embeddings similarity step.
# Using same model as judge produces slightly inflated scores (model agreeing
# with itself), but scores are stable and reproducible. A cross-model judge
# is the production recommendation.
def configure_ragas_judge():
    """Configure RAGAS to use Groq instead of OpenAI."""
    judge_llm = LangchainLLMWrapper(get_judge_llm())
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    faithfulness.llm = judge_llm
    context_precision.llm = judge_llm
    answer_relevancy.llm = judge_llm
    answer_relevancy.embeddings = judge_embeddings
    yield


# NOTE: autouse=True applies to all classes in this module.
# If adding a second test class, scope this fixture inside
# the class or make it non-autouse.
@pytest.fixture(autouse=True, scope="class")
def setup_scores(request, pipeline):
    dataset = build_ragas_dataset(pipeline)
    run_config = RunConfig(timeout=120, max_workers=1, max_retries=1)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        run_config=run_config
    )

    # create ragas_score.json
    create_ragas_score_json(results["faithfulness"],
                            results["context_precision"],
                            results["answer_relevancy"])

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
