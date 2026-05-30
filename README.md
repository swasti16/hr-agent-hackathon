# AI Testing Portfolio

A portfolio project demonstrating modern AI/LLM testing techniques for:
- Retrieval-Augmented Generation (RAG) systems
- Agentic AI workflows
- LLM safety testing
- RAG evaluation using RAGAS
- Embedding quality validation

## Project Goal

Traditional software testing approaches are not sufficient for validating LLM-based systems due to challenges such as hallucinations, prompt injection, semantic correctness, and retrieval quality.

This project was built to explore practical testing strategies for modern AI applications including:
- RAG pipeline validation
- LLM response evaluation
- Embedding quality checks
- AI safety testing
- Agent behavior validation

The repository focuses on applying QA engineering principles to GenAI systems using automated evaluation techniques and reproducible test workflows.

## Key Features

- HR-policy based RAG pipeline
- Automated RAGAS evaluation
- Embedding similarity validation
- Prompt injection testing
- Modular evaluation architecture
- Pytest-based validation framework

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (llama-3.3-70b-versatile)|
| Embeddings | Sentence Transformers|
| Vector DB | ChromaDB |
| Framework | LangChain |
| RAG Evaluation | RAGAS |
| Testing | Pytest |

## Project Structure

```
ai-testing-portfolio/
├── config/
│   ├── settings.py                 # Central configuration
│   └── agents/
│       └── hr_agent_config.py             # HR agent configuration/logic
│
├── src/
│   ├── rag_pipeline.py             # Main RAG pipeline implementation
│   ├── hr_rag_pipeline.py          # HR-specific RAG workflow
│   └── utils/
│       └── llm_factory.py          # LLM & embedding model factory
│
├── data/
│   └── hr_documents/               # HR policy documents for RAG
│       ├── leave_policy.txt
│       └── code_of_conduct.txt
│
├── tests/
│   ├── rag/
│   │   ├── test_ragas_evaluation.py
│   │   ├── test_embedding_quality.py
│   │   ├── test_document_indexing.py
│   │   └── test_hr_rag_pipeline.py
│   │
│   ├── agent/                      # Agent testing modules
│   │
│   ├── safety/                     # Prompt injection/jailbreak testing
│   │
│   ├── integration/                # Integration-level tests
│   │
│   ├── e2e/                        # End-to-end workflow tests
│   │
│   └── unit/                       # Unit tests
│
├── concepts/
│   ├── GenAI_Session1_Notes.md
│   ├── GenAI_Session2_Notes.md
│   ├── GenAI_Session3_Notes.md
│   └── GenAI_Session4_Notes.md
│
├── environment_sanity_test.py                   # Shared test setup
├── requirements.txt                # Project dependencies
├── .env.example                    # Environment variable template
├── README.md
└── .gitignore
```

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-testing-portfolio.git
cd ai-testing-portfolio

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install project as package
pip install -e .

# 5. Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 6. Verify setup
python environment_sanity_test.py
```

## Environment Variables

```bash
# Required
GROQ_API_KEY=your_groq_key        # Get from console.groq.com

```

## Test Coverage

| Module | Tests | What's Tested |
|---|---|---|
| RAG Pipeline | RAGAS Metrics | Faithfulness, Context Relevance, Answer Relevance |
| Embeddings | Quality + Consistency | Similarity scores, dimensions, model consistency |
| Indexing | Document Pipeline | Chunk count, metadata, retrieval after indexing |
| Agent | Behavior + Tools | Tool selection, infinite loop, parameter validation |
| Safety | Security Suite | Prompt injection, jailbreak, PII leakage, guardrails |

## Key Concepts Demonstrated

- **RAG Pipeline Testing** — RAGAS evaluation framework
- **Embedding Quality** — Cosine similarity, dimension verification
- **Hallucination Detection** — Faithfulness scoring
- **Prompt Injection Testing** — Direct and indirect injection
- **Agent Behavior Testing** — Tool selection, loop detection
- **Memory Isolation** — Cross-session data leakage tests
- **Black Box Safety Testing** — Guardrail validation suite
