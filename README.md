# 🤖 HR Agent — ABC Corporation
### Microsoft Agents League Hackathon · Creative Apps Track

> An enterprise-grade HR policy assistant powered by **GitHub Models (gpt-4o-mini)** and **GitHub Copilot**, featuring a production-quality RAG pipeline, multi-layer safety shields, intent classification, and quantitative evaluation via RAGAS.

---

## 🎯 Problem Statement

HR teams at large organizations spend significant time answering repetitive policy questions — leave entitlements, shift allowances, notice periods, disciplinary procedures. Employees get inconsistent answers depending on who they ask.

**HR Agent** solves this by grounding every response in verified policy documents, blocking unsafe inputs before they reach the LLM, classifying intent to route queries correctly, and providing measurable answer quality scores — all in a single deployable application.

---

## ✨ What Makes This Different

| Capability | What It Does |
|---|---|
| 🛡️ **Safety Shield** | Rule-based injection + PII detection — zero LLM calls, instant blocking |
| 🧠 **Intent Classifier** | Routes queries to correct policy domain before RAG retrieval |
| 📚 **RAG Pipeline** | ChromaDB + HuggingFace embeddings grounded on 4 real HR policy PDFs |
| 🔗 **Reasoning Chain** | Every response exposes a structured debug trace (shield → intent → retrieval → answer) |
| 📊 **RAGAS Evaluation** | Faithfulness, Context Precision, Answer Relevancy scored against a 16-question golden dataset |
| 🎯 **Quality Dashboard** | Live session stats + offline RAGAS scores in a dedicated UI tab |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   Safety Shield                     │
│  (Regex + pattern matching — no LLM, <10ms)         │
│  Blocks: prompt injection, PII, jailbreak phrases   │
└────────────────────┬────────────────────────────────┘
                     │ PASS
                     ▼
┌─────────────────────────────────────────────────────┐
│              Greeting Short-Circuit                 │
│  (Rule-based — no LLM for hi/bye/thanks)            │
└────────────────────┬────────────────────────────────┘
                     │ Not a greeting
                     ▼
┌─────────────────────────────────────────────────────┐
│             Intent Classifier (LLM)                 │
│  Labels: LEAVE_QUERY | SHIFT_ALLOWANCE |            │
│          CONDUCT_POLICY | NOTICE_PERIOD |           │
│          OUT_OF_SCOPE | GREETING                    │
└────────────────────┬────────────────────────────────┘
                     │ In-scope intent
                     ▼
┌─────────────────────────────────────────────────────┐
│              RAG Retrieval                          │
│  ChromaDB · all-MiniLM-L6-v2 embeddings · Top-5     │
│  Query: original (not enriched) for precision       │
└────────────────────┬────────────────────────────────┘
                     │ Contexts retrieved
                     ▼
┌─────────────────────────────────────────────────────┐
│         Reasoning + Response (LLM)                  │
│  Structured prompt with policy context + history    │
│  Returns: answer + reasoning chain fields           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
              Response to User
         (with debug trace in UI)
```

---

## 📊 Evaluation Results

Evaluated on a **16-question golden dataset** built from actual HR policy documents.  
Judge model: `gpt-4.1-mini` (separate from pipeline model to avoid self-evaluation bias).

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness (Groundedness) | **0.916** | 0.80 | ✅ PASS |
| Context Precision | **0.942** | 0.75 | ✅ PASS |
| Answer Relevancy | **0.846** | 0.75 | ✅ PASS |

---

## 🛡️ Safety Test Coverage

`tests/safety/test_safety_shield.py` — 20 parametrized pytest cases:

- Direct prompt injection
- Typo-obfuscated injection (`ign0re`, `1gnore`)
- Base64-encoded injection
- Identity assumption attacks (`As an admin...`)
- Document poisoning attempts
- PII detection (email, phone, Aadhaar)
- False positive validation (legitimate queries pass through)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- GitHub account with Models access
- `GITHUB_TOKEN` with `models:read` permission

```bash
# 1. Clone
git clone https://github.com/swasti16/hr-agent-hackathon.git
cd hr-agent-hackathon

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install
pip install -r requirements.txt
pip install -e .

# 4. Configure
cp .env.example .env
# Add your GITHUB_TOKEN to .env

# 5. Run
python app.py
```

### Force reindex (when documents change)
```bash
python app.py --force_reindex   # or -f
```

---

## 🧪 Running Tests

```bash
# Safety shield tests (no LLM — instant)
pytest tests/safety/ -v

# Integration tests (~6 LLM calls)
pytest tests/integration/ -v

# Full suite
pytest -v
```

---

## 📁 Project Structure

```
hr-agent-hackathon/
├── app.py                          # Gradio UI — Chat + Quality Dashboard tabs
├── config/
│   └── settings.py                 # Central config — models, thresholds, paths
├── src/
│   ├── hr_rag_pipeline.py          # ChromaDB indexing + retrieval
│   └── agent/
│       ├── hr_agent.py             # Main orchestrator + session_stats
│       ├── intent_classifier.py    # LLM-based intent routing
│       ├── reasoning_chain.py      # Structured reasoning + response generation
│       └── safety_shield.py        # Rule-based injection/PII blocking
├── data/
│   └── hr_documents/               # 4 HR policy PDFs (source of truth)
├── reports/
│   └── ragas_scores.json           # Latest RAGAS evaluation output
└── tests/
    ├── safety/                     # 20 shield test cases
    ├── integration/                # Agent integration tests
    └── e2e/                        # End-to-end respond() tests
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| LLM | `gpt-4o-mini` via GitHub Models |
| Judge LLM | `gpt-4.1-mini` (RAGAS evaluation only) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector DB | ChromaDB |
| Framework | LangChain |
| RAG Evaluation | RAGAS |
| UI | Gradio |
| Testing | Pytest (parametrized fixtures) |
| Auth | `GITHUB_TOKEN` |

---

## 👩‍💻 Built By

**Swasti Shrivastava** 
GitHub: [@swasti16](https://github.com/swasti16)
