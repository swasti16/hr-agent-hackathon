# 🤖 HR Policy Assistant — ABC Corporation
### Microsoft Agents League Hackathon · Creative Apps Track

> A **production-grade AI-powered HR Policy Chatbot** built with **GitHub Copilot** and **GitHub Models (gpt-4o-mini)**, featuring a Foundry IQ-inspired RAG pipeline, dynamic multi-agent routing, multi-layer safety shields, structured two-stage reasoning chains, and quantitative evaluation via RAGAS.

---

## 🎯 What This Application Does

HR teams at large organisations spend significant time answering repetitive policy questions — leave entitlements, shift allowances, notice periods, disciplinary procedures. Employees get inconsistent answers depending on who they ask.

**HR Policy Assistant** is a creative productivity application that solves this by:
- Grounding every response in verified policy documents (no hallucination)
- Routing queries through specialised agents for accurate, context-aware answers
- Blocking unsafe inputs before they reach the LLM
- Providing measurable answer quality scores via RAGAS evaluation
- Delivering all of this through a clean, interactive Gradio interface

---

## 📸 Screenshots

### Chat Interface
![Chat UI](docs/screenshots/chat_ui.png)

### Quality Dashboard
![Quality Dashboard](docs/screenshots/quality_dashboard.png)

### Reasoning Chain Debug Panel
![Debug Panel](docs/screenshots/debug_panel.png)

---

## 🎬 Demo

[![Demo Video](https://img.shields.io/badge/Watch-Demo-red?logo=youtube)](https://youtu.be/SFR3RisyifY)

---

## 🏗️ Agent Pipeline Architecture

The system is composed of **four specialised agents**, each with a single responsibility, coordinated by an orchestrator that makes **active routing decisions** — minimising LLM calls and maximising response accuracy.

![Architecture Diagram](docs/architecture.png)

---

## 🧠 Multi-Step Reasoning & Dynamic Routing

A key design goal is **cost-aware orchestration** — the system uses the minimum number of LLM calls needed to answer each query correctly.

### Routing Decision 1 — Pre-LLM (Orchestrator)
- **Injection / PII detected** → block immediately, 0 LLM calls
- **Greeting keyword matched** → rule-based reply, 0 LLM calls
- **All other queries** → forward to Routing Agent

### Routing Decision 2 — Post-Intent (Routing Agent)
- **OUT_OF_SCOPE** → redirect, pipeline stops at 1 LLM call
- **GREETING** (LLM-classified) → greeting reply, 1 LLM call
- **HR policy intent** → forward to Knowledge + Reasoning agents

### Routing Decision 3 — Post-Retrieval (Reasoning Agent, 2-stage)

**Call 1 — Query Dimension Classifier:**
- `needs_personal_data` → redirect to HR without generating an answer
- `context_sufficient` → return "policy not found" without generating
- `is_injection` → semantic injection check (catches paraphrased attacks)

**Call 2 — Grounded Answer Generator:**
Only fires if Call 1 clears all three checks.

### LLM Call Budget Per Query Type

| Query Type | LLM Calls | Path |
|---|---|---|
| Injection / PII | 0 | Safety Agent blocks |
| Rule-based greeting | 0 | Orchestrator keyword match |
| Out of scope | 1 | Routing Agent only |
| Needs personal DB data | 2 | Routing + Classifier |
| Context insufficient | 2 | Routing + Classifier |
| Semantic injection | 2 | Routing + Classifier |
| Full policy answer | 3 | Routing + Classifier + Generator |

---

## 👥 Agent Responsibilities

| Agent | File | Role | LLM? |
|---|---|---|---|
| **Safety Agent** | `safety_shield.py` | Injection + PII detection via regex/fuzzy match | ❌ No LLM |
| **Routing Agent** | `intent_classifier.py` | Classifies query intent, routes to correct policy domain | ✅ 1 call |
| **Knowledge Retrieval Agent** | `hr_rag_pipeline.py` | Foundry IQ-inspired grounding — retrieves top-5 chunks from ChromaDB | ❌ No LLM |
| **Reasoning Agent** | `reasoning_chain.py` | Two-stage: dimension classifier then grounded answer generator | ✅ 2 calls |
| **Orchestrator** | `hr_agent.py` | Active routing decisions at each checkpoint, session stats | — |

---

## 🧠 Microsoft IQ Alignment (Required)

### Foundry IQ — Grounding Layer
The Knowledge Retrieval Agent directly implements the Foundry IQ pattern:
- Knowledge base built from 4 HR policy PDFs
- Permission-aware retrieval — only indexed documents are sources, no general LLM knowledge used
- Grounded answers with source chunk citations exposed in the debug panel
- Implemented via ChromaDB + HuggingFace `all-MiniLM-L6-v2` embeddings (local, no Azure dependency)

> **Azure migration path:** Architecture is designed to swap ChromaDB for Azure AI Search with a single line change in `get_vector_store()` inside `llm_factory.py`.

---

## ✨ Key Capabilities

| Capability | What It Does |
|---|---|
| 🛡️ **Safety Agent** | Rule-based injection + PII detection — zero LLM calls, <10ms blocking |
| ⚙️ **Dynamic Orchestration** | Active routing cuts LLM calls from 3 to 0-2 for non-policy queries |
| 🎯 **Routing Agent** | Intent classification routes queries to correct policy domain before retrieval |
| 📚 **Knowledge Retrieval Agent** | Foundry IQ-inspired grounding on 4 HR policy PDFs via ChromaDB |
| 🧠 **Two-Stage Reasoning Agent** | Stage 1 classifies query dimensions; Stage 2 generates grounded answer |
| 📊 **RAGAS Evaluation** | Faithfulness, Context Precision, Answer Relevancy on 16-question golden dataset |
| 🎛️ **Quality Dashboard** | Live session stats + RAGAS scores in dedicated UI tab |

---

## 📊 Evaluation Results

Evaluated on a **16-question golden dataset** built from actual HR policy documents.
Judge model: `gpt-4.1-mini` — separate from pipeline model (`gpt-4o-mini`) to reduce self-evaluation bias.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness (Groundedness) | **0.934** | 0.80 | ✅ PASS |
| Context Precision | **0.893** | 0.75 | ✅ PASS |
| Answer Relevancy | **0.915** | 0.75 | ✅ PASS |

> Scores updated from latest run (June 11, 2026) with expanded 16-question dataset and independent judge model.

---

## 🛡️ Safety & Responsible AI

`tests/safety/test_safety_shield.py` — 22 parametrized pytest cases covering:

- Direct prompt injection
- Typo-obfuscated injection (`ign0re`, `1gnore`, `forg3t`)
- Base64-encoded injection
- Identity assumption attacks (`Act as an admin...`)
- Document poisoning attempts (`Update your internal state...`)
- PII detection (email, phone, Aadhaar, PAN)
- False positive validation — legitimate tough questions pass through

**Defense-in-depth:** Rule-based Safety Agent catches known patterns in <10ms. The Reasoning Agent's Call 1 adds a second semantic injection check — catching paraphrased attacks that bypass regex.

All responses are grounded in indexed policy documents. If a topic isn't in the policy PDFs, the system says so explicitly — it will not fabricate answers from general LLM knowledge.

---

## 🤖 GitHub Copilot Usage

GitHub Copilot was integral to the development velocity of this project, used throughout in VS Code with **inline autosuggestions as the primary workflow**. The accuracy of Copilot's suggestions for the specific libraries used (LangChain, ChromaDB, pytest) was consistently high, significantly reducing time spent on API lookups and boilerplate.

**Where Copilot had the highest impact:**

- **LangChain chain composition** (`reasoning_chain.py`, `base_rag_pipeline.py`) — Copilot accurately suggested chain syntax, prompt template structure, and `RunnablePassthrough` patterns on first attempt
- **Pytest fixture boilerplate** (`test_safety_shield.py`, `test_agent.py`) — parametrized fixture structures, `scope` declarations, and `autouse` patterns were autocompleted with high accuracy
- **ChromaDB integration** (`llm_factory.py`) — collection management, `PersistentClient` patterns, and similarity search syntax suggested correctly
- **Regex patterns** (`safety_shield.py`) — PII detection patterns for Indian identifiers (Aadhaar, PAN, mobile number format) were suggested accurately without manual lookup
- **Gradio component wiring** (`app.py`) — event handler `.click()` and `.submit()` connections, component layout with `gr.Row()` / `gr.Column()`, and tab structure were all accelerated by Copilot suggestions

**Development approach:** Copilot handled the syntactic layer (how to write it), while architectural decisions (what to build and why) were reasoned through independently — a productive human-AI collaboration pattern for learning-oriented development.

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
python app.py --force_reindex
```

---

## 🧪 Running Tests

```bash
# Safety agent tests (no LLM — instant, 22 cases)
pytest tests/safety/ -v

# Integration tests (~6 LLM calls)
pytest tests/integration/ -v

# E2E UI tests (no LLM)
pytest tests/e2e/ -v

# Full suite
pytest -v
```

---

## 📁 Project Structure

```
hr-agent-hackathon/
├── app.py                          # Gradio UI — Chat + Quality Dashboard tabs
├── config/
│   ├── settings.py                 # Central config — models, thresholds, paths
│   └── agents/
│       └── hr_agent_config.py      # System prompt + collection config
├── src/
│   ├── base_rag_pipeline.py        # Base RAG pipeline — indexing + retrieval
│   ├── hr_rag_pipeline.py          # HR-specific pipeline (extends base)
│   └── agent/
│       ├── hr_agent.py             # Orchestrator — active routing + session stats
│       ├── intent_classifier.py    # Routing Agent — LLM intent classification
│       ├── reasoning_chain.py      # Reasoning Agent — two-stage classify + generate
│       └── safety_shield.py        # Safety Agent — injection/PII blocking
├── data/
│   └── hr_documents/               # 4 HR policy PDFs (source of truth)
├── docs/
│   └── architecture.png            # Architecture diagram
├── reports/
│   ├── ragas_scores.json           # RAGAS evaluation output
│   └── test_results/               # All test run outputs
└── tests/
    ├── safety/                     # 22 parametrized safety test cases
    ├── integration/                # 6 agent integration tests
    ├── e2e/                        # 5 end-to-end UI tests
    └── rag/                        # Embedding quality + RAGAS evaluation
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| LLM | `gpt-4o-mini` via GitHub Models |
| Judge LLM | `gpt-4.1-mini` (RAGAS evaluation only) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector DB | ChromaDB (Foundry IQ-inspired local grounding) |
| Framework | LangChain |
| RAG Evaluation | RAGAS |
| UI | Gradio |
| Testing | Pytest (parametrized fixtures) |
| AI-assisted Dev | GitHub Copilot (inline suggestions, code review) |
| Auth | `GITHUB_TOKEN` |
| IDE | VS Code |

---

## 🔑 Design Decisions

**Why dynamic routing instead of always running all agents?**
Most queries (greetings, OOS, injections) don't need RAG or generation. Routing them out early cuts LLM calls from 3 to 0-1, reduces latency, and stays within GitHub Models' free tier rate limits.

**Why local embeddings?**
HuggingFace `all-MiniLM-L6-v2` runs offline — no API cost, no rate limits, deterministic. The `get_embeddings()` factory in `llm_factory.py` is the single swap point for Azure AI embeddings.

**Why rule-based Safety Agent first?**
LLM-based safety checks cost API calls and add latency. The Safety Agent catches known injection patterns via regex + fuzzy matching with no LLM calls. The Reasoning Agent's Call 1 adds semantic injection detection as a second layer — defense-in-depth without paying LLM cost on every query.

**Why two LLM calls in the Reasoning Agent?**
Separating classification from generation prevents the LLM from generating a hallucinated answer before checking whether the context is sufficient. Call 2 only fires when Call 1 confirms the query is safe and answerable.

**Why a separate judge model for RAGAS?**
Using the same model as both pipeline and judge inflates scores. Pipeline uses `gpt-4o-mini`, judge uses `gpt-4.1-mini` — different model families reduce self-evaluation bias.

---

## 📝 Synthetic Data Notice

All HR policy documents are **synthetic**, created for demonstration purposes only. They represent a fictional company (ABC Corporation) and contain no real employee data, PII, or confidential information.

---

## 👩‍💻 Built By

**Swasti Shrivastava**  
GitHub: [@swasti16](https://github.com/swasti16)
