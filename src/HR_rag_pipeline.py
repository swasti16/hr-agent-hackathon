"""
HR RAG Pipeline — Core Implementation

Builds a RAG pipeline over HR policy documents.
Used as the system under test for all RAG evaluation tests.

Pipeline:
  1. Load HR documents from data/hr_documents/
  2. Split into chunks
  3. Embed using HuggingFace sentence-transformers
  4. Store in ChromaDB vector store
  5. Query: retrieve top-k chunks + generate answer via LLM
"""
import os
import sys

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.base_rag_pipeline import RagPipeline
import config.agents.hr_agent_config as hr_config


class HRRagPipeline(RagPipeline):
    def __init__(self):
        super().__init__(docs_dir=hr_config.DOCS_DIR, collection_name=hr_config.COLLECTION_NAME,
                         system_prompt=hr_config.SYSTEM_PROMPT)

# ======== Quick Test ==================================

if __name__ == "__main__":
    
    print("\n🚀 HR RAG Pipeline — Quick Test")
    print("=" * 50)

    pipeline = HRRagPipeline()
    chunk_count = pipeline.load_and_index()

    print(f"\n📊 Total chunks in DB: {pipeline.get_chunk_count()}")

    # Test questions
    test_questions = [
        "How many annual leave days do employees get?",
        "What is the maternity leave policy?",
        "What is the notice period for resignation?",
        "What happens if an employee violates the code of conduct?"
    ]

    print("\n💬 Testing Questions:")
    print("-" * 50)

    for question in test_questions:
        result = pipeline.ask(question)
        print(result)
        print(f"\nQ: {question}")
        print(f"A: {result['answer'][:200]}...")
        print(f"Sources: {set(result['sources'])}")

    print("\n✅ Pipeline test complete!")

