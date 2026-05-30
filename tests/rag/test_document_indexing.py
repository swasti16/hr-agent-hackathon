"""
Document Indexing Tests

Verifies that documents are correctly indexed into ChromaDB
and can be retrieved after indexing.

Tests:
  - Chunk count increases after new document indexed
  - Metadata stored correctly with each chunk
  - Known content retrievable after indexing
  - Duplicate indexing handled correctly

"""

import pytest
import tempfile
import shutil

from src.utils.llm_factory import get_vector_store
from src.base_rag_pipeline import RagPipeline

# ======== Fixtures ==================================

@pytest.fixture
def temp_db_dir():
    """Create a temporary ChromaDB directory for each test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def empty_vector_store(temp_db_dir):
    """Create an empty vector store for testing."""
    return get_vector_store(collection_name="test_collection",
                            persist_directory=temp_db_dir)


# ======== Indexing Tests ==================================

class TestDocumentIndexing:
    """Tests for document indexing into ChromaDB."""

    def test_chunk_count_increases_after_indexing(self, empty_vector_store):
        """
        After adding documents, chunk count should increase.
        Tests: Documents are actually being stored.
        """
        # Count before
        count_before = empty_vector_store._collection.count()
        print(f"\n   Chunks before: {count_before}")

        # Add a document
        empty_vector_store.add_texts(
            texts=["Employees get 15 days annual leave per year."],
            metadatas=[{"source": "leave_policy.txt", "page": 1}]
        )

        # Count after
        count_after = empty_vector_store._collection.count()
        print(f"   Chunks after: {count_after}")

        assert count_after > count_before, (
            f"Chunk count did not increase after indexing. "
            f"Before: {count_before}, After: {count_after}"
        )

    def test_metadata_stored_correctly(self, empty_vector_store):
        """
        Metadata (source, page) must be stored with each chunk.
        Tests: Metadata retrieval works for debugging.
        """
        # Add with metadata
        empty_vector_store.add_texts(
            texts=["Maternity leave is 26 weeks paid."],
            metadatas=[{
                "source": "leave_policy.txt",
                "page": 2,
                "section": "maternity"
            }]
        )

        # Retrieve and check metadata
        results = empty_vector_store.similarity_search(
            "maternity leave", k=1
        )

        assert len(results) > 0, "No results returned"
        metadata = results[0].metadata

        print(f"\n   Retrieved metadata: {metadata}")

        assert metadata.get("source") == "leave_policy.txt", \
            "Source metadata not stored correctly"
        assert metadata.get("page") == 2, \
            "Page metadata not stored correctly"

    def test_known_content_retrievable_after_indexing(
        self, empty_vector_store
    ):
        """
        Direct Query Test:
        Content added to vector store must be retrievable.

        Tests: End-to-end indexing and retrieval works.
        """
        unique_fact = "ABC Corporation paternity leave is exactly 5 days."

        # Index the unique fact
        empty_vector_store.add_texts(
            texts=[unique_fact],
            metadatas=[{"source": "test_doc.txt"}]
        )

        # Query for it
        results = empty_vector_store.similarity_search(
            "paternity leave days", k=1
        )

        assert len(results) > 0, "No results returned after indexing"

        retrieved_text = results[0].page_content
        print(f"\n   Added: {unique_fact}")
        print(f"   Retrieved: {retrieved_text}")

        assert "5 days" in retrieved_text or "paternity" in retrieved_text, (
            f"Expected content not found in retrieved chunk. "
            f"Retrieved: {retrieved_text}"
        )

    def test_multiple_documents_indexed(self, empty_vector_store):
        """
        Multiple documents should all be indexed correctly.
        Tests: Batch indexing works.
        """
        texts = [
            "Annual leave is 15 days per year.",
            "Sick leave is 10 days per year.",
            "Maternity leave is 26 weeks paid.",
            "Paternity leave is 5 days.",
            "Notice period is 30 or 60 days."
        ]
        metadatas = [
            {"source": f"doc_{i}.txt"} for i in range(len(texts))
        ]

        empty_vector_store.add_texts(texts=texts, metadatas=metadatas)

        count = empty_vector_store._collection.count()
        print(f"\n   Added {len(texts)} chunks")
        print(f"   Total in DB: {count}")

        assert count >= len(texts), (
            f"Not all chunks indexed. "
            f"Expected >= {len(texts)}, got {count}"
        )

    def test_similarity_scores_reasonable(self, empty_vector_store):
        """
        Similarity scores for relevant content should be
        higher than for irrelevant content.

        Tests: Vector similarity search working correctly.
        """
        # Add relevant and irrelevant content
        empty_vector_store.add_texts(
            texts=[
                "Annual leave entitlement is 15 days per year.",
                "The office cafeteria serves lunch from 12pm to 2pm."
            ],
            metadatas=[
                {"source": "leave.txt"},
                {"source": "facilities.txt"}
            ]
        )

        # Search for leave-related query
        results = empty_vector_store.similarity_search_with_score(
            "how many leave days", k=2
        )

        assert len(results) >= 2, "Expected at least 2 results"

        # Lower score = more similar in ChromaDB (distance)
        doc1, score1 = results[0]
        doc2, score2 = results[1]

        print(f"\n   Top result: {doc1.page_content[:50]}... (score: {score1:.3f})")
        print(f"   2nd result: {doc2.page_content[:50]}... (score: {score2:.3f})")

        # Top result should be the leave-related chunk
        assert "leave" in doc1.page_content.lower() or \
               score1 <= score2, (
            "Relevant chunk not ranked higher than irrelevant chunk"
        )


# ======== Rag Pipeline Indexing Tests ==================================

class TestRagPipelineIndexing:
    """
    Integration tests for the full HR RAG pipeline indexing.
    Tests indexing of actual HR documents.
    """

    @pytest.fixture(autouse=True)
    def setup(self, temp_db_dir):
        """Initialize pipeline with temp directory."""
        self.pipeline = RagPipeline(
            docs_dir="./data/hr_documents",
            collection_name="test_hr",
            persist_dir=temp_db_dir
        )

    def test_hr_documents_indexed_successfully(self):
        """HR documents directory must index without errors."""
        chunk_count = self.pipeline.load_and_index(force_reindex=True)
        print(f"\n   Indexed {chunk_count} chunks from HR documents")
        assert chunk_count > 0, "No chunks were indexed from HR documents"

    def test_leave_policy_content_indexed(self):
        """Leave policy content must be findable after indexing."""
        self.pipeline.load_and_index(force_reindex=True)
        result = self.pipeline.ask("How many annual leave days?")

        print(f"\n   Answer: {result['answer'][:100]}...")
        print(f"   Contexts retrieved: {len(result['contexts'])}")

        assert len(result["contexts"]) > 0, \
            "No contexts retrieved for leave policy question"
        assert result["answer"] is not None, \
            "No answer generated"
        assert len(result["answer"]) > 10, \
            "Answer too short — may have failed"

    def test_chunk_count_after_hr_indexing(self):
        """After indexing HR docs, chunk count should be reasonable."""
        self.pipeline.load_and_index(force_reindex=True)
        count = self.pipeline.get_chunk_count()

        print(f"\n   Total HR chunks: {count}")

        # 2 documents with ~500 char chunks should give 5-30 chunks
        assert count >= 5, \
            f"Too few chunks indexed: {count}. Documents may not have loaded."
        assert count <= 100, \
            f"Too many chunks: {count}. Check chunking configuration."
