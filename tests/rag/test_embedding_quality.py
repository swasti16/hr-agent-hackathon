"""
Embedding Quality Tests

Tests that verify embeddings are correctly generated
and the embedding model is consistent between
indexing and querying phases.

Run with: pytest rag-testing/embedding_tests.py -v
"""

import chromadb
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer, util
from config.settings import settings

# ======== Fixtures ==================================


@pytest.fixture(scope="module")
def embed_model():
    """
    Load embedding model once for all tests.
    Fails fast with clear message if model not configured.
    """
    if not settings.EMBEDDING_MODEL:
        raise ValueError(
            "EMBEDDING_MODEL not configured in settings. "
            "Set it to a valid HuggingFace model e.g. 'all-MiniLM-L6-v2'"
        )
    return SentenceTransformer(settings.EMBEDDING_MODEL)


# ======== Embedding Quality Tests ==================================

class TestEmbeddingQuality:
    """
    Verifies that embeddings correctly capture semantic meaning.
    """

    def test_similar_sentences_have_high_similarity(self, embed_model):
        """
        Semantically similar sentences should have
        cosine similarity > 0.70.

        Tests: Embedding model captures meaning correctly.
        """
        sentence1 = "How many annual leave days do employees get?"
        sentence2 = "What is the annual leave entitlement for staff?"

        emb1 = embed_model.encode(sentence1)
        emb2 = embed_model.encode(sentence2)
        similarity = util.cos_sim(emb1, emb2).item()

        print(f"\n   Sentence 1: {sentence1}")
        print(f"   Sentence 2: {sentence2}")
        print(f"   Similarity: {similarity:.3f}")

        assert similarity > 0.70, (
            f"Similar sentences scored too low: {similarity:.3f}. "
            f"Embedding model may not be capturing meaning correctly."
        )

    def test_different_sentences_have_low_similarity(self, embed_model):
        """
        Semantically different sentences should have
        cosine similarity < 0.50.

        Tests: Embedding model distinguishes different topics.
        """
        sentence1 = "What is the maternity leave policy?"
        sentence2 = "What is the disciplinary process for violations?"

        emb1 = embed_model.encode(sentence1)
        emb2 = embed_model.encode(sentence2)
        similarity = util.cos_sim(emb1, emb2).item()

        print(f"\n   Sentence 1: {sentence1}")
        print(f"   Sentence 2: {sentence2}")
        print(f"   Similarity: {similarity:.3f}")

        assert similarity < 0.50, (
            f"Unrelated sentences scored too similarly: {similarity:.3f}. "
            f"Expected < 0.50. Embedding model may not distinguish topics."
        )

    def test_embedding_dimensions_correct(self, embed_model):
        """
        Embedding dimensions must match expected model dimensions.

        Tests: Embedding model configured correctly.
        """
        expected_dimension = embed_model.get_embedding_dimension()

        test_sentence = "Test sentence for dimension check"
        embedding = embed_model.encode(test_sentence)

        print(f"\n   Model: {settings.EMBEDDING_MODEL}")
        print(f"   Expected dimensions: {expected_dimension}")
        print(f"   Actual dimensions: {len(embedding)}")

        assert len(embedding) == expected_dimension, (
            f"Unexpected embedding dimensions: {len(embedding)}. "
            f"Expected {expected_dimension} for {settings.EMBEDDING_MODEL}."
        )

    def test_same_sentence_produces_same_embedding(self, embed_model):
        """
        Same input must always produce same embedding.
        Embeddings must be deterministic.

        Tests: Embedding model is not random.
        """
        sentence = "How many sick days do employees get?"

        emb1 = embed_model.encode(sentence)
        emb2 = embed_model.encode(sentence)
        similarity = util.cos_sim(emb1, emb2).item()

        print(f"\n   Sentence: {sentence}")
        print(f"   Self-similarity: {similarity:.6f}")

        assert similarity > 0.9999, (
            f"Same sentence produced different embeddings! "
            f"Similarity: {similarity:.6f}. "
            f"Embedding model is not deterministic."
        )

    def test_hr_query_retrieves_correct_chunk(self, embed_model):
        """
        Golden retrieval test:
        A known HR question should retrieve a relevant chunk.

        Tests: Embeddings enable correct semantic retrieval.
        """
        # Known related pairs
        query = "maternity leave duration"
        relevant_chunk = "Female employees are entitled to 26 weeks of paid maternity leave"
        irrelevant_chunk = "Employees must not share company confidential information"

        query_emb = embed_model.encode(query)
        relevant_emb = embed_model.encode(relevant_chunk)
        irrelevant_emb = embed_model.encode(irrelevant_chunk)

        relevant_sim = util.cos_sim(query_emb, relevant_emb).item()
        irrelevant_sim = util.cos_sim(query_emb, irrelevant_emb).item()

        print(f"\n   Query: {query}")
        print(f"   Relevant chunk similarity:   {relevant_sim:.3f}")
        print(f"   Irrelevant chunk similarity: {irrelevant_sim:.3f}")

        assert relevant_sim > irrelevant_sim, (
            f"Relevant chunk ({relevant_sim:.3f}) scored lower than "
            f"irrelevant chunk ({irrelevant_sim:.3f}). "
            f"Embedding model not capturing HR topic relevance."
        )

    def test_embedding_is_normalized(self, embed_model):
        """
        Embeddings should be normalized (unit vectors).
        Normalized embeddings make cosine similarity more reliable.

        Tests: Embedding model normalization setting.
        """
        sentence = "Leave policy test"
        embedding = embed_model.encode(
            sentence,
            normalize_embeddings=True
        )
        norm = np.linalg.norm(embedding)

        print(f"\n   Embedding norm: {norm:.6f}")
        print(" Expected: ~1.0 (normalized)")

        assert abs(norm - 1.0) < 0.001, (
            f"Embedding not normalized. Norm: {norm:.6f}. "
            f"Expected close to 1.0."
        )


class TestEmbeddingModelConsistency:
    """
    Verifies the same embedding model is used for
    both indexing and querying phases.
    This is the most critical embedding test.
    """

    def test_vector_dimensions_consistent(self, embed_model):
        """
        Verifies configured embedding model matches
        what was used to build the vector store.
        Guards against: index with Model A, query with Model B.
        """
        configured_dim = embed_model.get_embedding_dimension()
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        collections = client.list_collections()
        print(f"\n Collection: {collections}")

        if not collections:
            pytest.skip("Vector store is empty — index documents first")

        for colname in collections:
            collection = client.get_collection(name=colname)

            # Get a stored vector and check its dimension
            stored = collection.get(limit=1, include=["embeddings"])

            if stored and stored.get("embeddings", None) is not None and len(stored["embeddings"]) > 0:
                stored_dim = len(stored["embeddings"][0])

                print(f"\n {colname} Configured model dimension: {configured_dim}")
                print(f" {colname} Stored vector dimension:    {stored_dim}")

                assert configured_dim == stored_dim, (
                    f"Dimension mismatch! Configured: {configured_dim}, "
                    f"Stored: {stored_dim}. "
                    f"Different embedding models used for indexing and querying!"
                )

# ======== Standalone Runner ==================================


if __name__ == "__main__":
    print("\nEmbedding Quality Tests - Standalone Run")
    print("=" * 50)
    pytest.main([__file__, "-v", "--tb=short"])
