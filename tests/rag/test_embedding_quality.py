# rag-testing/embedding_tests.py
"""
Embedding Quality Tests

Tests that verify embeddings are correctly generated
and the embedding model is consistent between
indexing and querying phases.

Run with: pytest rag-testing/embedding_tests.py -v
"""

import pytest
import numpy as np
from sentence_transformers import SentenceTransformer, util
from config.settings import settings


# ======== Fixtures ==================================

@pytest.fixture(scope="module")
def embed_model():
    """Load embedding model once for all tests."""
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
            f"Similar sentences should not have low similarity: {similarity:.3f}. "
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
            f"Different sentences should not have high similarity: {similarity:.3f}. "
            f"Embedding model may not be distinguishing topics."
        )

    def test_embedding_dimensions_correct(self, embed_model):
        """
        Embedding dimensions must match expected model dimensions.
        all-MiniLM-L6-v2 produces 384-dimensional vectors.

        Tests: Embedding model configured correctly.
        """
        test_sentence = "Test sentence for dimension check"
        embedding = embed_model.encode(test_sentence)

        print(f"\n   Model: {settings.EMBEDDING_MODEL}")
        print(f"   Expected dimensions: 384")
        print(f"   Actual dimensions: {len(embedding)}")

        assert len(embedding) == 384, (
            f"Unexpected embedding dimensions: {len(embedding)}. "
            f"Expected 384 for {settings.EMBEDDING_MODEL}."
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
        print(f"   Expected: ~1.0 (normalized)")

        assert abs(norm - 1.0) < 0.001, (
            f"Embedding not normalized. Norm: {norm:.6f}. "
            f"Expected close to 1.0."
        )


# ======== Embedding Model Consistency Tests ==================================

class TestEmbeddingModelConsistency:
    """
    Verifies the same embedding model is used for
    both indexing and querying phases.
    This is the most critical embedding test.
    """

    def test_config_specifies_embedding_model(self):
        """
        Embedding model must be explicitly configured.
        Should never be None or empty.
        """
        assert settings.EMBEDDING_MODEL, \
            "EMBEDDING_MODEL not configured in settings"
        assert len(settings.EMBEDDING_MODEL) > 0, \
            "EMBEDDING_MODEL is empty string"
        print(f"\n   Embedding model: {settings.EMBEDDING_MODEL}")

    def test_embedding_model_loads_successfully(self):
        """
        Configured embedding model must load without errors.
        Catches: Wrong model name, missing model files.
        """
        try:
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            assert model is not None
            print(f"\n Model loaded Successfully: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            pytest.fail(
                f"Failed to load embedding model "
                f"'{settings.EMBEDDING_MODEL}': {e}"
            )

    def test_vector_dimensions_consistent(self):
        """
        Indexing embedding dimension must match query embedding dimension.
        Mismatch = retrieval will silently fail.

        This simulates the critical bug:
        Index with Model A → Query with Model B → Zero matches.
        """
        model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Simulate indexing phase
        index_sentence = "Employees get 15 days annual leave"
        index_embedding = model.encode(index_sentence)

        # Simulate querying phase (same model)
        query_sentence = "How many leave days?"
        query_embedding = model.encode(query_sentence)

        print(f"\n   Index embedding dim: {len(index_embedding)}")
        print(f"   Query embedding dim: {len(query_embedding)}")

        assert len(index_embedding) == len(query_embedding), (
            f"Dimension mismatch! "
            f"Index: {len(index_embedding)}, "
            f"Query: {len(query_embedding)}. "
            f"Embeddings are incompatible — retrieval will fail."
        )

# ======== Standalone Runner ==================================

if __name__ == "__main__":
    print("\nEmbedding Quality Tests - Standalone Run")
    print("=" * 50)
    pytest.main([__file__, "-v", "--tb=short"])