import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.hr_rag_pipeline import HRRagPipeline
import pytest


@pytest.fixture(scope="session")
def indexed_pipeline():
    """Load and index the HR documents once for this test session."""
    pipeline = HRRagPipeline()
    pipeline.load_and_index()
    return pipeline


def test_no_duplicate_chunks(indexed_pipeline):
    """
    Re-indexing same documents should not
    create duplicate chunks.
    """
    count_after_first_index = indexed_pipeline.get_chunk_count()

    # Re-initialize and re-index to test for duplicates
    indexed_pipeline.load_and_index()
    count_after_second_index = indexed_pipeline.get_chunk_count()

    assert count_after_first_index == count_after_second_index, (
        f"Duplicate chunks created on re-indexing! "
        f"Before: {count_after_first_index}, "
        f"After: {count_after_second_index}"
    )


def test_notice_period_sources(indexed_pipeline):
    """Test that a question about notice period retrieves the correct source and answer."""
    print("Chunk count in second test", indexed_pipeline.get_chunk_count())
    result = indexed_pipeline.ask(
        "What is the notice period for resignation?"
    )
    print(f"Answer: {result['answer']}")
    # At minimum — correct source must be present
    sources = result['sources']
    assert any("leave_policy" in s for s in sources), "leave_policy.txt not in sources!"

    # Answer must be correct regardless of extra sources
    assert "30 days" in result['answer'], "Correct notice period not in answer"
    assert "60 days" in result['answer'], "Full notice period info not in answer"

    # Log if irrelevant source appeared (warning not failure)
    if any("code_of_conduct" in s for s in sources):
        print("Warning: code_of_conduct.txt retrieved "
            "for notice period query. "
            "Consider reducing TOP_K or adding metadata filter.")