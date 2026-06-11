from src.hr_rag_pipeline import HRRagPipeline
import pytest


@pytest.fixture(scope="module")
def indexed_pipeline(tmp_path_factory):
    temp_dir = str(tmp_path_factory.mktemp("chromadb"))
    pipeline = HRRagPipeline(persist_dir=temp_dir)
    pipeline.load_and_index()
    return pipeline


def test_no_duplicate_chunks(indexed_pipeline):
    """
    Re-indexing same documents should not
    create duplicate chunks.
    """
    count_after_first_index = indexed_pipeline.get_chunk_count()

    # Re-initialize and re-index to test for duplicates
    indexed_pipeline.load_and_index(force_reindex=True)
    count_after_second_index = indexed_pipeline.get_chunk_count()

    assert count_after_first_index == count_after_second_index, (
        f"Duplicate chunks created on re-indexing! "
        f"Before: {count_after_first_index}, "
        f"After: {count_after_second_index}"
    )


def test_notice_period_sources(indexed_pipeline):
    result = indexed_pipeline.ask("What is the notice period for resignation?")

    # Test retrieval quality — chunks and sources
    sources = result['sources']
    assert any("notice" in s.lower() for s in sources), "notice_period doc not in sources"
    assert any("60" in ctx for ctx in result['contexts']), "60 days not found in retrieved chunks"


def test_shift_allowance_chunks_retrieved(indexed_pipeline):
    result = indexed_pipeline.ask("What is the night shift allowance?")
    assert any("shift" in s.lower() for s in result['sources'])
    assert any("450" in ctx for ctx in result['contexts']), "INR 450 not in retrieved chunks"
