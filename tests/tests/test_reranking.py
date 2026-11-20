import json
from datetime import date
from pathlib import Path

import pytest
from langchain_core.documents import Document

import reranker as reranker_module
from reranker import MeridianReranker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "output" / "chunks.json"


@pytest.fixture(scope="module")
def original_policy_documents():
    """Load the trainer-provided policy chunks; do not invent test policy text."""
    if not CHUNKS_PATH.exists():
        pytest.fail(f"Original project data not found: {CHUNKS_PATH}")

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    documents = [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {}),
        )
        for item in chunks
    ]

    # FC-006 is the current Flex long-haul rule; FC-023 is its superseded version.
    selected = [
    doc for doc in documents
    if doc.page_content.lstrip().startswith("## FC-006")
    or doc.page_content.lstrip().startswith("## FC-023")
]
    assert any("FC-006" in doc.page_content for doc in selected)
    assert any("FC-023" in doc.page_content for doc in selected)
    return selected


class FakeRanker:
    """Avoid downloading/running the FlashRank model during functional unit tests."""

    def __init__(self, *args, **kwargs):
        pass

    def rerank(self, request):
        # Deliberately score the superseded rule higher. The reranker should
        # still place the current rule first for a normal current-policy query.
        return [
            {"id": i, "score": 0.95 if "FC-023" in p["text"] else 0.60}
            for i, p in enumerate(request.passages)
        ]


@pytest.fixture
def reranker(monkeypatch, tmp_path):
    monkeypatch.setattr(reranker_module, "Ranker", FakeRanker)
    return MeridianReranker(
        top_k=5,
        min_score_threshold=0.05,
        cache_dir=str(tmp_path),
        reference_date=date(2026, 11, 18),
    )


def test_empty_candidates_return_empty_result(reranker):
    docs, latency = reranker.rerank("Flex baggage allowance", [])
    assert docs == []
    assert latency == 0.0


def test_current_policy_ranks_before_superseded_policy(
    reranker, original_policy_documents
):
    docs, latency = reranker.rerank(
        "What is the current Flex long-haul baggage allowance?",
        original_policy_documents,
    )

    assert docs
    assert latency >= 0
    assert "FC-006" in docs[0].page_content
    assert all(not doc.metadata["is_superseded"] for doc in docs)


def test_superseded_policy_is_available_for_historical_query(
    reranker, original_policy_documents
):
    docs, _ = reranker.rerank(
        "Show the previous superseded Flex long-haul policy",
        original_policy_documents,
    )

    assert any("FC-023" in doc.page_content for doc in docs)
    old_doc = next(doc for doc in docs if "FC-023" in doc.page_content)
    assert old_doc.metadata["is_superseded"] is True
    assert old_doc.metadata["status"] == "Inactive/Superseded"


def test_reranked_documents_keep_source_metadata(
    reranker, original_policy_documents
):
    docs, _ = reranker.rerank(
        "current Flex long-haul baggage policy",
        original_policy_documents,
    )

    assert docs
    for doc in docs:
        assert doc.metadata.get("source") == "Meridian_Air_Synthetic_Data_Pack.pdf"
        assert doc.metadata.get("document_id")
        assert "rerank_score" in doc.metadata


def test_top_k_limits_number_of_results(
    monkeypatch, tmp_path, original_policy_documents
):
    monkeypatch.setattr(reranker_module, "Ranker", FakeRanker)
    limited_reranker = MeridianReranker(
        top_k=1,
        cache_dir=str(tmp_path),
        reference_date=date(2026, 11, 18),
    )

    docs, _ = limited_reranker.rerank(
        "current Flex long-haul baggage policy",
        original_policy_documents,
    )

    assert len(docs) <= 1
