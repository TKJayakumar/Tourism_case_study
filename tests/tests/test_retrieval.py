
import json
import yaml
import pytest

from pathlib import Path
from langchain_core.documents import Document

from retrievemanager import RetrieveManager
from storagemanager import StoreManager


@pytest.fixture(scope="module")
def project_data():
    root = Path(__file__).resolve().parents[1]

    chunks_file = root / "output" / "chunks.json"
    config_file = root / "config.yaml"

    assert chunks_file.exists(), "chunks.json not found"
    assert config_file.exists(), "config.yaml not found"

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    documents = [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        for item in chunks
        if item.get("page_content", "").strip()
    ]

    return root, documents, config


@pytest.fixture(scope="module")
def retrieval_setup(project_data):
    root, documents, config = project_data

    # Use the existing ChromaDB folder
    config["storage"]["persist_directory"] = str(
        root / "chroma_db"
    )

    store = StoreManager(config)
    retriever = RetrieveManager()

    return documents, store, retriever


def test_original_chunks_are_loaded(project_data):
    _, documents, _ = project_data

    assert len(documents) > 0


def test_chromadb_contains_documents(retrieval_setup):
    _, store, _ = retrieval_setup

    assert store.get_collection_count() > 0


def test_retrieval_returns_relevant_documents(retrieval_setup):
    documents, store, retriever = retrieval_setup

    question = (
        "How many checked bags does a Flex fare allow "
        "on a long-haul route?"
    )

    results = retriever.query_hybrid(
        query=question,
        documents=documents,
        vector_store=store.vector_store
    )

    assert len(results) > 0


def test_retrieval_finds_correct_baggage_policy(retrieval_setup):
    documents, store, retriever = retrieval_setup

    question = (
        "How many checked bags does a Flex fare allow "
        "on a long-haul route?"
    )

    results = retriever.query_hybrid(
        query=question,
        documents=documents,
        vector_store=store.vector_store
    )

    combined_content = " ".join(
        doc.page_content.lower() for doc in results
    )

    assert "flex" in combined_content
    assert "long-haul" in combined_content
    assert "23 kg" in combined_content


def test_retrieval_returns_source_metadata(retrieval_setup):
    documents, store, retriever = retrieval_setup

    results = retriever.query_hybrid(
        query="Flex long-haul checked baggage allowance",
        documents=documents,
        vector_store=store.vector_store
    )

    assert len(results) > 0

    for doc in results:
        assert doc.metadata.get("source")


def test_empty_documents_return_empty_results(retrieval_setup):
    _, store, retriever = retrieval_setup

    results = retriever.query_hybrid(
        query="baggage allowance",
        documents=[],
        vector_store=store.vector_store
    )

    assert results == []
