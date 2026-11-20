
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from lcel_chain import format_reranked_docs


@pytest.fixture(scope="module")
def original_chunks():
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "output" / "chunks.json"

    assert chunks_path.exists(), "chunks.json not found"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        for item in chunks
    ]


def test_actual_documents_are_formatted(original_chunks):
    formatted = format_reranked_docs(original_chunks[:3])

    assert formatted.strip()
    assert "Document ID:" in formatted
    assert "Section:" in formatted


def test_document_content_is_preserved(original_chunks):
    document = original_chunks[0]

    formatted = format_reranked_docs([document])

    assert document.page_content in formatted


def test_document_section_is_included(original_chunks):
    document = original_chunks[0]

    formatted = format_reranked_docs([document])

    section = document.metadata.get("section", "General")

    assert section in formatted


def test_document_status_is_included(original_chunks):
    document = original_chunks[0]

    formatted = format_reranked_docs([document])

    status = document.metadata.get("status", "Current")

    assert status in formatted


def test_multiple_documents_are_formatted(original_chunks):
    formatted = format_reranked_docs(original_chunks[:3])

    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "[3]" in formatted


def test_empty_retrieved_documents():
    result = format_reranked_docs([])

    assert result == "No relevant documents found."
