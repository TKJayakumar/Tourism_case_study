
import json
from pathlib import Path

import pytest


@pytest.fixture
def chunks():
    project_dir = Path(__file__).resolve().parents[1]
    chunks_path = project_dir / "output" / "chunks.json"

    with open(chunks_path, "r", encoding="utf-8") as file:
        return json.load(file)


def test_chunks_file_exists():
    project_dir = Path(__file__).resolve().parents[1]

    assert (project_dir / "output" / "chunks.json").exists()


def test_chunks_are_generated(chunks):
    assert len(chunks) > 0


def test_chunk_content_not_empty(chunks):
    for chunk in chunks:
        assert chunk["page_content"].strip() != ""


def test_chunk_metadata_exists(chunks):
    for chunk in chunks:
        assert "metadata" in chunk
        assert isinstance(chunk["metadata"], dict)


def test_text_chunks_have_index(chunks):
    text_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"].get("document_type") == "text"
    ]

    for chunk in text_chunks:
        assert "chunk_index" in chunk["metadata"]


def test_table_chunks_have_title(chunks):
    table_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"].get("document_type") == "table"
    ]

    for chunk in table_chunks:
        assert "title" in chunk["metadata"]


def test_source_metadata_exists(chunks):
    for chunk in chunks:
        assert "source" in chunk["metadata"]


def test_table_chunks_exist(chunks):
    table_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"].get("document_type") == "table"
    ]

    assert len(table_chunks) > 0


def test_text_chunks_exist(chunks):
    text_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"].get("document_type") == "text"
    ]

    assert len(text_chunks) > 0
