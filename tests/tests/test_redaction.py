
from pathlib import Path

import pytest

from redact_manager import RedactManager


@pytest.fixture(scope="module")
def project_files():
    root = Path(__file__).resolve().parents[1]
    output = root / "output"

    original = output / "parsed_document.md"
    redacted = output / "redacted_document.md"

    assert original.exists(), "Original Markdown is missing"
    assert redacted.exists(), "Redacted Markdown is missing"

    return original.read_text(encoding="utf-8"), \
        redacted.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def redactor():
    return RedactManager()


def test_original_document_contains_content(project_files):
    original, _ = project_files

    assert len(original) > 0


def test_redacted_document_contains_content(project_files):
    _, redacted = project_files

    assert len(redacted) > 0


def test_redaction_preserves_document_structure(project_files):
    original, redacted = project_files

    assert len(original.splitlines()) > 0
    assert len(redacted.splitlines()) > 0


def test_email_addresses_are_redacted(project_files):
    _, redacted = project_files

    assert "@gmail.com" not in redacted
    assert "@example.com" not in redacted


def test_redaction_placeholders_exist(project_files):
    _, redacted = project_files

    assert "<EMAIL_ADDRESS>" in redacted
    assert "<PASSPORT_NUMBER>" in redacted


def test_original_and_redacted_are_different(project_files):
    original, redacted = project_files

    assert original != redacted


def test_actual_document_redaction(redactor, project_files):
    original, saved_redacted = project_files

    generated_redacted = redactor.pii_filter(original)

    assert generated_redacted == saved_redacted


def test_chunks_are_redacted():
    root = Path(__file__).resolve().parents[1]
    chunks_path = root / "output" / "chunks.json"

    import json

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk["page_content"].strip()
