
import json
import logging
from pathlib import Path

import yaml
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

from chunkmanager import ChunkManager
from storagemanager import StoreManager
from redact_manager import RedactManager
from prompt_injection_checker import PromptInjectionChecker


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def main():

    config_path = BASE_DIR / "config_poison_test.yaml"
    pdf_path = BASE_DIR / "poison_test.pdf"

    if not config_path.exists():
        raise FileNotFoundError("config_poison_test.yaml not found.")

    if not pdf_path.exists():
        raise FileNotFoundError("poison_test.pdf not found.")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    output_dir = BASE_DIR / "output_poison_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=False,
        generate_page_images=False,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    logger.info("Processing test PDF...")

    result = converter.convert(str(pdf_path))
    markdown_path = output_dir / "poison_test.md"

    result.document.save_as_markdown(markdown_path)

    markdown_text = markdown_path.read_text(encoding="utf-8")

    redactor = RedactManager()
    injection_checker = PromptInjectionChecker()

    redacted_markdown = redactor.pii_filter(markdown_text)

    chunks = ChunkManager(config).chunk_markdown(
        redacted_markdown
    )

    for chunk in chunks:
        chunk.metadata.setdefault("source", pdf_path.name)
        chunk.metadata.setdefault("document_type", "test")

    safe_chunks = []
    injection_report = []

    for index, chunk in enumerate(chunks):

        check = injection_checker.check_text(
            chunk.page_content
        )

        if check["is_suspicious"]:

            injection_report.append({
                "chunk_index": index,
                "matched_patterns": check["matched_patterns"],
                "content_preview": chunk.page_content[:500],
            })

            logger.warning(
                "Suspicious chunk %s excluded.",
                index
            )

        else:
            safe_chunks.append(chunk)

    report_path = output_dir / "injection_report.json"

    report_path.write_text(
        json.dumps(
            injection_report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    chunks_path = output_dir / "chunks.json"

    chunks_path.write_text(
        json.dumps(
            [
                {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                for chunk in safe_chunks
            ],
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    logger.info("Total chunks: %s", len(chunks))
    logger.info("Suspicious chunks: %s", len(injection_report))
    logger.info("Safe chunks: %s", len(safe_chunks))

    # Use only the separate poison-test ChromaDB collection.
    store = StoreManager(config)

    if config.get("storage", {}).get("reset_collection", False):
        store.reset_collection()

    if safe_chunks:
        count = store.embed_and_store(
            safe_chunks,
            source=pdf_path.name
        )
    else:
        count = 0

    logger.info("Documents embedded: %s", count)
    logger.info(
        "Test ChromaDB collection count: %s",
        store.get_collection_count()
    )

    print("\n" + "=" * 45)
    print("DOCUMENT POISONING TEST RESULTS")
    print("=" * 45)
    print("Total chunks:", len(chunks))
    print("Suspicious chunks:", len(injection_report))
    print("Safe chunks:", len(safe_chunks))
    print("Documents embedded:", count)
    print("Injection report:", report_path)
    print("Safe chunks file:", chunks_path)
    print("=" * 45)


if __name__ == "__main__":
    main()
