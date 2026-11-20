
"""One-time document processing + embedding build for Meridian Air."""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DocItemLabel, ImageRefMode, PictureItem
from langchain_core.documents import Document

from chunkmanager import ChunkManager
from storagemanager import StoreManager
from redact_manager import RedactManager
from prompt_injection_checker import PromptInjectionChecker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def main():

    with (BASE_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pdf_path = BASE_DIR / config.get("metadata", {}).get(
        "source",
        "Meridian_Air_Synthetic_Data_Pack.pdf"
    )

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = BASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    full_mode = config.get("processing", {}).get("full_mode", False)

    pipeline_options = PdfPipelineOptions(
        do_ocr=full_mode,
        do_table_structure=True,
        generate_picture_images=full_mode,
        generate_page_images=full_mode,
        images_scale=1.5 if full_mode else 1.0,
    )

    logger.info("Starting PDF conversion. Full mode=%s", full_mode)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(str(pdf_path))
    doc = result.document

    logger.info("PDF conversion completed.")

    text_labels = {
        DocItemLabel.TITLE,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.TEXT,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.CAPTION,
        DocItemLabel.FOOTNOTE,
    }

    md_path = output_dir / "parsed_document.md"

    doc.save_as_markdown(
        md_path,
        labels=text_labels,
        image_mode=ImageRefMode.PLACEHOLDER
    )

    logger.info("Markdown saved: %s", md_path)

    # Extract pictures only in full mode.
    picture_count = 0

    if full_mode:
        for element, _level in doc.iterate_items():

            if isinstance(element, PictureItem):

                image = element.get_image(doc)

                if image:
                    picture_count += 1
                    image.save(output_dir / f"picture_{picture_count}.png")

    # Extract tables and convert them into LangChain Documents.
    table_documents = []

    for old_csv in output_dir.glob("table_*.csv"):
        old_csv.unlink()

    for index, table in enumerate(doc.tables, start=1):

        csv_path = output_dir / f"table_{index}.csv"

        dataframe = table.export_to_dataframe(doc=doc)
        dataframe.to_csv(csv_path, index=False)

        if not dataframe.empty:

            table_documents.append(
                Document(
                    page_content=(
                        f"Source table: {csv_path.name}\n"
                        "Document type: Airline policy table\n\n"
                        f"{dataframe.to_markdown(index=False)}"
                    ),
                    metadata={
                        "title": csv_path.stem,
                        "document_type": "table",
                        "source": pdf_path.name,
                    },
                )
            )

    logger.info(
        "Extracted %s tables and %s pictures.",
        len(table_documents),
        picture_count
    )

    markdown_text = md_path.read_text(encoding="utf-8")

    # Initialize security components.
    redactor = RedactManager()
    injection_checker = PromptInjectionChecker()

    # Redact PII from the complete Markdown document.
    redacted_markdown = redactor.pii_filter(markdown_text)

    redacted_md_path = output_dir / "redacted_document.md"

    redacted_md_path.write_text(
        redacted_markdown,
        encoding="utf-8"
    )

    logger.info("Redacted Markdown saved: %s", redacted_md_path)

    # Split original Markdown into chunks.
    text_chunks = ChunkManager(config).chunk_markdown(markdown_text)

    for chunk in text_chunks:
        chunk.metadata.setdefault("source", pdf_path.name)
        chunk.metadata.setdefault("document_type", "text")

    all_chunks = text_chunks + table_documents

    logger.info("Created %s text/table chunks.", len(all_chunks))

    # Redact PII from all chunks before scanning and storing.
    all_chunks = redactor.redact_documents(all_chunks)

    logger.info("PII redaction completed for %s chunks.", len(all_chunks))

    # Prompt-injection detection on each redacted chunk.
    safe_chunks = []
    injection_report = []

    for index, chunk in enumerate(all_chunks):

        result = injection_checker.check_text(chunk.page_content)

        if result["is_suspicious"]:

            injection_report.append({
                "chunk_index": index,
                "metadata": chunk.metadata,
                "matched_patterns": result["matched_patterns"],
                "content_preview": chunk.page_content[:500],
            })

            logger.warning(
                "Suspicious content detected in chunk %s. Excluding it from retrieval.",
                index
            )

        else:
            safe_chunks.append(chunk)

    # Save the injection detection report.
    report_path = output_dir / "injection_report.json"

    report_path.write_text(
        json.dumps(
            injection_report,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    logger.info("Injection report saved: %s", report_path)
    logger.info("Suspicious chunks found: %s", len(injection_report))
    logger.info("Safe chunks remaining: %s", len(safe_chunks))

    # Save only safe, redacted chunks for BM25 retrieval.
    chunks_path = output_dir / "chunks.json"

    serializable_chunks = [
        {
            "page_content": d.page_content,
            "metadata": d.metadata
        }
        for d in safe_chunks
    ]

    chunks_path.write_text(
        json.dumps(
            serializable_chunks,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    logger.info("Saved safe BM25 source chunks: %s", chunks_path)

    # Store only safe, redacted chunks in ChromaDB.
    store = StoreManager(config)

    if config.get("storage", {}).get("reset_collection", False):
        store.reset_collection()

    count = store.embed_and_store(
        safe_chunks,
        source=pdf_path.name
    )

    logger.info("Embedded and stored %s safe documents.", count)
    logger.info("Chroma collection count: %s", store.get_collection_count())

    logger.info("Processing complete. Next run: python hybrid_retrieval.py")


if __name__ == "__main__":
    main()
