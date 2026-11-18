
import logging
from pathlib import Path

import pandas as pd
import yaml

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling_core.types.doc import (
    DocItemLabel,
    ImageRefMode,
    PictureItem
)

from langchain_core.documents import Document

from chunkmanager import ChunkManager
from storagemanager import StoreManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Step 1: Configuration
# ---------------------------------------------------------------

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# ---------------------------------------------------------------
# Step 2: PDF extraction using Docling
# ---------------------------------------------------------------

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    generate_picture_images=True,
    generate_page_images=True,
    images_scale=2.0
)

pdf_path = Path("airline_case_study.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(
        f"PDF file not found: {pdf_path}"
    )

output_dir = Path("output")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

result = converter.convert(str(pdf_path))

doc = result.document

logger.info("PDF converted successfully.")


# ---------------------------------------------------------------
# Step 3: Save only text as Markdown
# ---------------------------------------------------------------

text_labels = {
    DocItemLabel.TITLE,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.TEXT,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CAPTION,
    DocItemLabel.FOOTNOTE
}

md_path = output_dir / "parsed_document.md"

doc.save_as_markdown(
    md_path,
    labels=text_labels,
    image_mode=ImageRefMode.PLACEHOLDER
)

logger.info("Markdown saved: %s", md_path)


# ---------------------------------------------------------------
# Step 4: Extract images separately
# ---------------------------------------------------------------

picture_count = 0

for element, level in doc.iterate_items():

    if isinstance(element, PictureItem):

        picture_count += 1

        image = element.get_image(doc)

        if image:

            image_path = output_dir / (
                f"picture_{picture_count}.png"
            )

            image.save(image_path)

            logger.info(
                "Extracted picture: %s",
                image_path
            )


# ---------------------------------------------------------------
# Step 5: Extract tables separately as CSV
# ---------------------------------------------------------------

table_count = 0

for table in doc.tables:

    table_count += 1

    dataframe = table.export_to_dataframe(doc=doc)

    csv_path = output_dir / (
        f"table_{table_count}.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False
    )

    logger.info(
        "Extracted table: %s",
        csv_path
    )


logger.info("Total extracted images: %s", picture_count)
logger.info("Total extracted tables: %s", table_count)


# ---------------------------------------------------------------
# Step 6: Read extracted Markdown
# ---------------------------------------------------------------

markdown_text = md_path.read_text(
    encoding="utf-8"
)

logger.info("Markdown loaded successfully.")


# ---------------------------------------------------------------
# Step 7: Chunk Markdown using ChunkManager
# ---------------------------------------------------------------

chunk_manager = ChunkManager(config)

all_chunks = chunk_manager.chunk_markdown(
    markdown_text
)

logger.info(
    "Total text chunks generated: %s",
    len(all_chunks)
)


# ---------------------------------------------------------------
# Step 8: Convert extracted CSV tables into Documents
# ---------------------------------------------------------------

table_documents = []

for csv_path in sorted(output_dir.glob("table_*.csv")):

    dataframe = pd.read_csv(csv_path)

    if dataframe.empty:
        continue

    table_text = dataframe.to_markdown(
        index=False
    )

    table_document = Document(
        page_content=(
            f"Source table: {csv_path.name}\n"
            f"Document type: Airline policy table\n\n"
            f"{table_text}"
        ),
        metadata={
            "title": csv_path.stem,
            "document_type": "table"
        }
    )

    table_documents.append(table_document)


logger.info(
    "Total table documents generated: %s",
    len(table_documents)
)


# ---------------------------------------------------------------
# Step 9: Combine text chunks and table documents
# ---------------------------------------------------------------

all_chunks.extend(table_documents)

logger.info(
    "Total documents for embedding: %s",
    len(all_chunks)
)


# ---------------------------------------------------------------
# Step 10: Initialize StoreManager
# ---------------------------------------------------------------

store_manager = StoreManager(config)


# ---------------------------------------------------------------
# Step 11: Reset collection if configured
# ---------------------------------------------------------------

if config["storage"].get("reset_collection", False):

    store_manager.reset_collection()


# ---------------------------------------------------------------
# Step 12: Generate embeddings and store in ChromaDB
# ---------------------------------------------------------------

total_stored = store_manager.embed_and_store(
    chunks=all_chunks,
    source=config["metadata"]["source"]
)

logger.info(
    "Total documents processed: %s",
    total_stored
)


# ---------------------------------------------------------------
# Step 13: Verify ChromaDB
# ---------------------------------------------------------------

collection_count = store_manager.get_collection_count()

logger.info(
    "Total documents in ChromaDB: %s",
    collection_count
)


# ---------------------------------------------------------------
# Step 14: Test semantic similarity search
# ---------------------------------------------------------------

query = "What is the passenger baggage allowance?"

retrieved_documents = store_manager.similarity_search(
    query=query,
    k=3
)

print("\n" + "=" * 70)
print("SEMANTIC SEARCH RESULTS")
print("=" * 70)

for index, document in enumerate(
    retrieved_documents,
    start=1
):

    print(f"\nResult {index}")

    print("Metadata:")
    print(document.metadata)

    print("\nContent:")
    print(document.page_content)

    print("-" * 70)


logger.info("Pipeline execution completed successfully.")
