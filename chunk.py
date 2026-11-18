#mukesh
from pathlib import Path
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from docling_core.types.doc import (
    DocItemLabel,
    ImageRefMode,
    PictureItem
)

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    generate_picture_images=True,
    generate_page_images=True,
    images_scale=2.0
)

pdf_path = "airline_case_study.pdf"

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)
result = converter.convert(pdf_path)

doc = result.document

# ---------------------------------------------------------------
# Step 1: save only the TEXT as a markdown file (no tables, no pictures)
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
print("Saved text markdown:", md_path)

# ---------------------------------------------------------------
# Step 2: save images separately
# ---------------------------------------------------------------
picture_count = 0

for element, level in doc.iterate_items():
    if isinstance(element, PictureItem):
        picture_count += 1
        image = element.get_image(doc)
        if image:
            image.save(output_dir / f"picture_{picture_count}.png")
        print(f"Extracted picture: {picture_count}")

# ---------------------------------------------------------------
# Step 3: save tables separately as csv
# ---------------------------------------------------------------
table_count = 0

for table in doc.tables:
    table_count += 1
    df = table.export_to_dataframe(doc=doc)
    df.to_csv(
        output_dir / f"table_{table_count}.csv",
        index=False
    )
    print(f"Extracted table: {table_count}")

print(f"total extracted images are : {picture_count}")
print(f"total extracted tables are : {table_count}")

# ---------------------------------------------------------------
# Step 4: chunk the text from the md file
# ---------------------------------------------------------------
class ChunkManager:
    def __init__(self, config: dict):
        self.chn_config_size = config["chunking"]["chunk_size"]
        self.chn_config_overlap = config["chunking"]["chunk_overlap"]

    def chunk_markdown(self, markdown_text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chn_config_size,
            chunk_overlap=self.chn_config_overlap,
            separators=["### ", "## ", "# ", "\n\n", "\n", " ", ""],
        )
        return text_splitter.split_text(markdown_text)


config = {
    "chunking": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}

markdown_text = md_path.read_text(encoding="utf-8")

chunk_manager = ChunkManager(config)
all_chunks = chunk_manager.chunk_markdown(markdown_text)

print("Number of chunks:", len(all_chunks))

for i, text in enumerate(all_chunks):
    print(f"Chunk {i + 1}:")
    print(text)
    print("-" * 40)
