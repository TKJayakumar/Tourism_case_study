
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document


class ChunkManager:

    def __init__(self, config: dict):

        chunk_config = config["chunking"]

        self.chunk_size = chunk_config["chunk_size"]
        self.chunk_overlap = chunk_config["chunk_overlap"]

        self.headers_to_split_on = [
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection")
        ]

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
            add_start_index=True
        )

    def chunk_markdown(self, markdown_text: str) -> list[Document]:

        if not markdown_text or not markdown_text.strip():
            raise ValueError("Markdown document is empty.")

        header_documents = self.header_splitter.split_text(
            markdown_text
        )

        final_documents = []

        for document in header_documents:

            chunks = self.text_splitter.split_documents(
                [document]
            )

            final_documents.extend(chunks)

        if not final_documents:
            raise ValueError("No chunks generated from document.")

        for index, document in enumerate(final_documents):

            document.metadata["chunk_index"] = index
            document.metadata["chunk_length"] = len(
                document.page_content
            )

        return final_documents
