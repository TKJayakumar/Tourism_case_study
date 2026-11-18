
import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


logger = logging.getLogger(__name__)


class StoreManager:

    def __init__(self, config: dict):

        embedding_config = config["embedding"]
        storage_config = config["storage"]

        self.model_name = embedding_config["model_name"]
        self.device = embedding_config.get("device", "cpu")

        self.embedding_batch_size = embedding_config.get(
            "batch_size", 32
        )

        self.storage_batch_size = storage_config.get(
            "batch_size", 100
        )

        self.persist_directory = str(
            Path(storage_config["persist_directory"]).resolve()
        )

        self.collection_name = storage_config["collection_name"]

        self.distance_metric = storage_config.get(
            "distance_metric", "cosine"
        )

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True
        )

        self._initialize_embeddings()
        self._initialize_vector_store()

    def _initialize_embeddings(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": self.device
            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": self.embedding_batch_size
            }
        )

        logger.info("Embedding model loaded successfully.")

    def _initialize_vector_store(self):

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={
                "hnsw:space": self.distance_metric
            }
        )

        logger.info("ChromaDB initialized successfully.")

    @staticmethod
    def _generate_document_id(source: str, content: str) -> str:

        unique_content = f"{source}:{content}"

        return hashlib.sha256(
            unique_content.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _prepare_metadata(metadata: dict) -> dict:

        result = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                result[key] = str(value)

        return result

    def _prepare_documents(
        self,
        chunks: list[Document],
        source: str
    ):

        documents = []
        document_ids = []

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, Document):
                raise TypeError(
                    "Chunks must be LangChain Document objects."
                )

            content = chunk.page_content.strip()

            if not content:
                continue

            content_hash = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()

            metadata = dict(chunk.metadata)

            metadata.update({
                "source": source,
                "chunk_index": index,
                "chunk_length": len(content),
                "content_hash": content_hash
            })

            metadata = self._prepare_metadata(metadata)

            document_id = self._generate_document_id(
                source,
                content
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

            document_ids.append(document_id)

        if not documents:
            raise ValueError("No valid documents to embed.")

        return documents, document_ids

    def embed_and_store(
        self,
        chunks: list[Document],
        source: str
    ) -> int:

        documents, document_ids = self._prepare_documents(
            chunks,
            source
        )

        total_documents = len(documents)

        logger.info(
            "Starting embedding for %s documents.",
            total_documents
        )

        for start in range(
            0,
            total_documents,
            self.storage_batch_size
        ):

            end = start + self.storage_batch_size

            batch_documents = documents[start:end]
            batch_ids = document_ids[start:end]

            self.vector_store.add_documents(
                documents=batch_documents,
                ids=batch_ids
            )

            logger.info(
                "Processed %s / %s documents.",
                min(end, total_documents),
                total_documents
            )

        logger.info("Embedding and storage completed.")

        return total_documents

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> list[Document]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.vector_store.similarity_search(
            query=query,
            k=k
        )

    def get_collection_count(self) -> int:

        return self.vector_store._collection.count()

    def reset_collection(self):

        self.vector_store.delete_collection()

        self._initialize_vector_store()

        logger.info("ChromaDB collection reset successfully.")
