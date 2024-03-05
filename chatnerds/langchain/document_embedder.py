import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatnerds.langchain.chroma_database import (
    ChromaDatabase,
    DEFAULT_SOURCES_COLLECTION_NAME,
    DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
)
from chatnerds.langchain.llm_factory import LLMFactory
from chatnerds.tools.event_emitter import EventEmitter
from chatnerds.config import Config

DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 100

_RUN_TASKS_LIMIT = 1_000  # Maximum number of tasks to run in a single call to run()


class DocumentEmbedder(EventEmitter):
    config: Dict[str, Any] = {}

    def __init__(self, nerd_config: Dict[str, Any]):
        super().__init__()
        self.config = nerd_config

    # @staticmethod
    def run(self, documents: List[Document]) -> Tuple[List[str], List[any]]:
        logging.debug("Running document embedder...")

        if len(documents) == 0:
            logging.debug("No documents to embed")
            return [], []

        # Limit number of tasks to run
        if len(documents) > _RUN_TASKS_LIMIT:
            documents = documents[:_RUN_TASKS_LIMIT]
            logging.warning(
                f"Number of documents to embeed exceeds limit of {_RUN_TASKS_LIMIT}. Only processing first {_RUN_TASKS_LIMIT} videos."
            )

        sources_database = ChromaDatabase(
            collection_name=DEFAULT_SOURCES_COLLECTION_NAME,
            config=self.config["chroma"],
        )
        parent_chunks_database = ChromaDatabase(
            collection_name=DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
            config=self.config["chroma"],
        )
        child_chunks_database = ChromaDatabase(config=self.config["chroma"])

        # Emit start event (show progress bar in UI)
        self.emit("start", len(documents))

        results = []
        errors = []
        max_workers = max(1, os.cpu_count() // 4)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            response_futures = [
                executor.submit(self.split_and_embed_document, document)
                for document in documents
            ]

            for response_future in as_completed(response_futures):
                try:
                    response = response_future.result()
                    if response:
                        # Add source documents to the database
                        source_documents = response["documents_lists"][0]
                        source_ids = response["ids_lists"][0]
                        source_embeddings = response["embeddings_lists"][0]
                        sources_database.add_documents_with_embeddings(
                            documents=source_documents,
                            ids=source_ids,
                            embeddings=source_embeddings,
                        )

                        # Add parent documents to the database
                        parent_documents = response["documents_lists"][1]
                        parent_ids = response["ids_lists"][1]
                        parent_embeddings = response["embeddings_lists"][1]
                        parent_chunks_database.add_documents_with_embeddings(
                            documents=parent_documents,
                            ids=parent_ids,
                            embeddings=parent_embeddings,
                        )

                        # Add child documents to the database
                        child_documents = response["documents_lists"][2]
                        child_ids = response["ids_lists"][2]
                        child_embeddings = response["embeddings_lists"][2]
                        child_chunks_database.add_documents_with_embeddings(
                            documents=child_documents,
                            ids=child_ids,
                            embeddings=child_embeddings,
                        )

                        results.append(
                            source_ids[0]
                        )  # append the id of the source document
                except Exception as err:
                    errors.append(err)
                    continue
                finally:
                    self.emit("update")

        self.emit("end")
        return results, errors

    @staticmethod
    def split_and_embed_document(document: Document) -> Dict[str, List[any]]:
        config = Config().get_nerd_config()
        embeddings: Embeddings = LLMFactory(config).get_embedding_function()

        child_splitter_config = {
            "separators": ["\n\n", "\n", ".", ",", " "],
            "keep_separator": False,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "chunk_size": DEFAULT_CHUNK_SIZE,
        } | config["splitter"]

        # fix chunk_size: It must be less than the model max sequence
        child_splitter_config["chunk_size"] = min(
            child_splitter_config["chunk_size"],
            embeddings.client.get_max_seq_length(),
        )

        # parent splitter config: twice the size of the child
        parent_splitter_config = {
            **child_splitter_config,
            "chunk_size": 2 * child_splitter_config["chunk_size"],
            "chunk_overlap": 2 * child_splitter_config["chunk_overlap"],
        }

        # Add created_at metadata
        created_at_utc_iso = datetime.now(timezone.utc).isoformat()
        document.metadata["created_at"] = created_at_utc_iso

        source_documents = [document]
        source_ids = [str(uuid.uuid1())]

        parent_chunks, parent_chunk_ids = DocumentEmbedder.split_documents(
            source_documents,
            parent_ids=source_ids,
            splitter_kwargs=parent_splitter_config,
        )

        child_chunks, child_chunk_ids = DocumentEmbedder.split_documents(
            parent_chunks,
            parent_ids=parent_chunk_ids,
            splitter_kwargs=child_splitter_config,
        )

        documents_lists = [source_documents, parent_chunks, child_chunks]
        ids_lists = [source_ids, parent_chunk_ids, child_chunk_ids]

        embeddings_lists = []
        for (
            documents
        ) in documents_lists:  # iterate over source, parent, and child documents
            embeddings_lists.append(
                embeddings.embed_documents(
                    [document.page_content for document in documents]
                )
            )

        return {
            "documents_lists": documents_lists,
            "ids_lists": ids_lists,
            "embeddings_lists": embeddings_lists,
        }

    @staticmethod
    def split_documents(
        documents: List[Document],
        parent_ids: Optional[List[str]] = None,
        splitter_kwargs: Dict[str, Any] = {},
    ) -> Tuple[List[Document], List[str]]:
        """
        Load documents and split in chunks
        """

        splitter_kwargs = {
            "separators": ["\n\n", "\n", ".", ",", " "],
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "keep_separator": False,
        } | splitter_kwargs

        text_splitter = RecursiveCharacterTextSplitter(
            **splitter_kwargs
        )  # .from_tiktoken_encoder

        texts, metadatas = [], []
        for index, document in enumerate(documents):
            texts.append(document.page_content)
            metadatas.append(
                {
                    **document.metadata,
                    "parent_id": (
                        parent_ids[index]
                        if (parent_ids and len(parent_ids) > index)
                        else None
                    ),
                }
            )
        chunks = text_splitter.create_documents(texts, metadatas=metadatas)

        # Generate chunk ids
        chunk_ids = [str(uuid.uuid1()) for _ in chunks]

        return chunks, chunk_ids
