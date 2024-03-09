# Resources:
# https://github.com/pprados/langchain-rag/blob/master/docs/integrations/vectorstores/rag_vectorstore.ipynb

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import chromadb
import uuid
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.config import Settings
from chatnerds.utils import divide_list_in_chunks

DEFAULT_CHUNKS_COLLECTION_NAME = "chatnerd_chunks"
DEFAULT_PARENT_CHUNKS_COLLECTION_NAME = "chatnerd_parent_chunks"
DEFAULT_SOURCES_COLLECTION_NAME = "chatnerd_sources"


# Source: https://github.com/abasallo/rag/blob/master/vector_db/chroma_database.py
class ChromaDatabase:
    client: Chroma
    collection: chromadb.Collection

    def __init__(
        self,
        config: Dict[str, Any],
        collection_name: str = DEFAULT_CHUNKS_COLLECTION_NAME,
        embeddings: Embeddings | None = None,
    ):
        self.client = Chroma(
            collection_name=collection_name,
            persist_directory=config["persist_directory"],
            embedding_function=embeddings,
            client_settings=Settings(**config),
        )
        self.collection = self.client.get()

    def add_documents(self, documents: List[Document], **metadatas) -> List[str]:
        try:
            for document in documents:
                for key, value in metadatas.items():
                    document.metadata[key] = value

            # Add documents to the database (in batches of size max_batch_size)
            if hasattr(self.client._collection._client, "max_batch_size"):
                max_batch_size = self.client._collection._client.max_batch_size
            else:
                max_batch_size = len(documents)

            ids = []
            for batch_documents in list(
                divide_list_in_chunks(documents, max_batch_size)
            ):
                batch_ids = self.client.add_documents(batch_documents)
                self.client.persist()

                ids.extend(batch_ids)
            return ids
        except ValueError as err:
            logging.error(f"Error loading documents: \n{str(err)}")
            return []

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        **add_metadatas,
    ) -> List[str]:
        page_contents = []
        metadatas = []
        for document in documents:
            if document.page_content is None:
                continue
            page_contents.append(document.page_content)

            # Add extra metadata
            for key, value in add_metadatas.items():
                document.metadata[key] = value

            metadatas.append(document.metadata)

        # Generate ids
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in page_contents]

        try:
            self.client._collection.upsert(
                metadatas=metadatas,
                embeddings=embeddings,
                documents=page_contents,
                ids=ids,
            )
        except ValueError as e:
            if "Expected metadata value to be" in str(e):
                msg = (
                    "Try filtering complex metadata from the document using "
                    "langchain_community.vectorstores.utils.filter_complex_metadata."
                )
                raise ValueError(e.args[0] + "\n\n" + msg)
            else:
                raise e

        return ids

    def find_similar_docs(
        self, query: str, k: int = 4, with_score: bool = False
    ) -> List[Document] | List[Tuple[Document, float]]:
        if with_score:
            return self.client.similarity_search_with_score(query, k)
        else:
            return self.client.similarity_search(query, k)

    @staticmethod
    def does_vectorstore_exist(persist_directory: str) -> bool:
        """
        Checks if vectorstore exists
        """
        return os.path.exists(os.path.join(persist_directory, "chroma.sqlite3"))
