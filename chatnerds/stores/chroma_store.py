# Resources:
# https://github.com/pprados/langchain-rag/blob/master/docs/integrations/vectorstores/rag_vectorstore.ipynb

import logging
from typing import List, Dict, Any, Optional
import uuid
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.config import Settings
from chatnerds.stores.store_base import StoreBase
from chatnerds.lib.helpers import divide_list_in_chunks

DEFAULT_CHUNKS_COLLECTION_NAME = "chatnerd_chunks"


# Source: https://github.com/abasallo/rag/blob/master/vector_db/chroma_database.py
class ChromaStore(Chroma, StoreBase):
    def __init__(
        self,
        config: Dict[str, Any],
        collection_name: Optional[str] = DEFAULT_CHUNKS_COLLECTION_NAME,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ):
        super().__init__(
            collection_name=collection_name,
            persist_directory=config["persist_directory"],
            embedding_function=embeddings,
            client_settings=Settings(**config),
            **kwargs,
        )

    def add_documents(
        self,
        documents: List[Document],
        extra_metadata: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        if extra_metadata is not None:
            for document in documents:
                for key, value in extra_metadata.items():
                    document.metadata[key] = value

        # Add documents to the database (in batches of size max_batch_size)
        if hasattr(self._collection._client, "max_batch_size"):
            max_batch_size = self._collection._client.max_batch_size
        else:
            max_batch_size = len(documents)

        ids = []
        for batch_documents in list(divide_list_in_chunks(documents, max_batch_size)):
            batch_ids = super().add_documents(batch_documents, **kwargs)

            try:
                self.persist()
            except Exception as e:
                logging.warning(f"Error persisting database client: \n{str(e)}")

            ids.extend(batch_ids)
        return ids

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
            self._collection.upsert(
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

    def is_thread_safe(self) -> bool:
        return False

    # def close(self):
    #     Chroma(self)._client.clear_system_cache()

    @classmethod
    def does_vectorstore_exist(cls) -> bool:
        return True
