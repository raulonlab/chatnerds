import logging
from typing import List, Dict, Any, Optional
import threading
from qdrant_client import QdrantClient, models
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.fake import FakeEmbeddings
from chatnerds.stores.store_base import StoreBase
from chatnerds.lib.helpers import divide_list_in_chunks

DEFAULT_CHUNKS_COLLECTION_NAME = "chatnerd_chunks"


class QdrantStore(Qdrant, StoreBase):
    __local = threading.local()
    config: Dict[str, Any] = {}
    qdrant_client: QdrantClient = None

    def __init__(
        self,
        config: Dict[str, Any],
        collection_name: Optional[str] = DEFAULT_CHUNKS_COLLECTION_NAME,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ):
        self.config = config

        # Keep a singleton and thread safe instance of the QdrantClient
        if not hasattr(self.__local, "qdrant_client"):
            self.__local.qdrant_client = QdrantClient(
                # path=config["path"],
                **config,
            )

        # Create collection if it does not exist
        if embeddings and not self.__local.qdrant_client.collection_exists(
            collection_name=collection_name
        ):
            # Get sentence_embedding_dimension from embeddings
            sentence_embedding_dimension = (
                embeddings.client.get_sentence_embedding_dimension()
            )

            self.__local.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    distance=models.Distance.COSINE,
                    size=sentence_embedding_dimension,
                ),
                # optimizers_config=models.OptimizersConfigDiff(memmap_threshold=0),
                # hnsw_config=models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)
            )

        # Create Fake embeddings if embeddings are not provided
        if not embeddings:
            embeddings = FakeEmbeddings(size=0)

        super().__init__(
            client=self.__local.qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings,
            **kwargs,
        )

    def close(self):
        if hasattr(self.__local, "qdrant_client"):
            self.__local.qdrant_client.close()
            self.__local.qdrant_client = None

    def add_documents(
        self,
        documents: List[Document],
        extra_metadata: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        if extra_metadata is not None and isinstance(extra_metadata, dict):
            for document in documents:
                for key, value in extra_metadata.items():
                    document.metadata[key] = value

        # Add documents to the database (in batches of size max_batch_size)
        max_batch_size = 64

        ids = []
        for batch_documents in list(divide_list_in_chunks(documents, max_batch_size)):
            batch_ids = super().add_documents(batch_documents, **kwargs)
            try:
                self.persist()
            except Exception as e:
                logging.warning(f"Error persisting database client: \n{str(e)}")

            ids.extend(batch_ids)
        return ids

    def get(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Method 'get' not implemented for QdrantStore")

    def is_thread_safe(self) -> bool:
        return self.config.get("is_thread_safe", False)

    @classmethod
    def does_vectorstore_exist(cls) -> bool:
        return True
