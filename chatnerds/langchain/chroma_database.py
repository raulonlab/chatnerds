# Resources:
# https://github.com/pprados/langchain-rag/blob/master/docs/integrations/vectorstores/rag_vectorstore.ipynb

import os
import logging
from typing import List, Type, Dict, Any, Tuple
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.config import Settings

DEFAULT_CHUNKS_COLLECTION_NAME = "chatnerd_chunks"
DEFAULT_SOURCES_COLLECTION_NAME = "chatnerd_sources"


# Source: https://github.com/abasallo/rag/blob/master/vector_db/chroma_database.py
class ChromaDatabase:
    client: Chroma

    def __init__(
        self,
        embeddings: Embeddings,
        config: Dict[str, Any],
        collection_name: str = DEFAULT_CHUNKS_COLLECTION_NAME,
    ):
        self.client = Chroma(
            collection_name=collection_name,
            persist_directory=config["persist_directory"],
            embedding_function=embeddings,
            client_settings=Settings(**config),
        )
        self.collection = self.client.get()

    def add_documents(self, documents: List[Type["Document"]]) -> List[str]:
        try:
            ids = self.client.add_documents(documents)
            self.client.persist()
            return ids
        except ValueError as e:
            logging.error("Error loading documents: \n", e)
            return []

    def find_similar_docs(
        self, query, k, score=None
    ) -> List[Document] | List[Tuple[Document, float]]:
        if score:
            return self.client.similarity_search_with_score(query, k)
        else:
            return self.client.similarity_search(query, k)

    @staticmethod
    def does_vectorstore_exist(persist_directory: str) -> bool:
        """
        Checks if vectorstore exists
        """
        return os.path.exists(os.path.join(persist_directory, "chroma.sqlite3"))
