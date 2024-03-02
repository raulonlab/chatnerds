# Resources:
# https://github.com/pprados/langchain-rag/blob/master/docs/integrations/vectorstores/rag_vectorstore.ipynb

import os
import logging
from typing import List, Type, Dict, Any, Tuple
from rich import print as rprint
import chromadb
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.config import Settings

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

    def add_documents(self, documents: List[Type["Document"]]) -> List[str]:
        try:
            ids = self.client.add_documents(documents)
            self.client.persist()
            return ids
        except ValueError as e:
            logging.error("Error loading documents: \n", e)
            return []

    def find_similar_docs(
        self, query: str, k: int = 4, with_score: bool = False
    ) -> List[Document] | List[Tuple[Document, float]]:
        if with_score:
            return self.client.similarity_search_with_score(query, k)
        else:
            return self.client.similarity_search(query, k)

    def print_short_chunks(self):
        chunks = self.client.get(
            include=["documents", "metadatas"],
            limit=1000,
            offset=0,
        )

        print(f"{len(chunks.get('documents'))} chunks fetched...")

        for i in range(len(chunks.get("documents"))):
            if len(chunks.get("documents")[i]) < 100:
                rprint(
                    f"\n[bright_blue]Source: [bold]{chunks.get('metadatas')[i].get('source')}",
                    end="",
                    flush=True,
                )
                # rprint(chunks.metadatas[i].get("source"))
                print("")
                print(chunks.get("documents")[i])
                print("-----------------------------")

    @staticmethod
    def does_vectorstore_exist(persist_directory: str) -> bool:
        """
        Checks if vectorstore exists
        """
        return os.path.exists(os.path.join(persist_directory, "chroma.sqlite3"))
