# Resources:
# https://github.com/pprados/langchain-rag/blob/master/docs/integrations/vectorstores/rag_vectorstore.ipynb

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

DEFAULT_CHUNKS_COLLECTION_NAME = "chatnerd_chunks"


class StoreBase:
    def is_thread_safe(self) -> bool:
        return True

    def close(self):
        pass

    def find_similar_docs(
        self, query: str, k: int = 4, with_score: bool = False
    ) -> List[Document] | List[Tuple[Document, float]]:
        if not isinstance(self, VectorStore):
            raise ValueError("StoreBase should only be used by VectorStore instances")

        if with_score:
            return self.similarity_search_with_score(query, k)
        else:
            return self.similarity_search(query, k)

    @classmethod
    def does_vectorstore_exist(cls) -> bool:
        return True
