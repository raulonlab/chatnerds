from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from langchain_core.embeddings import Embeddings
from chatnerds.stores.status_store import StatusStore
from chatnerds.stores.chroma_store import ChromaStore
from chatnerds.stores.qdrant_store import QdrantStore
from chatnerds.config import Config

DEFAULT_NERD_STORE_DIRECTORY = ".nerd_store"
DEFAULT_CHUNKS_COLLECTION_NAME = "chatnerd_chunks"


class StoreFactory:
    config: Dict[str, Any] = {}
    selected_store: Optional[str] = None
    selected_store_config: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selected_store, self.selected_store_config = (
            self.get_selected_store_and_config()
        )

    def get_vector_store(
        self,
        embeddings: Optional[Embeddings] = None,
        collection_name: str = DEFAULT_CHUNKS_COLLECTION_NAME,
    ) -> ChromaStore | QdrantStore:
        if self.selected_store == "chroma":
            return self.get_chroma_store(
                self.selected_store_config, embeddings, collection_name
            )
        elif self.selected_store == "qdrant":
            return self.get_qdrant_store(
                self.selected_store_config, embeddings, collection_name
            )
        else:
            raise ValueError(f"Unknown store '{self.selected_store}' in config file")

    def get_vector_store_class(self) -> ChromaStore.__class__ | QdrantStore.__class__:
        if self.selected_store == "chroma":
            return ChromaStore
        elif self.selected_store == "qdrant":
            return QdrantStore
        else:
            raise ValueError(f"Unknown store '{self.selected_store}' in config file")

    def get_status_store(self, **kwargs: Any) -> StatusStore:
        store_directory_path = str(
            Path(self.config["_nerd_base_path"], Config._NERD_STORE_DIRECTORYNAME)
        )
        return StatusStore(store_directory_path, **kwargs)

    def get_chroma_store(
        self,
        chroma_config: Dict[str, Any],
        embeddings: Optional[Embeddings] | None = None,
        collection_name: str = DEFAULT_CHUNKS_COLLECTION_NAME,
    ) -> ChromaStore:
        persist_directory = str(
            Path(self.config["_nerd_base_path"], DEFAULT_NERD_STORE_DIRECTORY, "chroma")
        )
        return ChromaStore(
            config={
                **chroma_config,
                "persist_directory": persist_directory,
            },
            collection_name=collection_name,
            embeddings=embeddings,
        )

    def get_qdrant_store(
        self,
        qdrant_config: Dict[str, Any],
        embeddings: Embeddings,
        collection_name: str = DEFAULT_CHUNKS_COLLECTION_NAME,
    ) -> QdrantStore:
        path = str(
            Path(self.config["_nerd_base_path"], DEFAULT_NERD_STORE_DIRECTORY, "qdrant")
        )
        return QdrantStore(
            config={
                **qdrant_config,
                "path": path,
            },
            collection_name=collection_name,
            embeddings=embeddings,
        )

    def get_selected_store_and_config(self) -> Tuple[str, Dict[str, Any]]:
        selected_store = self.config.get("vector_store", None)

        if not selected_store:
            raise ValueError("Key 'vector_store' not found in config file")

        if isinstance(selected_store, list) and len(selected_store) > 0:
            selected_store = str(selected_store[0]).strip()
        elif isinstance(selected_store, str) and len(selected_store.strip()) > 0:
            selected_store = selected_store.strip()
        else:
            raise ValueError(
                f"Invalid value '{selected_store}' in config's key 'vector_store'"
            )

        # Get selected store config
        selected_store_config = self.config.get(selected_store, None) or {}

        if not isinstance(selected_store_config, dict):
            raise ValueError(
                f"Store configuration '{selected_store}' is not a valid dictionary in config file"
            )

        # Return a cloned dictionary of the selected config
        return selected_store, dict(selected_store_config)
