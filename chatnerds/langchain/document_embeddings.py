import os
import glob
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from multiprocessing import Pool
from tqdm import tqdm
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatnerds.document_loaders.transcript_loader import TranscriptLoader
from chatnerds.langchain.chroma_database import (
    ChromaDatabase,
    DEFAULT_SOURCES_COLLECTION_NAME,
    DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
)
from chatnerds.utils import TimeTaken

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100


# Map file extensions to document loaders and their arguments
_LOADER_MAPPING = {
    ".csv": (CSVLoader, {"encoding": "utf8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".transcript": (TranscriptLoader, {}),
    # ".mp3": (OpenAIWhisperLoader, {"device": "cpu", "lang_model": "openai/whisper-base"}),
    # Add more mappings for other file extensions and loaders as needed
}


class DocumentEmbeddings:
    config: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_embedding_function(self) -> Embeddings:
        embeddings_config = {**self.config["embeddings"]}
        if embeddings_config["model_name"].startswith("hkunlp/"):
            provider_class = HuggingFaceInstructEmbeddings
        else:
            provider_class = HuggingFaceEmbeddings
        return provider_class(**embeddings_config)

    def embed_directories(self, source_directories: List[Union[str, Path]]) -> None:
        embeddings = self.get_embedding_function()

        sources_database = ChromaDatabase(
            collection_name=DEFAULT_SOURCES_COLLECTION_NAME,
            embeddings=embeddings,
            config=self.config["chroma"],
        )

        # Get existing sources to not embed them again
        collection = sources_database.client.get()
        all_sources = [
            metadata["source"]
            for metadata in collection["metadatas"]
            if collection.get("metadatas") is not None
        ]

        existing_sources = []
        for source in all_sources:
            if source not in existing_sources:
                existing_sources.append(source)

        source_documents = []
        for source_directory in source_directories:
            # Load source documents
            documents = self.load_documents(
                source_directory, ignored_files=existing_sources
            )
            if len(documents) > 0:
                source_documents.extend(documents)

        if len(source_documents) == 0:
            return

        # Add source documents to the database
        with TimeTaken("Add sources to database"):
            source_ids = sources_database.add_documents(source_documents)

        # Split documents into chunks
        # Get splitter config and fix chunk_size: It must be less than the model max sequence
        # splitter_config = {**self.config["splitter"]}
        # splitter_config["chunk_size"] = min(
        #     splitter_config.get("chunk_size", DEFAULT_CHUNK_SIZE),
        #     embeddings.client.get_max_seq_length(),
        #     )

        # Get child splitter config and fix chunk_size: It must be less than the model max sequence
        child_splitter_config = {
            "separators": ["\n\n", "\n", ".", ",", " "],
            "keep_separator": False,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "chunk_size": DEFAULT_CHUNK_SIZE,
        } | self.config["splitter"]

        # fix chunk_size: It must be less than the model max sequence
        child_splitter_config["chunk_size"] = min(
            child_splitter_config["chunk_size"],
            embeddings.client.get_max_seq_length(),
        )

        # Get parent splitter config: twice the size of the child
        parent_splitter_config = {
            **child_splitter_config,
            "chunk_size": 2 * child_splitter_config["chunk_size"],
            "chunk_overlap": 2 * child_splitter_config["chunk_overlap"],
        }

        # print(f"child_splitter_config: {child_splitter_config}")
        # print(f"parent_splitter_config: {parent_splitter_config}")

        with TimeTaken("Split source to parent chunks"):
            parent_chunks = self.split_documents(
                source_documents, ids=source_ids, **parent_splitter_config
            )

        # Add parent chunks to the database
        if len(parent_chunks) > 0:
            parent_chunks_database = ChromaDatabase(
                collection_name=DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
                embeddings=embeddings,
                config=self.config["chroma"],
            )
            with TimeTaken("Add parent chunks to database"):
                parent_chunk_ids = parent_chunks_database.add_documents(parent_chunks)

            with TimeTaken("Split parent to child chunks"):
                child_chunks = self.split_documents(
                    parent_chunks, ids=parent_chunk_ids, **child_splitter_config
                )

            # Add parent chunks to the database
            if len(child_chunks) > 0:
                child_chunks_database = ChromaDatabase(
                    embeddings=embeddings, config=self.config["chroma"]
                )
                with TimeTaken("Add child chunks to database"):
                    child_chunks_database.add_documents(child_chunks)

    # def embed_directories_with_parent_child_retriever(self, source_directories: List[Union[str, Path]]) -> None:
    #     embeddings: HuggingFaceInstructEmbeddings = self.get_embedding_function()

    #     sources_database = ChromaDatabase(
    #         embeddings=embeddings,
    #         config=self.config["chroma"],
    #         collection_name=DEFAULT_SOURCES_COLLECTION_NAME,
    #     )

    #     # Get existing sources to not embed them again
    #     collection = sources_database.client.get()
    #     all_sources = [
    #         metadata["source"]
    #         for metadata in collection["metadatas"]
    #         if collection.get("metadatas") is not None
    #     ]

    #     existing_sources = []
    #     for source in all_sources:
    #         if source not in existing_sources:
    #             existing_sources.append(source)

    #     source_documents = []
    #     for source_directory in source_directories:
    #         # Load source documents
    #         documents = self.load_documents(
    #             source_directory, ignored_files=existing_sources
    #         )
    #         if len(documents) > 0:
    #             source_documents.extend(documents)

    #     if len(source_documents) == 0:
    #         return

    #     # Add source documents to the database
    #     sources_database.add_documents(source_documents)

    #     # Split documents into chunks
    #     # Get child splitter config and fix chunk_size: It must be less than the model max sequence
    #     child_splitter_config = {
    #         "separators": ["\n\n", "\n", ".", ",", " "],
    #         "keep_separator": False,
    #         "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    #         "chunk_size": DEFAULT_CHUNK_SIZE,
    #     } | self.config["splitter"]

    #     # fix chunk_size: It must be less than the model max sequence
    #     child_splitter_config["chunk_size"] = min(
    #             child_splitter_config["chunk_size"],
    #             embeddings.client.get_max_seq_length(),
    #         )

    #     print(f"child_splitter_config: {child_splitter_config}")

    #     # This text splitter is used to create the child documents
    #     # It should create documents smaller than the parent
    #     child_splitter = RecursiveCharacterTextSplitter(**child_splitter_config)

    #     # Get parent splitter config: twice the size of the child
    #     parent_splitter_config = {
    #         **child_splitter_config,
    #         "chunk_size": 2 * child_splitter_config["chunk_size"],
    #         "chunk_overlap": 2 * child_splitter_config["chunk_overlap"],
    #     }

    #     # This text splitter is used to create the parent documents
    #     parent_splitter = RecursiveCharacterTextSplitter(**parent_splitter_config)

    #     child_database = ChromaDatabase(
    #         embeddings=embeddings,
    #         config=self.config["chroma"]
    #     )

    #     parent_database = ChromaDatabase(
    #         collection_name=DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
    #         embeddings=embeddings,
    #         config=self.config["chroma"]
    #     )

    #     parent_child_retriever = ParentDocumentRetriever(
    #         vectorstore=child_database.client,
    #         docstore=parent_database.client,
    #         child_splitter=child_splitter,
    #         parent_splitter=parent_splitter,
    #     )

    #     parent_child_retriever.add_documents(source_documents)

    #     print("parent_child_retriever.add_documents DONE!")

    @classmethod
    def split_documents(
        cls,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **splitter_kwargs,
    ) -> List[Document]:
        """
        Load documents and split in chunks
        """

        splitter_kwargs = {
            "separators": ["\n\n", "\n", ".", ",", " "],
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "keep_separator": False,
        } | {**splitter_kwargs}

        text_splitter = RecursiveCharacterTextSplitter(
            **splitter_kwargs
        )  # .from_tiktoken_encoder

        texts, metadatas = [], []
        for index, document in enumerate(documents):
            texts.append(document.page_content)
            metadatas.append(
                {
                    **document.metadata,
                    "parent_id": ids[index] if (ids and len(ids) > index) else None,
                }
            )
        with TimeTaken("Run text splitter to create documents"):
            chunks = text_splitter.create_documents(texts, metadatas=metadatas)
        return chunks

    @classmethod
    def load_documents(
        cls, source_dir: str, ignored_files: List[str] = []
    ) -> List[Document]:
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        all_files = []
        for ext in _LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
            )
        filtered_files = [
            file_path for file_path in all_files if file_path not in ignored_files
        ]

        if len(filtered_files) == 0:
            return []

        with Pool(processes=os.cpu_count()) as pool:
            results = []
            with tqdm(
                total=len(filtered_files), desc="Loading new documents", ncols=80
            ) as pbar:
                for i, docs in enumerate(
                    pool.imap_unordered(cls.load_single_document, filtered_files)
                ):
                    results.extend(docs)
                    pbar.update()

        return results

    @classmethod
    def load_single_document(cls, file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in _LOADER_MAPPING:
            loader_class, loader_args = _LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            loader_documents = loader.load()

            return loader_documents

        raise ValueError(f"Unsupported file extension '{ext}'")
