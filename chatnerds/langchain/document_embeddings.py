import os
import glob
from pathlib import Path
from typing import Any, Dict, List, Union
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
from langchain_core.documents import Document
from chatnerds.document_loaders import TranscriptLoader
from chatnerds.langchain.chroma_database import (
    ChromaDatabase,
    DEFAULT_SOURCES_COLLECTION_NAME,
)


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
            embeddings=embeddings,
            config=self.config["chroma"],
            collection_name=DEFAULT_SOURCES_COLLECTION_NAME,
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
        sources_database.add_documents(source_documents)

        # Split documents into chunks
        chunk_documents = self.split_documents(source_documents)

        # Add chunks to the database
        if len(chunk_documents) > 0:
            chunks_database = ChromaDatabase(
                embeddings=embeddings, config=self.config["chroma"]
            )
            chunks_database.add_documents(chunk_documents)

    @classmethod
    def split_documents(cls, documents: List[Document]) -> List[Document]:
        """
        Load documents and split in chunks
        """

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "], chunk_size=900, chunk_overlap=50
        )  # chunk_size=500, chunk_overlap=50

        texts, metadatas = [], []
        for document in documents:
            texts.append(document.page_content)
            metadatas.append(
                {"parent_source": document.metadata["source"], **document.metadata}
            )

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
