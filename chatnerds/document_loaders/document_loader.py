import os
from pathlib import Path
import glob
import logging
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from chatnerds.document_loaders.transcript_loader import TranscriptLoader
from chatnerds.langchain.chroma_database import (
    ChromaDatabase,
    DEFAULT_SOURCES_COLLECTION_NAME,
)
from chatnerds.tools.event_emitter import EventEmitter


_RUN_TASKS_LIMIT = 1_000  # Maximum number of tasks to run in a single call to run()


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


class DocumentLoader(EventEmitter):
    config: Dict[str, Any] = {}
    source_directories: List[str] = []

    def __init__(
        self, nerd_config: Dict[str, Any], source_directories: List[str | Path]
    ):
        super().__init__()
        self.config = nerd_config
        self.source_directories = [str(directory) for directory in source_directories]

    def run(self) -> Tuple[List[Document], List[any]]:
        logging.debug("Running document loader...")

        logging.debug("Finding documents loaded...")
        sources_database = ChromaDatabase(
            collection_name=DEFAULT_SOURCES_COLLECTION_NAME,
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

        results: List[Document] = []
        errors = []
        for source_directory in self.source_directories:
            # Load source documents
            documents, errors = self.load_documents(
                source_directory, ignored_files=existing_sources
            )
            if len(documents) > 0:
                results.extend(documents)
            if len(errors) > 0:
                errors.extend(errors)

        return results, errors

    def load_documents(
        self, source_dir: str, ignored_files: List[str] = []
    ) -> Tuple[List[Document], List[any]]:
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
            return [], []

        # Limit number of tasks to run
        if len(filtered_files) > _RUN_TASKS_LIMIT:
            filtered_files = filtered_files[:_RUN_TASKS_LIMIT]
            logging.warning(
                f"Number of documents to load exceeds limit of {_RUN_TASKS_LIMIT}. Only processing first {_RUN_TASKS_LIMIT} videos."
            )

        # Emit start event (show progress bar in UI)
        self.emit(
            "start", len(filtered_files), desc=f"Loading {os.path.basename(source_dir)}"
        )

        results = []
        errors = []
        max_workers = max(1, os.cpu_count() // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            response_futures = [
                executor.submit(DocumentLoader.load_single_document, file_argument)
                for file_argument in filtered_files
            ]

            for response_future in as_completed(response_futures):
                try:
                    response = response_future.result()
                    if response:
                        results.extend(response)
                except Exception as err:
                    errors.append(err)
                    continue
                finally:
                    self.emit("update")

        self.emit("end")
        return results, errors

    @classmethod
    def load_single_document(cls, file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in _LOADER_MAPPING:
            loader_class, loader_args = _LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            loader_documents = loader.load()

            return loader_documents

        raise ValueError(f"Unsupported file extension '{ext}'")
