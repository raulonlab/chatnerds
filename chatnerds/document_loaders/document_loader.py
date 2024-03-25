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
from chatnerds.stores.store_factory import StoreFactory
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

    def run(self, limit: int = _RUN_TASKS_LIMIT) -> Tuple[List[Document], List[any]]:
        logging.debug("Running document loader...")

        # Get studied documents to exclude them
        studied_sources = []
        with StoreFactory(self.config).get_status_store() as status_store:
            studied_sources = status_store.get_studied_document_ids()

        results: List[Document] = []
        errors = []
        for source_directory in self.source_directories:
            # Load source documents
            documents, errors = self.load_documents(
                source_directory, ignored_files=studied_sources, limit=limit
            )
            if len(documents) > 0:
                results.extend(documents)
            if len(errors) > 0:
                errors.extend(errors)

        return results, errors

    def load_documents(
        self,
        source_dir: str,
        ignored_files: List[str] = [],
        limit: int = _RUN_TASKS_LIMIT,
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
        if not limit or not 0 < limit <= _RUN_TASKS_LIMIT:
            limit = _RUN_TASKS_LIMIT
        if len(filtered_files) > limit:
            logging.warning(
                f"Number of documents to load cut to limit {limit} (out of {len(filtered_files)})"
            )
            filtered_files = filtered_files[:limit]

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
            documents = loader.load()

            return documents

        raise ValueError(f"Unsupported file extension '{ext}'")
