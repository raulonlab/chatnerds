"""Loads audio file transcription."""

import os
from typing import Iterator, List, Optional
from langchain_core.documents import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.blob_loaders.file_system import FileSystemBlobLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParserLocal


# source: https://github.com/langchain-ai/langchain/pull/8517/files
class OpenAIWhisperLoader(BaseLoader):
    """Loads audio file transcription using OpenAI Whisper.
    OpenAIWhisperParserLocal Source: langchain/document_loaders/parsers/audio.py"""

    def __init__(
        self,
        file_path: str,
        device: Optional[str] = "cpu",
        lang_model: Optional[str] = "openai/whisper-base",
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.device = device
        self.lang_model = lang_model
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

    def _get_loader(self) -> GenericLoader:
        return GenericLoader(
            FileSystemBlobLoader(
                os.path.dirname(self.file_path), glob=os.path.basename(self.file_path)
            ),
            OpenAIWhisperParserLocal(device=self.device, lang_model=self.lang_model),
        )

    def load(self) -> List[Document]:
        """Load audio transcription into Document objects."""
        loader = self._get_loader()
        return loader.load()

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Documents."""
        loader = self._get_loader()
        yield from loader.lazy_load()
