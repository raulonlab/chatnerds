from typing import List
from dotenv import dotenv_values

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class TranscriptLoader(BaseLoader):
    """Load transcript files.


    Args:
        file_path: Path to the file to load.
    """

    def __init__(
        self,
        file_path: str,
    ):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load from file path."""

        transcript_dict = dotenv_values(self.file_path)

        page_content = transcript_dict.pop("transcript")
        transcript_dict.pop("summary", None)  # Ignore summary for now
        metadata = {"source": self.file_path, **transcript_dict}

        return [Document(page_content=page_content, metadata=metadata)]
