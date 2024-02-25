import logging
from typing import Optional
from chatnerds.langchain.document_embeddings import DocumentEmbeddings
from chatnerds.cli import cli_utils
from chatnerds.config import Config

_global_config = Config.environment_instance()


def study(
    directory_filter: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    if not directory_filter and not source:
        raise Exception("No directory or source specified")

    source_directories = cli_utils.get_source_directory_paths(
        directory_filter=directory_filter,
        source=source,
        base_path=_global_config.get_nerd_base_path(),
    )

    try:
        DocumentEmbeddings(config=_global_config.get_nerd_config()).embed_directories(
            source_directories=source_directories
        )
    except Exception as e:
        logging.error(f"Error adding source directories: {source_directories}.")
        raise e
