import typer
from typing_extensions import Annotated
from chatnerds.cli.cli_utils import validate_confirm_active_nerd, grep_match
from chatnerds.langchain.chroma_database import (
    ChromaDatabase,
    DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
    DEFAULT_SOURCES_COLLECTION_NAME,
)
from chatnerds.enums import LogColors
from chatnerds.config import Config

_global_config = Config.environment_instance()
app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        summary_command()


@app.command("summary", help="Print a summary of the embeddings DB")
def summary_command():
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    sources_db = ChromaDatabase(
        collection_name=DEFAULT_SOURCES_COLLECTION_NAME, config=nerd_config["chroma"]
    )
    parent_chunks_db = ChromaDatabase(
        collection_name=DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
        config=nerd_config["chroma"],
    )
    child_chunks_db = ChromaDatabase(config=nerd_config["chroma"])

    sources_collection = sources_db.client.get(include=["metadatas"])
    parent_chunks_collection = parent_chunks_db.client.get(
        include=["metadatas", "documents"]
    )
    child_chunks_collection = child_chunks_db.client.get(
        include=["metadatas", "documents"]
    )

    # Print summary
    print("DB Summary:")
    print(
        f"- DB Path:              {LogColors.BOLD}{nerd_config['chroma']['persist_directory']}{LogColors.ENDC}"
    )
    if hasattr(sources_db.client._collection._client, "max_batch_size"):
        print(
            f"- max_batch_size: {LogColors.BOLD}{sources_db.client._collection._client.max_batch_size}{LogColors.ENDC}"
        )
    print(
        f"- Num source documents: {LogColors.BOLD}{len(sources_collection['metadatas'])}{LogColors.ENDC}"
    )
    print(
        f"- Num parent chunks:    {LogColors.BOLD}{len(parent_chunks_collection['metadatas'])}{LogColors.ENDC}"
    )
    print(
        f"- Num child chunks:     {LogColors.BOLD}{len(child_chunks_collection['metadatas'])}{LogColors.ENDC}"
    )


@app.command(
    "sources", help="Print all the source documents stored in the embeddings DB"
)
def sources_command(
    grep: Annotated[
        str,
        typer.Option(
            "--grep",
            "-g",
            case_sensitive=False,
            help="Filter source documents by a string",
        ),
    ]
):
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    sources_db = ChromaDatabase(
        collection_name=DEFAULT_SOURCES_COLLECTION_NAME, config=nerd_config["chroma"]
    )
    sources_collection = sources_db.client.get(include=["metadatas"])

    print(f"Source documents ({len(sources_collection['metadatas'])}):")
    for source_i, source_metadata in enumerate(sources_collection["metadatas"]):
        if not grep_match(
            grep,
            source_metadata["source"],
            source_metadata["artist"],
            source_metadata["title"],
            source_metadata["album"],
        ):
            continue

        pretty_artist = source_metadata.get("artist", "")
        if not pretty_artist:
            pretty_artist = source_metadata.get("album", "")
        if not pretty_artist:
            pretty_artist = f"{LogColors.DISABLED}(Unknown artist){LogColors.ENDC}"

        pretty_title = source_metadata.get("title", "")
        if not pretty_title:
            pretty_title = f"{LogColors.DISABLED}(Unknown title){LogColors.ENDC}"

        pretty_url = source_metadata.get("comment", "")
        if not pretty_url:
            pretty_url = f"{LogColors.DISABLED}(no url){LogColors.ENDC}"

        pretty_source = str(source_metadata.get("source", "")).removeprefix(
            str(_global_config.get_nerd_base_path())
        )
        if not pretty_source:
            pretty_source = f"{LogColors.DISABLED}(Source not found){LogColors.ENDC}"

        print(f"- {LogColors.BOLDSOURCE}{pretty_source}{LogColors.ENDC}")
        print(
            f"  {pretty_artist} - {pretty_title}\n  {LogColors.UNDERLINE}{pretty_url}{LogColors.ENDC}"
        )


@app.command(
    "chunks",
    help="Print all the source documents with children chunk stats stored in the embeddings DB",
)
def chunks_command(
    grep: Annotated[
        str,
        typer.Option(
            "--grep",
            "-g",
            case_sensitive=False,
            help="Filter source documents by a string",
        ),
    ]
):
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    sources_db = ChromaDatabase(
        collection_name=DEFAULT_SOURCES_COLLECTION_NAME, config=nerd_config["chroma"]
    )
    parent_chunks_db = ChromaDatabase(
        collection_name=DEFAULT_PARENT_CHUNKS_COLLECTION_NAME,
        config=nerd_config["chroma"],
    )
    child_chunks_db = ChromaDatabase(config=nerd_config["chroma"])

    sources_collection = sources_db.client.get(include=["metadatas"])
    parent_chunks_collection = parent_chunks_db.client.get(
        include=["metadatas", "documents"]
    )
    child_chunks_collection = child_chunks_db.client.get(
        include=["metadatas", "documents"]
    )

    print(f"Source documents ({len(sources_collection['metadatas'])}):")
    for source_i, source_metadata in enumerate(sources_collection["metadatas"]):
        if not grep_match(
            grep,
            source_metadata["source"],
            source_metadata["artist"],
            source_metadata["title"],
            source_metadata["album"],
        ):
            continue

        pretty_artist = source_metadata.get("artist", "")
        if not pretty_artist:
            pretty_artist = source_metadata.get("album", "")
        if not pretty_artist:
            pretty_artist = f"{LogColors.DISABLED}(Unknown artist){LogColors.ENDC}"

        pretty_title = source_metadata.get("title", "")
        if not pretty_title:
            pretty_title = f"{LogColors.DISABLED}(Unknown title){LogColors.ENDC}"

        pretty_url = source_metadata.get("comment", "")
        if not pretty_url:
            pretty_url = f"{LogColors.DISABLED}(no url){LogColors.ENDC}"

        pretty_source = str(source_metadata.get("source", "")).removeprefix(
            str(_global_config.get_nerd_base_path())
        )
        if not pretty_source:
            pretty_source = f"{LogColors.DISABLED}(Source not found){LogColors.ENDC}"

        print(f"- {LogColors.BOLDSOURCE}{pretty_source}{LogColors.ENDC}")
        print(
            f"  {pretty_artist} - {pretty_title}\n  {LogColors.UNDERLINE}{pretty_url}{LogColors.ENDC}"
        )
        # for key, value in source_metadata.items():
        #     if key != "source":
        #         print(f"  {key}: {value}")
        # Get parent chunks count
        count_parent_chunks = 0
        count_parent_characters = 0
        for parent_i, parent_metadata in enumerate(
            parent_chunks_collection["metadatas"]
        ):
            if parent_metadata["parent_id"] == sources_collection["ids"][source_i]:
                count_parent_chunks += 1
                count_parent_characters += len(
                    parent_chunks_collection["documents"][parent_i]
                )
        print(f"  num parent chunks: {count_parent_chunks}")
        print(
            f"  avg parent chunk size: {count_parent_characters / count_parent_chunks}"
        )

        # Get child chunks count
        count_child_chunks = 0
        count_child_characters = 0
        for child_i, child_metadata in enumerate(child_chunks_collection["metadatas"]):
            if child_metadata["source"] == source_metadata["source"]:
                count_child_chunks += 1
                count_child_characters += len(
                    child_chunks_collection["documents"][child_i]
                )
        print(f"  num child chunks: {count_child_chunks}")
        print(f"  avg child chunk size: {count_child_characters / count_child_chunks}")
