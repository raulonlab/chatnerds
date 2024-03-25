from typing import Optional
import typer
import logging
from typing_extensions import Annotated
from chatnerds.cli.cli_utils import (
    OrderedCommandsTyperGroup,
    validate_confirm_active_nerd,
    grep_match,
)
from chatnerds.langchain.llm_factory import LLMFactory
from chatnerds.stores.store_factory import StoreFactory
from chatnerds.document_loaders.document_loader import DocumentLoader
from chatnerds.lib.enums import LogColors
from chatnerds.config import Config

_global_config = Config.environment_instance()
app = typer.Typer(cls=OrderedCommandsTyperGroup, no_args_is_help=True)


# Default command: summary
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        status_command()


@app.command(
    "status",
    help="Print a summary of the embeddings DB including invalid sources and duplicated sources (if any)",
)
def status_command():
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    store_factory = StoreFactory(nerd_config)
    chunks_store = store_factory.get_vector_store()

    with store_factory.get_status_store() as status_store:
        database_path = str(status_store.database_path)
        num_studied_documents = len(status_store.get_studied_documents())
        pragmas = status_store.get_pragma_compile_options()

    try:
        chunks_collection = chunks_store.get()
        num_chunk_documents = len(chunks_collection["metadatas"])
    except NotImplementedError:
        num_chunk_documents = "(Not supported by the store)"

    print("SQLite compile options:")
    if pragmas:
        print(
            f"- THREADSAFE:        {LogColors.BOLD}{pragmas.get('THREADSAFE', '?')}{LogColors.ENDC}"
        )

    print("Status DB Summary:")
    print(f"- SQlite DB Path:        {LogColors.BOLD}{database_path}{LogColors.ENDC}")
    print(
        f"- Num studied documents: {LogColors.BOLD}{num_studied_documents}{LogColors.ENDC}"
    )
    print(
        f"- Num studied chunks:    {LogColors.BOLD}{num_chunk_documents}{LogColors.ENDC}"
    )


@app.command(
    "sources", help="Print all the source documents stored in the embeddings DB"
)
def sources_command(
    grep: Annotated[
        Optional[str],
        typer.Option(
            "--grep",
            "-g",
            case_sensitive=False,
            help="Filter source documents by a string",
        ),
    ] = None,
):
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    store_factory = StoreFactory(nerd_config)

    studied_documents = []
    with store_factory.get_status_store() as status_store:
        studied_documents = status_store.get_studied_documents()

    print(f"Nerd base path: {_global_config.get_nerd_base_path()}")
    print(f"Source documents ({len(studied_documents)}):")
    for studied_document in studied_documents:
        metadata = studied_document.get("metadata", {})
        source = studied_document.get("source", "")
        if grep and not grep_match(
            grep,
            source,
            metadata.get("artist", ""),
            metadata.get("title", ""),
            metadata.get("album", ""),
        ):
            continue

        pretty_artist = metadata.get("artist", "")
        if not pretty_artist:
            pretty_artist = metadata.get("album", "")
        if not pretty_artist:
            pretty_artist = f"{LogColors.DISABLED}(Unknown artist){LogColors.ENDC}"

        pretty_title = metadata.get("title", "")
        if not pretty_title:
            pretty_title = f"{LogColors.DISABLED}(Unknown title){LogColors.ENDC}"

        pretty_url = metadata.get("comment", "")
        if not pretty_url:
            pretty_url = f"{LogColors.DISABLED}(no url){LogColors.ENDC}"

        pretty_source = str(source).removeprefix(
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

    store_factory = StoreFactory(nerd_config)

    studied_documents = []
    with store_factory.get_status_store() as status_store:
        studied_documents = status_store.get_studied_documents()

    print(f"Nerd base path: {_global_config.get_nerd_base_path()}")
    print(f"Source documents ({len(studied_documents)}):")
    for studied_document in studied_documents:
        metadata = studied_document.get("metadata", {})
        source = studied_document.get("source", "")
        if grep and not grep_match(
            grep,
            source,
            metadata.get("artist", ""),
            metadata.get("title", ""),
            metadata.get("album", ""),
        ):
            continue

        pretty_artist = metadata.get("artist", "")
        if not pretty_artist:
            pretty_artist = metadata.get("album", "")
        if not pretty_artist:
            pretty_artist = f"{LogColors.DISABLED}(Unknown artist){LogColors.ENDC}"

        pretty_title = metadata.get("title", "")
        if not pretty_title:
            pretty_title = f"{LogColors.DISABLED}(Unknown title){LogColors.ENDC}"

        pretty_url = metadata.get("comment", "")
        if not pretty_url:
            pretty_url = f"{LogColors.DISABLED}(no url){LogColors.ENDC}"

        pretty_source = str(source).removeprefix(
            str(_global_config.get_nerd_base_path())
        )
        if not pretty_source:
            pretty_source = f"{LogColors.DISABLED}(Source not found){LogColors.ENDC}"

        print(f"- {LogColors.BOLDSOURCE}{pretty_source}{LogColors.ENDC}")
        print(
            f"  {pretty_artist} - {pretty_title}\n  {LogColors.UNDERLINE}{pretty_url}{LogColors.ENDC}"
        )

        chunks_store = store_factory.get_vector_store()
        try:
            # Get chunk chunks stats
            chunks_collection = chunks_store.get(
                where={"source": source}, include=["metadatas", "documents"]
            )
            count_chunks = 0
            count_chunk_characters = 0
            for chunk_i, chunk_metadata in enumerate(chunks_collection["metadatas"]):
                if chunk_metadata["source"] == source:
                    count_chunks += 1
                    count_chunk_characters += len(
                        chunks_collection["documents"][chunk_i]
                    )
                    # chunk_metadatas.append(chunk_metadata)
            print(f"  num chunk chunks: {count_chunks}")
            if count_chunks > 0:
                print(
                    f"  avg chunk chunk size: {count_chunk_characters / count_chunks}"
                )
            # print("  chunk chunk metadatas:")
            # for chunk_metadata_i, chunk_metadata in enumerate(chunk_metadatas):
            #     print(f"    chunk #{chunk_metadata_i}:")
            #     for chunk_key, chunk_value in chunk_metadata.items():
            #         if chunk_key != "source":
            #             print(f"      - {chunk_key}: {chunk_value}")
        except NotImplementedError:
            print("  chunks information not available for the current vector store")


@app.command("fix-store", help="Fix inconsistent data in stores")
def fix_store_command():
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    store_factory = StoreFactory(nerd_config)
    chunks_store = store_factory.get_vector_store()

    with store_factory.get_status_store() as status_store:
        studied_sources = status_store.get_studied_document_ids()

        try:
            chunks_collection = chunks_store.get(include=["metadatas"])
        except NotImplementedError:
            logging.error("Not supported by the store")
            typer.Abort()
            return

        chunk_sources = set()  # Remember sources of existing chunks

        for chunk_metadata in chunks_collection["metadatas"]:
            source = chunk_metadata.get("source", None)

            if not source:
                logging.error(
                    "Chunk without source found!. Please, remove the store and re-study the source documents"
                )
                typer.Abort()
                return

            chunk_sources.add(source)

            if source not in studied_sources:
                # Try to load source document
                try:
                    source_documents = DocumentLoader.load_single_document(source)

                except Exception as e:
                    logging.error(f"Error loading source document {source}", exc_info=e)
                    typer.Abort()
                    return

                status_store.add_studied_document(
                    id=source,
                    source=source,
                    page_content=source_documents[0].page_content,
                    metadata=source_documents[0].metadata,
                )

                studied_sources.add(source)

                print(f"Added '{source}' to status store")

        # Print sources whitout chunks
        for source in studied_sources:
            if source not in chunk_sources:
                print(
                    f"Source '{source}' does not have chunks in the store. Delete it with command 'delete-source' and study it again."
                )

    print("Done")


@app.command("delete-source", help="Delete documents of a source from the DB")
def delete_source_command(
    source: str,
):
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    store_factory = StoreFactory(nerd_config)
    chunks_store = store_factory.get_vector_store()

    try:
        chunks_collection = chunks_store.get(include=["metadatas"])
    except NotImplementedError:
        logging.error("Not supported by the store")
        typer.Abort()
        return

    chunks_collection = chunks_store.get(
        include=["metadatas"], where={"source": source}
    )

    if len(chunks_collection["ids"]) > 0:
        try:
            chunks_store.delete(ids=chunks_collection["ids"])
            print("Chunk documents deleted successfully...")
        except Exception as e:
            logging.error("Error deleting chunk documents", exc_info=e)
            typer.Abort()
            return

        try:
            with store_factory.get_status_store() as status_store:
                status_store.delete_studied_document(source)
                print("Source document deleted successfully...")
        except Exception as e:
            logging.error("Error deleting chunk documents", exc_info=e)
            typer.Abort()
            return


@app.command(
    "embeddings-parameters",
    help="Print embeddings parameter like max_seq_length and sentence_embedding_dimension",
)
def embeddings_parameters_command():
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    from langchain_core.embeddings import Embeddings

    embeddings: Embeddings = LLMFactory(nerd_config).get_embedding_function()

    print("max_seq_length: ", embeddings.client.get_max_seq_length())
    print(
        "sentence_embedding_dimension: ",
        embeddings.client.get_sentence_embedding_dimension(),
    )
