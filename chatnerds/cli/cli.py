from typing import Optional
from typing_extensions import Annotated
import typer
import logging
from chatnerds.config import Config
from chatnerds.logging_setup import setup as setup_logging
from chatnerds.cli import cli_nerds, cli_sources, cli_tools, cli_utils, cli_db
from chatnerds.utils import get_source_directory_paths
from chatnerds.cli.cli_utils import TqdmHolder

_global_config = Config.environment_instance()

_log_terminal_level = logging.DEBUG
if _global_config.VERBOSE == 0:
    _log_terminal_level = logging.WARNING
elif _global_config.VERBOSE == 1:
    _log_terminal_level = logging.INFO
setup_logging(
    log_terminal_level=_log_terminal_level,
    log_file_level=_global_config.LOG_FILE_LEVEL,
    log_file_path=_global_config.LOG_FILE_PATH,
)

app = typer.Typer(cls=cli_utils.OrderCommands)


# Import commands from other modules
app.registered_commands += cli_nerds.app.registered_commands
app.registered_commands += cli_sources.app.registered_commands


@app.command("study", help="Add source documents to embeddings DB")
def study_command(
    directory_filter: cli_utils.DirectoryFilterArgument = None,
    source: cli_utils.SourceOption = None,
):
    if not directory_filter and not source:
        typer.echo(
            "No directory or source specified. Please specify one of them. See chatnerds add --help."
        )
        raise typer.Abort()

    cli_utils.validate_confirm_active_nerd()

    try:
        source_directories = get_source_directory_paths(
            directory_filter=directory_filter,
            source=source,
            base_path=_global_config.get_nerd_base_path(),
        )

        from chatnerds.document_loaders.document_loader import DocumentLoader

        document_loader = DocumentLoader(
            nerd_config=_global_config.get_nerd_config(),
            source_directories=source_directories,
        )

        tqdm_holder = TqdmHolder(desc="Loading documents", ncols=80)
        document_loader.on("start", tqdm_holder.start)
        document_loader.on("update", tqdm_holder.update)
        document_loader.on("end", tqdm_holder.close)

        document_loader_results, document_loader_errors = document_loader.run()

        tqdm_holder.close()
        logging.info(
            f"{len(document_loader_results)} documents loaded successfully with {len(document_loader_errors)} errors...."
        )

        if len(document_loader_errors) > 0:
            logging.error("Error loading documents", exc_info=document_loader_errors[0])
            return

        from chatnerds.langchain.document_embedder import DocumentEmbedder

        document_embedder = DocumentEmbedder(_global_config.get_nerd_config())

        tqdm_holder = TqdmHolder(desc="Embedding documents", ncols=80)
        document_embedder.on("start", tqdm_holder.start)
        document_embedder.on("update", tqdm_holder.update)
        document_embedder.on("end", tqdm_holder.close)

        document_embedder_results, document_embedder_errors = document_embedder.run(
            documents=document_loader_results,
        )

        tqdm_holder.close()
        logging.info(
            f"{len(document_embedder_results)} documents embedded successfully with {len(document_embedder_errors)} errors...."
        )

        if len(document_embedder_errors) > 0:
            logging.error(
                "Error embedding documents", exc_info=document_embedder_errors[0]
            )

    except Exception as e:
        logging.error(
            "Error studying sources. Try to load the sources separately with the option --source. See chatnerds add --help."
        )
        raise e
    except SystemExit:
        raise typer.Abort()


@app.command(
    "chat", help="Start a chat session with the documents added in the embeddings DB"
)
def chat_command(
    query: Annotated[
        Optional[str],
        typer.Argument(
            help="The query to use for retrieval. If not specified, runs in interactive mode."
        ),
    ] = None,
    # filter: Annotated[
    #     Optional[str],
    #     typer.Option(
    #         "--filter",
    #         "-f",
    #         case_sensitive=False,
    #         help="Filter source documents by a string",
    #     ),
    # ] = None,
):
    cli_utils.validate_confirm_active_nerd()

    from chatnerds.chat import chat

    chat(query=query)


@app.command("config", help="Print runtime configuration")
def config_command():
    print(str(_global_config))


app.add_typer(
    cli_tools.app,
    name="tools",
    help="Other tools",
    short_help="Other tools",
    epilog="* These tools work independently of your active nerd environment.",
)

app.add_typer(
    cli_db.app,
    name="db",
    help="Database operations",
    short_help="Database operations",
    epilog="* These commands require an active nerd environment.",
)
