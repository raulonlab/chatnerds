from typing import Optional
from typing_extensions import Annotated
import typer
import logging
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from chatnerds.config import Config
from chatnerds.logging_setup import setup as setup_logging
from chatnerds.cli import cli_nerds, cli_sources, cli_tools, cli_utils, cli_db
from chatnerds.utils import get_filtered_directories
from chatnerds.tools.chat_logger import ChatLogger

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

app = typer.Typer(cls=cli_utils.OrderedCommandsTyperGroup, no_args_is_help=True)


# Import commands from other modules
app.registered_commands += cli_nerds.app.registered_commands
app.registered_commands += cli_sources.app.registered_commands


# Default command
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        app.show_help(ctx)


@app.command(
    "study",
    help="Start studying (ingesting) the source documents and save the embeddings in the local DB",
)
def study_command(
    directory_filter: cli_utils.DirectoryFilterArgument = None,
):
    cli_utils.validate_confirm_active_nerd()

    try:
        source_directories = get_filtered_directories(
            directory_filter=directory_filter,
            base_path=Path(_global_config.get_nerd_base_path(), "source_documents"),
        )

        from chatnerds.document_loaders.document_loader import DocumentLoader

        document_loader = DocumentLoader(
            nerd_config=_global_config.get_nerd_config(),
            source_directories=source_directories,
        )

        tqdm_holder = cli_utils.TqdmHolder(desc="Loading documents", ncols=80)
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

        tqdm_holder = cli_utils.TqdmHolder(desc="Embedding documents", ncols=80)
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


@app.command("chat", help="Start an interactive chat session with your active nerd")
def chat_command(
    query: Annotated[
        Optional[str],
        typer.Argument(
            help="Send a one-off query to your active nerd and exit. If not specified, runs in interactive mode."
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


@app.command("review", help="Append a review value to the last chat log")
def review_command(
    review_value: Annotated[
        Optional[int],
        typer.Argument(
            help="Value of the review in the range [1, 5]",
        ),
    ] = None,
):
    if not review_value:
        review_value_response = typer.prompt(
            "Review of the last chat iteration [1..5]",
            type=int,
        )
        review_value = int(review_value_response)

    if review_value < 1:
        review_value = 1
    elif review_value > 5:
        review_value = 5

    chat_logger = ChatLogger()
    chat_logger.append_to_log("review", review_value)


@app.command("env", help="Print the current value of environment variables")
def env_command(
    default: Annotated[
        Optional[bool],
        typer.Option(
            "--default",
            "-d",
            help="Print the default value of environment variables instead of the current ones.",
        ),
    ] = False
):

    console = Console()

    if default:
        env_string = (
            f"# Default environment configuration\n{str(Config.default_instance())}"
        )
    else:
        env_string = f"# Current environment configuration\n{str(_global_config)}"

    syntax = Syntax(env_string, "ini", line_numbers=False)  # , theme="monokai"
    console.print(syntax)


@app.command("config", help="Print the active nerd configuration (config.yaml)")
def config_command():

    console = Console()

    config_yaml = yaml.safe_dump(
        _global_config.get_nerd_config(), stream=None, default_flow_style=False
    )
    config_string = f"# Active configuration\n{config_yaml}"

    syntax = Syntax(config_string, "yaml", line_numbers=False)  # , theme="monokai"
    console.print(syntax)


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
    help="Other commands related with the local DB",
    short_help="Other commands related with the local DB",
    epilog="* These commands require an active nerd environment.",
)
