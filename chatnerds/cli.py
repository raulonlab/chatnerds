
import os
from time import sleep
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import typer
import logging
from .config import Config
from .logging import setup as setup_logging
from . import cli_common, cli_nerd, transcriber, youtuber, podcaster, utils

# load config
global_config = Config.environment_instance()

log_terminal_level = logging.DEBUG
if global_config.VERBOSE == 0:
    log_terminal_level = logging.WARNING
elif global_config.VERBOSE == 1:
    log_terminal_level = logging.INFO

setup_logging(
    log_terminal_level=log_terminal_level,
    log_file_level=global_config.LOG_FILE_LEVEL,
    log_file_path=global_config.LOG_FILE_PATH,
)

app = typer.Typer()

# Import commands from other modules
app.add_typer(cli_nerd.app, name="nerd")

# validate and prompt to confirm the active nerd
def validate_confirm_active_nerd(skip_confirmation: bool = False):
    if not global_config.get_active_nerd():
        typer.echo(f"No active nerd set. Please set an active nerd first. See chatnerds nerd --help. ")
        raise typer.Abort()

    if not skip_confirmation:
        prompt_repsonse = cli_common.prompt_active_nerd(active_nerd=global_config.get_active_nerd(), nerd_base_path=global_config.get_nerd_base_path())
        if not prompt_repsonse:
            raise typer.Abort()

@app.command("download", help="Download audio files from sources (youtube and podcasts)")
def download_command():
    validate_confirm_active_nerd()

    # Youtube downloader
    youtube_downloader = youtuber.YoutubeDownloader(source_urls=global_config.get_nerd_youtube_sources(), config=global_config)
    youtube_downloader.run(output_path=str(Path(global_config.get_nerd_base_path(), "downloads", "youtube")))

    # Padcasts downloader
    podcast_downloader = podcaster.PodcastDownloader(feeds=global_config.get_nerd_podcast_sources(), config=global_config)
    podcast_downloader.run(output_path=str(Path(global_config.get_nerd_base_path(), "downloads", "podcasts")))


@app.command("transcribe", help="Transcribe audio files")
def transcribe_command(
    directory_filter: cli_common.DirectoryFilterArgument = None,
    source: cli_common.SourceOption = None,
):
    validate_confirm_active_nerd()

    source_directories = cli_common.get_source_directory_paths(directory_filter=directory_filter, source=source, base_path=global_config.get_nerd_base_path())

     # Audio transcriber
    for source_directory in source_directories:
        audio_transcriber = transcriber.AudioTranscriber(source_directory=str(source_directory), config=global_config)
        logging.info(f"Transcribing directory: {source_directory}")
        audio_transcriber.run()


@app.command("add", help="Add source documents to embeddings DB")
def add_command(
    directory_filter: cli_common.DirectoryFilterArgument = None,
    source: cli_common.SourceOption = None,
):
    if not directory_filter and not source:
        typer.echo("No directory or source specified. Please specify one of them. See chatnerds add --help.")
        raise typer.Abort()

    validate_confirm_active_nerd()

    source_directories = cli_common.get_source_directory_paths(directory_filter=directory_filter, source=source, base_path=global_config.get_nerd_base_path())
    
    from chatdocs.add import add as chatdocs_add
    chatdocs_config = global_config.get_chatdocs_config()
    for source_directory in source_directories:
        logging.info(f"Adding source directory: {source_directory}")
        try:
            chatdocs_add(config=chatdocs_config, source_directory=str(source_directory))
        except SystemExit:
            logging.error(f"Error adding source directory: {source_directory}. Try to load the sources separately with the option --source. See chatnerds add --help.")
            raise typer.Abort()

@app.command("chat", help="Start a chat session with the documents added in the embeddings DB")
def chat_command(
    query: Annotated[
        Optional[str],
        typer.Argument(
            help="The query to use for retrieval. If not specified, runs in interactive mode."
        ),
    ] = None,
):
    validate_confirm_active_nerd()

    from .chatdocs_chat import chat as chatdocs_chat
    chatdocs_config = global_config.get_chatdocs_config()
    chatdocs_chat(config=chatdocs_config, query=query)

@app.command("db", help="Print summary of the embeddings DB")
def db_command():
    validate_confirm_active_nerd(skip_confirmation=True)

    from chatdocs.vectorstores import get_vectorstore
    chatdocs_config = global_config.get_chatdocs_config()
    chroma = get_vectorstore(config=chatdocs_config)

    # Get sources
    collection = chroma.get(include=["metadatas"])
    sources = [metadata["source"] for metadata in collection["metadatas"]]

    # Get distinct documents loaded in the db
    documents = []
    for source in sources:
        document_filename = os.path.basename(source)
        if document_filename not in documents:
            documents.append(document_filename)

    # Print summary
    print("Chroma DB:")
    print(f"  DB Path:       {utils.LogColors.BOLD}{chatdocs_config['chroma']['persist_directory']}{utils.LogColors.ENDC}")
    print(f"  Documents ({len(documents)}):")
    for document in documents:
        print(f"    - {document}")

@app.command("config", help="Print runtime configuration")
def config_command():
    print(str(global_config))
