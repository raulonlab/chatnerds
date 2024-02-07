import os
from time import sleep
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import typer
import logging
from rich import print
from rich.markup import escape
from rich.panel import Panel
from .config import Config
from .logging_setup import setup as setup_logging
from . import cli_common, cli_nerd, cli_test, transcriber, youtuber, podcaster, utils

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
app.add_typer(cli_test.app, name="test")

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
def download_command(source: cli_common.SourceOption = None):
    validate_confirm_active_nerd()

    # Youtube downloader
    if not source or source == cli_common.SourceEnum.youtube:
        youtube_downloader = youtuber.YoutubeDownloader(source_urls=global_config.get_nerd_youtube_sources(), config=global_config)
        youtube_downloader.run(output_path=str(Path(global_config.get_nerd_base_path(), "downloads", "youtube")))

    # Padcasts downloader
    if not source or source == cli_common.SourceEnum.podcasts:
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
    
    from .document_embeddings import DocumentEmbeddings
    nerd_config = global_config.get_nerd_config()
    try:
        DocumentEmbeddings(config=nerd_config).embed_directories(source_directories=source_directories)
    except SystemExit:
        logging.error(f"Error adding source directories: {source_directories}. Try to load the sources separately with the option --source. See chatnerds add --help.")
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

    from .chat import chat
    nerd_config = global_config.get_nerd_config()
    chat(config=nerd_config, query=query)


@app.command("youtube", help="Transcribe youtube video")
def youtube_command(
    url: cli_common.UrlFilterArgument = None,
):
    youtube_downloader = youtuber.YoutubeDownloader(source_urls=[url], config=global_config)
    audio_output_file_paths = youtube_downloader.run()

    if (len(audio_output_file_paths) == 0):
        typer.echo(f"Unable to download audio from url: {url}")
        raise typer.Abort()
    
    audio_output_file_path = audio_output_file_paths[0]
    source_directory = Path(audio_output_file_path).parent
    audio_transcriber = transcriber.AudioTranscriber(source_directory=str(source_directory), config=global_config)
    logging.info(f"Transcribing directory: {source_directory}")
    transcript_output_file_paths = audio_transcriber.run()

    if (len(transcript_output_file_paths) == 0):
        typer.echo(f"Unable to transcript audio file: {audio_output_file_path}")
        raise typer.Abort()

    transcript_output_file_path = transcript_output_file_paths[0]
    with open(transcript_output_file_path, "r") as transcript_file:
        transcript = transcript_file.read()
        print(f"Transcript of youtube video {url}:\n\n{transcript}")


@app.command("podcast", help="Transcribe podcast")
def podcast_command(
    url: cli_common.UrlFilterArgument = None,
):
    podcast_downloader = podcaster.PodcastDownloader(feeds=[url], config=global_config)
    podcast_downloader.run()


@app.command("db", help="Print summary of the embeddings DB")
def db_command():
    validate_confirm_active_nerd(skip_confirmation=True)

    from .document_embeddings import DocumentEmbeddings
    from .chroma_database import ChromaDatabase
    nerd_config = global_config.get_nerd_config()

    embeddings = DocumentEmbeddings(config=nerd_config).get_embeddings()

    database = ChromaDatabase(embeddings=embeddings, config=nerd_config["chroma"])

    collection = database.client.get(include=["metadatas"])

    # Get (distinct) documents loaded in the db
    distinct_documents = []
    for metadata in collection["metadatas"]:
        metadata["source"] = os.path.basename(metadata["source"])
        if (metadata["source"] not in map(lambda x: x["source"], distinct_documents)):
            distinct_documents.append(metadata)

    # Print summary
    print("Chroma DB information:")
    print(f"DB Path:       {utils.LogColors.BOLD}{nerd_config['chroma']['persist_directory']}{utils.LogColors.ENDC}")
    print(f"Documents ({len(distinct_documents)}):")
    for document in distinct_documents:
        print(f"  - {document['source']}")
        for key, value in document.items():
            if key != "source":
                print(f"      - {key}: {value}")


@app.command("search", help="Search similar documents from a query in the embeddings DB")
def search_command(
    query: Annotated[
        Optional[str],
        typer.Argument(
            help="The query to use for search. If not specified, runs in interactive mode."
        ),
    ] = None,
):
    validate_confirm_active_nerd(skip_confirmation=True)

    interactive = not query
    print()
    if interactive:
        print("Type your query below and press Enter.")
        print("Type 'exit' or 'quit' or 'q' to exit the application.\n")

    print("[bold]Q: ", end="", flush=True)
    if interactive:
        query = input()
    else:
        print(escape(query))
    print()
    if query.strip() in ["exit", "quit", "q"]:
        print("Exiting...\n")
        return

    print("Searching...\n", end="", flush=True)

    from .document_embeddings import DocumentEmbeddings
    from .chroma_database import ChromaDatabase
    nerd_config = global_config.get_nerd_config()

    embeddings = DocumentEmbeddings(config=nerd_config).get_embeddings()

    database = ChromaDatabase(embeddings=embeddings, config=nerd_config["chroma"])

    similar_chunks = database.find_similar_docs(query="hello", k=5)

    # Get distinct sources documents
    distinct_similar_sources = []
    for doc in similar_chunks:
        if (doc.metadata["source"] not in distinct_similar_sources):
            distinct_similar_sources.append(doc.metadata["source"])

    print(f"\n[bold]Similar documents found: ({len(distinct_similar_sources)})", end="", flush=True)
    for source in distinct_similar_sources:
        print(f"\n[bright_blue]Source: [bold]{escape(source)}\n", end="", flush=True)
        metadata = {}
        merged_page_contents = ""
        for doc in similar_chunks:
            if doc.metadata["source"] == source:
                metadata = doc.metadata
                if (merged_page_contents == ""):
                    merged_page_contents = doc.page_content
                else:
                    merged_page_contents = merged_page_contents + "\n\n(...)\n\n" + doc.page_content

        print(
            Panel(
                f"URL: {escape(metadata.get('comment', '-'))}\n\n{escape(merged_page_contents)}"
            )
        )


@app.command("config", help="Print runtime configuration")
def config_command():
    print(str(global_config))

