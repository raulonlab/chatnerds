import logging
from pathlib import Path
from typing import Optional
import typer
from typing_extensions import Annotated
from rich import print as rprint
from rich.markup import escape
from rich.panel import Panel
from chatnerds.cli import cli_utils
from chatnerds.langchain.document_embeddings import DocumentEmbeddings
from chatnerds.langchain.chroma_database import ChromaDatabase
from chatnerds.document_loaders.transcript_loader import TranscriptLoader
from chatnerds.langchain.document_embeddings import DocumentEmbeddings
from chatnerds.tools.audio_transcriber import AudioTranscriber
from chatnerds.tools.youtube_downloader import YoutubeDownloader
from chatnerds.langchain.summarizer import Summarizer
from chatnerds.config import Config

_global_config = Config.environment_instance()

app = typer.Typer()


@app.command("transcribe-youtube", help="Transcribe youtube video")
def transcribe_youtube_command(
    url: cli_utils.UrlFilterArgument = None,
):
    youtube_downloader = YoutubeDownloader(source_urls=[url], config=_global_config)
    audio_output_file_paths = youtube_downloader.run()

    if len(audio_output_file_paths) == 0:
        typer.echo(f"Unable to download audio from url: {url}")
        raise typer.Abort()

    audio_output_file_path = audio_output_file_paths[0]
    source_directory = Path(audio_output_file_path).parent
    audio_transcriber = AudioTranscriber(
        source_directory=str(source_directory), config=_global_config
    )
    logging.info(f"Transcribing directory: {source_directory}")
    transcript_output_file_paths = audio_transcriber.run()

    if len(transcript_output_file_paths) == 0:
        typer.echo(f"Unable to transcript audio file: {audio_output_file_path}")
        raise typer.Abort()

    transcript_output_file_path = transcript_output_file_paths[0]
    with open(transcript_output_file_path, "r") as transcript_file:
        transcript = transcript_file.read()
        print(f"Transcript of youtube video {url}:\n\n{transcript}")


@app.command("split")
def split_command(input_file_path: str):
    nerd_config = _global_config.get_nerd_config()
    try:
        transcript_documents = TranscriptLoader(input_file_path).load()
        chunks = DocumentEmbeddings(config=nerd_config).split_documents(
            transcript_documents
        )

        print(f"Split {input_file_path}:")
        print("#########################")
        for chunk in chunks:
            print(chunk.page_content)
            print("-----------------------------")

    except Exception as e:
        logging.error(
            f"Error running test split with input file {input_file_path}:\n{e}"
        )
        raise typer.Abort()


@app.command("summarize")
def summarize_command(input_file_path: str):
    nerd_config = _global_config.get_nerd_config()
    try:
        transcript_documents = TranscriptLoader(input_file_path).load()

        summarizer = Summarizer(nerd_config)
        summary = summarizer.summarize_text(transcript_documents[0].page_content)

        print(f"Summarize {input_file_path}:")
        print("#########################")
        print(summary)

    except Exception as e:
        logging.error(
            f"Error running test summarize with input file {input_file_path}:\n{e}"
        )
        raise typer.Abort()


@app.command(
    "search", help="Search similar documents from a query in the embeddings DB"
)
def search_command(
    query: Annotated[
        Optional[str],
        typer.Argument(
            help="The query to use for search. If not specified, runs in interactive mode."
        ),
    ] = None,
):
    cli_utils.validate_confirm_active_nerd(skip_confirmation=True)

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

    nerd_config = _global_config.get_nerd_config()

    embeddings = DocumentEmbeddings(config=nerd_config).get_embedding_function()

    database = ChromaDatabase(embeddings=embeddings, config=nerd_config["chroma"])

    similar_chunks = database.find_similar_docs(query="hello", k=5)

    # Get distinct sources documents
    distinct_similar_sources = []
    for doc in similar_chunks:
        if doc.metadata["source"] not in distinct_similar_sources:
            distinct_similar_sources.append(doc.metadata["source"])

    rprint(
        f"\n[bold]Similar documents found: ({len(distinct_similar_sources)})",
        end="",
        flush=True,
    )
    for source in distinct_similar_sources:
        rprint(f"\n[bright_blue]Source: [bold]{escape(source)}\n", end="", flush=True)
        metadata = {}
        merged_page_contents = ""
        for doc in similar_chunks:
            if doc.metadata["source"] == source:
                metadata = doc.metadata
                if merged_page_contents == "":
                    merged_page_contents = doc.page_content
                else:
                    merged_page_contents = (
                        merged_page_contents + "\n\n(...)\n\n" + doc.page_content
                    )

        print(
            Panel(
                f"URL: {escape(metadata.get('comment', '-'))}\n\n{escape(merged_page_contents)}"
            )
        )
