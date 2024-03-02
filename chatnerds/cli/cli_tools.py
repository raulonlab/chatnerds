import logging
from pathlib import Path
from typing import Optional
import typer
from typing_extensions import Annotated
from rich import print as rprint
from rich.markup import escape
from rich.panel import Panel
from chatnerds.cli.cli_utils import UrlFilterArgument, validate_confirm_active_nerd
from chatnerds.config import Config


_global_config = Config.environment_instance()
app = typer.Typer()


@app.command("transcribe-youtube", help="Transcribe youtube video")
def transcribe_youtube_command(
    url: UrlFilterArgument = None,
):
    from chatnerds.tools.youtube_downloader import YoutubeDownloader

    youtube_downloader = YoutubeDownloader(source_urls=[url], config=_global_config)
    audio_output_file_paths = youtube_downloader.run()

    if len(audio_output_file_paths) == 0:
        typer.echo(f"Unable to download audio from url: {url}")
        raise typer.Abort()

    audio_output_file_path = audio_output_file_paths[0]
    source_directory = Path(audio_output_file_path).parent

    from chatnerds.tools.audio_transcriber import AudioTranscriber

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

    from chatnerds.document_loaders.transcript_loader import TranscriptLoader

    try:
        transcript_documents = TranscriptLoader(input_file_path).load()
        print("transcript_documents: ")
        print(transcript_documents)
        print("#########################\n")

        from chatnerds.langchain.document_embeddings import DocumentEmbeddings

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

    from chatnerds.document_loaders.transcript_loader import TranscriptLoader

    try:
        transcript_documents = TranscriptLoader(input_file_path).load()

        from chatnerds.langchain.summarizer import Summarizer

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

    nerd_config = _global_config.get_nerd_config()

    from chatnerds.langchain.document_embeddings import DocumentEmbeddings

    embeddings = DocumentEmbeddings(config=nerd_config).get_embedding_function()

    from chatnerds.langchain.chroma_database import ChromaDatabase

    database = ChromaDatabase(embeddings=embeddings, config=nerd_config["chroma"])

    similar_chunks = database.find_similar_docs(
        query=query,
        k=nerd_config["retriever"]["search_kwargs"].get("k", 4),
        with_score=True,
    )

    # Add score to metadata
    rprint("\n[bold]Similar documents found:", end="", flush=True)
    for doc, score in similar_chunks:
        doc.metadata["score"] = score
        rprint(
            f"\n[bright_blue]Score: [bold]{score}\n\n{escape(doc.page_content)}",
            end="",
            flush=True,
        )

    # Remove duplicates
    unique_ids = set()
    unique_similar_documents = [
        doc
        for doc, score in similar_chunks
        if doc.page_content not in unique_ids and (unique_ids.add(doc) or True)
    ]

    rprint(
        f"\n[bold]Similar documents found: ({len(unique_similar_documents)})",
        end="",
        flush=True,
    )
    for source in unique_similar_documents:
        rprint(
            f"\n[bright_blue]Source: [bold]{escape(source.metadata['source'])} [bright_green](score: {str(source.metadata['score'])})\n",
            end="",
            flush=True,
        )
        metadata = {}
        merged_page_contents = ""
        for doc in unique_similar_documents:
            if doc.metadata["source"] == source.metadata["source"]:
                metadata = doc.metadata
                if merged_page_contents == "":
                    merged_page_contents = doc.page_content
                else:
                    merged_page_contents = (
                        merged_page_contents + "\n\n(...)\n\n" + doc.page_content
                    )

        rprint(
            Panel(
                f"URL: {escape(metadata.get('comment', '-'))}\n\n{escape(merged_page_contents)}"
            )
        )


@app.command("test", help="Test")
def test_command():
    validate_confirm_active_nerd(skip_confirmation=True)

    nerd_config = _global_config.get_nerd_config()

    from chatnerds.langchain.document_embeddings import DocumentEmbeddings

    embeddings = DocumentEmbeddings(config=nerd_config).get_embedding_function()
    print(f"Maximum embedded sequence length: {embeddings.client.get_max_seq_length()}")

    from chatnerds.langchain.chroma_database import ChromaDatabase

    database = ChromaDatabase(embeddings=embeddings, config=nerd_config["chroma"])

    database.print_short_chunks()
