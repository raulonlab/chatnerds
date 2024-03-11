import logging
from typing import Optional
import typer
from typing_extensions import Annotated
from rich import print as rprint
from rich.markup import escape
from rich.panel import Panel
from chatnerds.cli.cli_utils import (
    OrderedCommandsTyperGroup,
    UrlFilterArgument,
    validate_confirm_active_nerd,
)
from chatnerds.config import Config


_global_config = Config.environment_instance()
app = typer.Typer(cls=OrderedCommandsTyperGroup, no_args_is_help=True)


@app.command("transcribe-youtube", help="Transcribe youtube video")
def transcribe_youtube_command(
    url: UrlFilterArgument = None,
):
    # Local import of YoutubeDownloader
    from chatnerds.tools.youtube_downloader import YoutubeDownloader

    youtube_downloader = YoutubeDownloader(source_urls=[url], config=_global_config)
    audio_output_file_paths = youtube_downloader.run()

    if len(audio_output_file_paths) == 0:
        logging.error(f"Unable to download audio from url: {url}")
        raise typer.Abort()

    # Local import of AudioTranscriber
    from chatnerds.tools.audio_transcriber import AudioTranscriber

    audio_transcriber = AudioTranscriber(config=_global_config)
    rprint(f"Transcribing audio file '{audio_output_file_paths[0]}'...")
    transcript_output_file_paths = audio_transcriber.run(
        source_files=audio_output_file_paths
    )

    if len(transcript_output_file_paths) == 0:
        logging.error(f"Unable to transcript audio file '{audio_output_file_paths[0]}'")
        raise typer.Abort()

    transcript_output_file_path = transcript_output_file_paths[0]
    with open(transcript_output_file_path, "r") as transcript_file:
        transcript = transcript_file.read()
        rprint(f"\n[bold]Transcript of youtube video {url}:", flush=True)
        rprint(Panel(transcript))


@app.command("transcribe-audio", help="Transcribe mp3 audio file")
def transcribe_audio_command(
    input_file_path: str,
    output_path: Optional[str] = None,
):
    # Local import of AudioTranscriber
    from chatnerds.tools.audio_transcriber import AudioTranscriber

    audio_transcriber = AudioTranscriber(config=_global_config)
    rprint(f"Transcribing audio file '{input_file_path}'...")
    results, errors = audio_transcriber.run(
        source_files=[input_file_path],
        output_path=output_path,
        force=True,
    )

    if len(errors) > 0:
        logging.error("Error transcribing audio file", exc_info=errors[0])
        raise typer.Abort()

    transcript_output_file_path = results[0]
    with open(transcript_output_file_path, "r") as transcript_file:
        transcript = transcript_file.read()
        rprint(f"\n[bold]Transcript of audio file '{input_file_path}':", flush=True)
        rprint(Panel(transcript))


@app.command("summarize")
def summarize_command(input_file_path: str):
    nerd_config = _global_config.get_nerd_config()

    # Local import of TranscriptLoader
    from chatnerds.document_loaders.transcript_loader import TranscriptLoader

    try:
        transcript_documents = TranscriptLoader(input_file_path).load()

        # Local import of Summarizer
        from chatnerds.langchain.summarizer import Summarizer

        summarizer = Summarizer(nerd_config)

        rprint(f"Summarizing file '{input_file_path}'...")
        summary = summarizer.summarize_text(transcript_documents[0].page_content)

        rprint(f"\n[bold]Summary of '{input_file_path}':", flush=True)
        rprint(Panel(summary))

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
    if interactive:
        rprint("\nType your query below and press Enter.")
        rprint("Type 'exit' or 'quit' or 'q' to exit the application.\n")

    rprint("[bold]Q: ", flush=True)
    if interactive:
        query = input()
    else:
        rprint(escape(query))
    rprint()
    if query.strip() in ["exit", "quit", "q"]:
        rprint("Exiting...\n")
        return

    rprint("Searching...\n", flush=True)

    nerd_config = _global_config.get_nerd_config()

    # Local import of LLMFactory
    from chatnerds.langchain.llm_factory import LLMFactory

    embeddings = LLMFactory(config=nerd_config).get_embedding_function()

    # Local import of ChromaDatabase
    from chatnerds.langchain.chroma_database import ChromaDatabase

    database = ChromaDatabase(embeddings=embeddings, config=nerd_config["chroma"])

    similar_chunks = database.find_similar_docs(
        query=query,
        k=nerd_config["retriever"]["search_kwargs"].get("k", 4),
        with_score=True,
    )

    # Add score to metadata
    rprint("\n[bold]Similar documents found:", flush=True)
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
