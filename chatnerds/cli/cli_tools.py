import os
import logging
from typing import Optional
import typer
from typing_extensions import Annotated
from rich import print as rprint
from rich.markup import escape
from rich.panel import Panel
from pyfzf.pyfzf import FzfPrompt
import glob
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from chatnerds.cli.cli_utils import (
    OrderedCommandsTyperGroup,
    UrlFilterArgument,
    validate_confirm_active_nerd,
)
from chatnerds.langchain.llm_factory import LLMFactory
from chatnerds.langchain.chain_factory import ChainFactory
from chatnerds.config import Config


_global_config = Config.environment_instance()
app = typer.Typer(cls=OrderedCommandsTyperGroup, no_args_is_help=True)


@app.command("download-youtube", help="Downloads audio file (.mp3) of youtube video")
def download_youtube_command(
    url: UrlFilterArgument = None,
):
    # Local import of YoutubeDownloader
    from chatnerds.tools.youtube_downloader import YoutubeDownloader

    try:
        youtube_downloader = YoutubeDownloader(source_urls=[url], config=_global_config)
        audio_output_file_paths = youtube_downloader.run()

        if len(audio_output_file_paths) == 0:
            logging.error(f"Unable to download audio from url: {url}")
            raise typer.Abort()

        rprint(
            f"\n[bold]Audio downloaded successfully in: {audio_output_file_paths}",
            flush=True,
        )

    except SystemExit:
        raise typer.Abort()


@app.command("transcribe-audio", help="Transcribe mp3 audio file")
def transcribe_audio_command(
    input_file_path: str,
    output_path: Optional[str] = None,
):
    # Local import of AudioTranscriber
    from chatnerds.tools.audio_transcriber import AudioTranscriber

    try:
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

    except SystemExit:
        raise typer.Abort()


@app.command("transcribe-youtube", help="Get transcript of youtube video")
def transcribe_youtube_command(
    url: UrlFilterArgument = None,
):
    try:
        youtube_loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        transcript_documents = youtube_loader.load()

        if not transcript_documents or len(transcript_documents) == 0:
            logging.error(f"Unable to download transcript of video url: {url}")
            raise typer.Abort()

        rprint(f"\n[bold]Transcript of youtube video {url}:", flush=True)
        rprint(Panel(transcript_documents[0].page_content))

    except SystemExit:
        raise typer.Abort()


@app.command("summarize")
def summarize_command(input_file_path: Optional[str] = None):
    nerd_config = _global_config.get_nerd_config()

    if not input_file_path:
        glob_root_dir = os.path.join(
            _global_config.get_nerd_base_path(), "source_documents"
        )
        source_files_iterator = glob.iglob(
            "**/*.*", root_dir=glob_root_dir, recursive=True
        )

        fzf = FzfPrompt()
        selected_source_files = fzf.prompt(
            source_files_iterator,
            "--cycle --no-multi --keep-right --scheme path --tiebreak end",
        )

        if not selected_source_files or len(selected_source_files) == 0:
            print("No source selected. Exiting...")
            return

        input_file_path = os.path.join(glob_root_dir, selected_source_files[0])

    # Local import of TranscriptLoader
    from chatnerds.document_loaders.transcript_loader import TranscriptLoader

    transcript_documents = TranscriptLoader(input_file_path).load()

    # Local import of Summarizer
    from chatnerds.langchain.summarizer import Summarizer

    llm, prompt_type = LLMFactory(config=nerd_config).get_summarize_model()
    summarizer = Summarizer(
        nerd_config,
        llm=llm,
        prompt_type=prompt_type,
    )

    rprint(f"Summarizing file '{input_file_path}'...")
    summary = summarizer.summarize_text(transcript_documents[0].page_content)

    rprint(
        Panel(
            f"{summary}",
            title="Summary",
            title_align="left",
        )
    )


@app.command("tag")
def tag_command(input_file_path: Optional[str] = None):
    nerd_config = _global_config.get_nerd_config()

    if not input_file_path:
        glob_root_dir = os.path.join(
            _global_config.get_nerd_base_path(), "source_documents"
        )
        source_files_iterator = glob.iglob(
            "**/*.*", root_dir=glob_root_dir, recursive=True
        )

        fzf = FzfPrompt()
        selected_source_files = fzf.prompt(
            source_files_iterator,
            "--cycle --no-multi --keep-right --scheme path --tiebreak end",
        )

        if not selected_source_files or len(selected_source_files) == 0:
            print("No source selected. Exiting...")
            return

        input_file_path = os.path.join(glob_root_dir, selected_source_files[0])

    # Local import of TranscriptLoader
    from chatnerds.document_loaders.transcript_loader import TranscriptLoader

    transcript_documents = TranscriptLoader(input_file_path).load()

    # Local import of Tagger
    from chatnerds.langchain.tagger import Tagger

    llm, prompt_type = LLMFactory(config=nerd_config).get_summarize_model()
    tagger = Tagger(
        nerd_config,
        llm=llm,
        prompt_type=prompt_type,
        prompt=nerd_config["prompts"].get("find_tags_prompt", None),
        n_tags=5,
    )

    rprint(f"Tagging file '{input_file_path}'...")
    tags = tagger.tag_text(transcript_documents[0].page_content)

    if isinstance(tags, str):
        tags = tags.split(",")

    if isinstance(tags, list):
        tags_str = ", ".join(tags)
    else:
        tags_str = str(tags)

    rprint(
        Panel(
            f"{tags_str}",
            title="Tags",
            title_align="left",
        )
    )


@app.command("find-expanded-questions")
def find_expanded_questions_command(
    query: Annotated[
        str,
        typer.Argument(help="The query input to find expanded questions"),
    ],
):
    nerd_config = _global_config.get_nerd_config()

    retrieve_chain_config = nerd_config.get("retrieve_chain", None)
    if isinstance(retrieve_chain_config, str) and retrieve_chain_config in nerd_config:
        retrieve_chain_config = nerd_config.get(retrieve_chain_config, None)

    if not isinstance(retrieve_chain_config, dict):
        raise ValueError(
            f"Invalid value in 'retrieve_chain' configuration: {retrieve_chain_config}"
        )

    question_expansion_chain = ChainFactory(nerd_config).get_question_expansion_chain(
        retrieve_chain_config
    )

    callbacks = []
    if _global_config.VERBOSE > 1:
        callbacks.append(ConsoleCallbackHandler())

    question_expansion_response = question_expansion_chain.invoke(
        query, config={"callbacks": callbacks}
    )

    if isinstance(question_expansion_response, str):
        questions_list = question_expansion_response.split("\n")
    elif isinstance(question_expansion_response, list):
        questions_list = question_expansion_response
    else:
        questions_list = [str(question_expansion_response)]

    questions_str = "\n".join([f"- {question}" for question in questions_list])

    rprint(
        Panel(
            questions_str,
            title="Expanded questions",
            title_align="left",
        )
    )


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

    # Local import of StoreFactory
    from chatnerds.stores.store_factory import StoreFactory

    store_factory = StoreFactory(nerd_config)
    database = store_factory.get_vector_store(embeddings=embeddings)

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
