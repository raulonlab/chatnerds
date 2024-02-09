import logging
from typing import Dict
import typer
from .document_loaders.transcript_loader import TranscriptLoader
from .document_embeddings import DocumentEmbeddings
from .llms.summarizer import Summarizer
from .config import Config

global_config = Config.environment_instance()

app = typer.Typer()

@app.command("split")
def split(input_file_path: str):
    nerd_config = global_config.get_nerd_config()
    try:
        transcript_documents = TranscriptLoader(input_file_path).load()
        chunks = DocumentEmbeddings(config=nerd_config).split_documents(transcript_documents)

        print(f"Split {input_file_path}:")
        print("#########################")
        for chunk in chunks:
            print(chunk.page_content)
            print("-----------------------------")

    except Exception as e:
        logging.error(f"Error running test split with input file {input_file_path}:\n{e}")
        raise typer.Abort()


@app.command("summarize")
def split(input_file_path: str):
    nerd_config = global_config.get_nerd_config()
    try:
        transcript_documents = TranscriptLoader(input_file_path).load()

        summarizer = Summarizer(nerd_config)
        summary = summarizer.summarize_text(transcript_documents[0].page_content)

        print(f"Summarize {input_file_path}:")
        print("#########################")
        print(summary)

    except Exception as e:
        logging.error(f"Error running test summarize with input file {input_file_path}:\n{e}")
        raise typer.Abort()



