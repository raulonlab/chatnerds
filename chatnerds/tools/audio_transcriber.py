# Resources:
# Audio transcriber: https://github.com/KostasEreksonas/Audio-transcriber/blob/main/transcriber.py
# Translate transcribed text. Credit to Harsh Jain at educative.io: https://www.educative.io/answers/how-do-you-translate-text-using-python

from typing import List, Optional, Dict, Tuple
import os
import glob
import logging
import torch
import whisper
import music_tag
from concurrent.futures import ProcessPoolExecutor, as_completed
from chatnerds.config import Config
from chatnerds.utils import check_for_package
from chatnerds.langchain.summarizer import Summarizer
from chatnerds.tools.event_emitter import EventEmitter

try:
    from whisper_mps import whisper as whispermps
except ImportError:
    pass


_RUN_TASKS_LIMIT = 1_000  # Maximum number of tasks to run in a single call to run()


class AudioTranscriber(EventEmitter):
    config: Config = Optional[Config]
    summarizer: Optional[Summarizer] = None

    def __init__(self, config: Config = None):
        super().__init__()

        if config:
            self.config = config
        else:
            self.config = Config.environment_instance()

    def run(
        self,
        source_directory: str = ".",
        source_files: List[str] = None,
        output_path: str = None,
        include_filter: str = None,
    ) -> Tuple[List[str], List[any]]:
        logging.debug("Running audio transcriber...")

        if source_files and len(source_files) > 0:
            audio_files = source_files
        elif source_directory:
            logging.debug("Finding audio files in source directory...")
            audio_files = self.find_audio_files(source_directory)
        else:
            raise ValueError("No audio files found")

        transcribe_audio_arguments = []
        for audio_file in audio_files:
            if include_filter and include_filter not in audio_file:
                continue

            # Get filename without ext
            input_audio_file_without_ext, ext = os.path.splitext(
                os.path.basename(audio_file)
            )

            # Get transcript file path
            if output_path:
                transcript_file_path = os.path.join(
                    output_path, f"{input_audio_file_without_ext}.transcript"
                )
            else:
                transcript_file_path = os.path.join(
                    os.path.dirname(audio_file),
                    f"{input_audio_file_without_ext}.transcript",
                )

            if os.path.exists(transcript_file_path):
                continue

            transcribe_audio_arguments.append(
                {
                    "input_audio_file_path": str(audio_file),
                    "transcript_file_path": transcript_file_path,
                    "model_name": self.config.WHISPER_TRANSCRIPTION_MODEL_NAME,
                }
            )

        # Return if no audio files to transcribe
        if len(transcribe_audio_arguments) == 0:
            logging.debug("No audio files to transcribe")
            return [], []

        # Limit number of tasks to run
        if len(transcribe_audio_arguments) > _RUN_TASKS_LIMIT:
            transcribe_audio_arguments = transcribe_audio_arguments[:_RUN_TASKS_LIMIT]
            logging.warning(
                f"Number of audios to transcribe exceeds limit of {_RUN_TASKS_LIMIT}. Only processing first {_RUN_TASKS_LIMIT} videos."
            )

        logging.debug(
            f"Start processing {len(transcribe_audio_arguments)} audio files..."
        )

        # Emit start event (show progress bar in UI)
        self.emit("start", len(transcribe_audio_arguments))

        results = []
        errors = []
        max_workers = max(1, os.cpu_count() // 3)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            response_futures = [
                executor.submit(AudioTranscriber.transcribe_audio, transcribe_argument)
                for transcribe_argument in transcribe_audio_arguments
            ]

            for response_future in as_completed(response_futures):
                try:
                    response = response_future.result()
                    if response:
                        results.append(response)
                except Exception as err:
                    errors.append(err)
                    continue
                finally:
                    self.emit("update")

        self.emit("end")
        return results, errors

    @staticmethod
    def transcribe_audio(input_arguments):
        input_audio_file_path = input_arguments["input_audio_file_path"]
        transcript_file_path = input_arguments["transcript_file_path"]
        model_name = input_arguments["model_name"]

        if check_for_package("whisper_mps"):
            transcript = AudioTranscriber.transcribe_audio_using_mps(
                input_audio_file_path=input_audio_file_path, model_name=model_name
            )
        else:
            transcript = AudioTranscriber.transcribe_audio_using_cpu(
                input_audio_file_path=input_audio_file_path, model_name=model_name
            )

        audio_file_tags = AudioTranscriber.read_audio_file_tags(
            str(input_audio_file_path)
        )

        with open(transcript_file_path, "w", encoding="utf-8") as file:
            file.write(
                f'transcript="{AudioTranscriber.escape_dotenv_value(transcript)}"\n'
            )
            for tag_key, tag_value in audio_file_tags.items():
                file.write(
                    f'{tag_key}="{AudioTranscriber.escape_dotenv_value(tag_value)}"\n'
                )

        return transcript_file_path

    @staticmethod
    def transcribe_audio_using_cpu(input_audio_file_path: str, model_name: str):
        """Transcribe audio file."""

        """Load speech recognition model."""
        logging.debug(f"Loading model '{model_name}'...")
        model = whisper.load_model(model_name, device=AudioTranscriber.check_device())

        logging.debug(
            f"Transcribing audio using CPU / GPU (No MPS): '{input_audio_file_path}' ..."
        )
        result = model.transcribe(input_audio_file_path)

        # Free memory?
        model = None

        """Put a newline character after each sentence in the transcript."""
        formatted_text = result["text"].replace(". ", ".\n")

        return formatted_text

    @staticmethod
    def transcribe_audio_using_mps(input_audio_file_path: str, model_name: str):
        """Transcribe audio file."""

        """Get speech recognition model."""
        logging.debug(f"Transcribing audio using MPS: '{input_audio_file_path}' ...")
        result = whispermps.transcribe(input_audio_file_path, model=model_name)

        """Put a newline character after each sentence in the transcript."""
        formatted_text = result["text"].replace(". ", ".\n")

        return formatted_text

    @staticmethod
    def find_audio_files(directory_path: str):
        pathname = os.path.join(directory_path, "./**/*.mp3")
        sorted_files = sorted(glob.iglob(pathname, recursive=True), reverse=True)

        return [os.path.abspath(file) for file in sorted_files]

    @staticmethod
    def read_audio_file_tags(filepath: str) -> Dict[str, str]:
        music_tag_file = music_tag.load_file(filepath)
        return {
            "title": str(music_tag_file["title"]),
            "album": str(music_tag_file["album"]),
            "artist": str(music_tag_file["artist"]),
            "comment": str(music_tag_file["comment"]),
        }

    @staticmethod
    def escape_dotenv_value(text: str) -> Dict[str, str]:
        """Strip characters and replace double quotes with single quotes."""
        return text.strip(' \n"').replace('"', "'")

    @staticmethod
    def check_device():
        """Check CUDA availability."""
        if torch.cuda.is_available() == 1:
            logging.debug("Using CUDA")
            device = "cuda"
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     logging.debug("Using MPS")
        #     device = "mps"
        else:
            logging.debug("Using CPU")
            device = "cpu"
        return device
