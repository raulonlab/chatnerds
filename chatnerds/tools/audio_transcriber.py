# Resources:
# Audio transcriber: https://github.com/KostasEreksonas/Audio-transcriber/blob/main/transcriber.py
# Translate transcribed text. Credit to Harsh Jain at educative.io: https://www.educative.io/answers/how-do-you-translate-text-using-python

from typing import List, Optional, Dict, Tuple, Any
import os
from time import sleep
import glob
import logging
import torch
import whisper
import music_tag
from concurrent.futures import ProcessPoolExecutor, as_completed
from chatnerds.config import Config
from chatnerds.lib.helpers import check_for_package
from chatnerds.tools.event_emitter import EventEmitter

try:
    from whisper_mps import whisper as whispermps
except ImportError:
    pass


_RUN_TASKS_LIMIT = 1_000  # Maximum number of tasks to run in a single call to run()


class AudioTranscriber(EventEmitter):
    config: Config = Optional[Config]
    use_whipser_mps: bool = False

    def __init__(self, config: Config = None):
        super().__init__()

        if config:
            self.config = config
        else:
            self.config = Config.environment_instance()

        if not self.config.WHISPER_TRANSCRIPTION_MODEL_NAME:
            raise ValueError(
                "No transcription model name provided in 'config.WHISPER_TRANSCRIPTION_MODEL_NAME'"
            )

        if self.check_device() == "mps" and check_for_package("whisper_mps"):
            self.use_whipser_mps = True

    def run(
        self,
        source_directory: str = ".",
        source_files: List[str] = None,
        output_path: str = None,
        include_filter: str = None,
        force: bool = False,
        dry_run: bool = False,
        limit: int = _RUN_TASKS_LIMIT,
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

            if not force and os.path.exists(transcript_file_path):
                continue

            transcribe_audio_arguments.append(
                {
                    "input_audio_file_path": str(audio_file),
                    "transcript_file_path": transcript_file_path,
                    "model_name": self.config.WHISPER_TRANSCRIPTION_MODEL_NAME,
                    "use_whipser_mps": self.use_whipser_mps,
                }
            )

        # Return dry run result
        if dry_run == True:
            return [
                argument["transcript_file_path"]
                for argument in transcribe_audio_arguments
            ], []

        # Return if no audio files to transcribe
        if len(transcribe_audio_arguments) == 0:
            logging.debug("No audio files to transcribe")
            return [], []

        # Limit number of tasks to run
        if not limit or not 0 < limit < _RUN_TASKS_LIMIT:
            limit = _RUN_TASKS_LIMIT
        if len(transcribe_audio_arguments) > limit:
            logging.warning(
                f"Number of audios to transcribe cut to limit {limit} (out of {len(transcribe_audio_arguments)})"
            )
            transcribe_audio_arguments = transcribe_audio_arguments[:limit]

        logging.debug(
            f"Start transcribing {len(transcribe_audio_arguments)} audio files with model '{self.config.WHISPER_TRANSCRIPTION_MODEL_NAME}'..."
        )
        if self.use_whipser_mps:
            logging.debug("Using package 'whisper_mps' with device type 'MPS'...")
        else:
            logging.debug(
                "Using package 'whisper' with device type 'CPU / GPU (No MPS)'..."
            )

        # Emit start event (show progress bar in UI)
        self.emit("start", len(transcribe_audio_arguments))

        results = []
        errors = []
        # max_workers = 2 if os.cpu_count() > 4 else 1  # Maximum 2 workers (2 models loaded in memory at a time)
        max_workers = 1
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

                    try:
                        response_parts = str(response).split("/")
                        if len(response_parts) > 1:
                            self.emit(
                                "write",
                                f"✔ .../{response_parts[-2]}/{response_parts[-1]}",
                            )
                        else:
                            self.emit("write", "✔ ...")
                    except:
                        pass
                except Exception as err:
                    errors.append(err)
                    continue
                finally:
                    self.emit("update")

        self.emit("end")
        return results, errors

    @staticmethod
    def transcribe_audio(input_arguments: Dict[str, Any]):
        input_audio_file_path = input_arguments["input_audio_file_path"]
        transcript_file_path = input_arguments["transcript_file_path"]
        model_name = input_arguments["model_name"]
        use_whipser_mps: bool = input_arguments.get("use_whipser_mps", False)

        if use_whipser_mps:
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

        # Unblock system resources
        sleep(0.2)

        return transcript_file_path

    @staticmethod
    def transcribe_audio_using_cpu(input_audio_file_path: str, model_name: str):
        """Transcribe audio file."""

        """Load speech recognition model."""

        # Whisper doesn't support MPS devices
        device = AudioTranscriber.check_device()
        if device != "cuda" and device != "cpu":
            device = "cpu"
        model = whisper.load_model(model_name, device=device)

        result = model.transcribe(
            input_audio_file_path,
        )

        # Free memory?
        model = None

        # Put a newline character after each sentence in the transcript.
        # formatted_text = result["text"].replace(". ", ".\n")

        return result.get("text", "")

    @staticmethod
    def transcribe_audio_using_mps(input_audio_file_path: str, model_name: str):
        """Transcribe audio file."""

        """Get speech recognition model."""

        result = whispermps.transcribe(input_audio_file_path, model=model_name)

        # Put a newline character after each sentence in the transcript.
        # formatted_text = result["text"].replace(". ", ".\n")

        return result.get("text", "")

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
    def escape_dotenv_value(text: str) -> str:
        """Strip characters and replace double quotes with single quotes."""
        return text.strip(' \n"').replace('"', "'")

    @staticmethod
    def check_device():
        """Check CUDA availability."""
        if torch.cuda.is_available() == 1:
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
