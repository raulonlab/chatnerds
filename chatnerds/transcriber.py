# Resources:
# Audio transcriber: https://github.com/KostasEreksonas/Audio-transcriber/blob/main/transcriber.py
# Translate transcribed text. Credit to Harsh Jain at educative.io: https://www.educative.io/answers/how-do-you-translate-text-using-python

from typing import List, Union, Optional
import os
import glob
import logging
import torch
import whisper
from .config import Config

class AudioTranscriber:
    source_directory: str
    config: Config = Optional[Config]

    def __init__(self, source_directory: str = ".", config: Config = None):
        if config:
            self.config = config
        else:
            self.config = Config.environment_instance()

        self.source_directory = source_directory
    
    def run(self, filter_directory:str = "", output_path = "."):
        audio_files = self.find_audio_files(self.source_directory)
        
        logging.info(f"Running transcriber with {len(audio_files)} audio files")

        for audio_file in audio_files:
            if filter_directory and filter_directory not in audio_file:
                continue

            logging.debug(f"  ...transcribing: {audio_file}")
            transcript_file_path = self.transcribe_audio(
                input_audio_file_path = str(audio_file),
                model_name = self.config.WHISPER_TRANSCRIPTION_MODEL_NAME,
                # output_path = str(Path(self.config.get_nerd_base_path(), "downloads")),
            )
            logging.debug(f"  ...done! Output: '{transcript_file_path}'")

        logging.info("\nyoutube downloader finished....")


    def transcribe_audio(self, input_audio_file_path: str, model_name: str = "tiny", output_path: Union[str, None] = None):
        """Transcribe audio file."""

        # Get filename without ext
        input_audio_file_without_ext, ext = os.path.splitext(os.path.basename(input_audio_file_path))

        # Fix output_path: If None, use input_audio_file_path's directory
        if not output_path:
            output_path = os.path.dirname(input_audio_file_path)

        # If transcription file (.txt) exists, return it
        transcript_output_file_path = os.path.join(output_path, f"{input_audio_file_without_ext}.txt")
        if os.path.exists(transcript_output_file_path):
            return transcript_output_file_path
        
        """Get speech recognition model."""
        # model_name = input("Select speech recognition model name (tiny, base, small, medium, large): ")
        logging.debug("Transcribing audio...")
        model = whisper.load_model(model_name, device=self.check_device())
        result = model.transcribe(input_audio_file_path)

        """Put a newline character after each sentence in the transcript."""
        formatted_text = result["text"].replace(". ", ".\n")

        with open(transcript_output_file_path, "w", encoding="utf-8") as file:
            file.write(formatted_text)
            logging.debug(f"Transcript completed successfully in '{transcript_output_file_path}'")
        
        return transcript_output_file_path

    @staticmethod
    def find_audio_files(directory_path: str):
        audio_files = []
        for filename in glob.iglob(os.path.join(directory_path, "./**/*.mp3"), recursive=True):
            audio_files.append(os.path.abspath(filename))
        
        return audio_files

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

