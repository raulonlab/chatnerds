# Resources:
# Audio transcriber: https://github.com/KostasEreksonas/Audio-transcriber/blob/main/transcriber.py
# Translate transcribed text. Credit to Harsh Jain at educative.io: https://www.educative.io/answers/how-do-you-translate-text-using-python

from typing import List, Union, Optional, Dict
import os
import glob
import logging
import torch
import whisper
import music_tag
import json
from .config import Config
from .llms.summarizer import Summarizer

class AudioTranscriber:
    source_directory: str
    config: Config = Optional[Config]
    model: any = None
    summarizer: Summarizer = None

    def __init__(self, source_directory: str = ".", config: Config = None):
        if config:
            self.config = config
        else:
            self.config = Config.environment_instance()

        self.source_directory = source_directory
    
    def run(self, filter_directory:str = "", output_path: str = None) -> List[str]:
        audio_files = self.find_audio_files(self.source_directory)
        
        logging.info(f"Running transcriber with {len(audio_files)} audio files")

        transcript_file_paths = []
        for audio_file in audio_files:
            if filter_directory and filter_directory not in audio_file:
                continue

            # Get filename without ext
            input_audio_file_without_ext, ext = os.path.splitext(os.path.basename(audio_file))

            # Get transcript file path
            if output_path:
                transcript_file_path = os.path.join(output_path, f"{input_audio_file_without_ext}.transcript")
            else:
                transcript_file_path = os.path.join(os.path.dirname(audio_file), f"{input_audio_file_without_ext}.transcript")

            if os.path.exists(transcript_file_path):
                transcript_file_paths.append(transcript_file_path)
                continue
                
            logging.debug(f"  ...transcribing: {audio_file}")
            transcript = self.transcribe_audio(
                input_audio_file_path = str(audio_file),
                model_name = self.config.WHISPER_TRANSCRIPTION_MODEL_NAME,
            )

            audio_file_tags = self.read_audio_file_tags(str(audio_file))

            if (self.config.TRANSCRIPT_ADD_SUMMARY):
                if (self.summarizer is None):
                    self.summarizer = Summarizer(config=self.config.get_nerd_config())
                summary = self.summarizer.summarize_text(transcript)

            with open(transcript_file_path, "w", encoding="utf-8") as file:
                file.write(f"transcript=\"{self.escape_dotenv_value(transcript)}\"\n")
                if (summary is not None):
                    file.write(f"summary=\"{self.escape_dotenv_value(summary)}\"\n")
                for tag_key, tag_value in audio_file_tags.items():
                    file.write(f"{tag_key}=\"{self.escape_dotenv_value(tag_value)}\"\n")

            transcript_file_paths.append(transcript_file_path)

        logging.info("\nTranscript finished....")
        return transcript_file_paths


    def transcribe_audio(self, input_audio_file_path: str, model_name: str = "tiny"):
        """Transcribe audio file."""

        """Get speech recognition model."""
        logging.info(f"Transcribing audio '{input_audio_file_path}' ...")
        if self.model is None:
            logging.info(f"Loading model '{model_name}' with device '{self.check_device()}' ...")
            self.model = whisper.load_model(model_name, device=self.check_device())
            logging.info("Model loaded successfully")
        result = self.model.transcribe(input_audio_file_path)

        """Put a newline character after each sentence in the transcript."""
        formatted_text = result["text"].replace(". ", ".\n")

        return formatted_text


    @staticmethod
    def find_audio_files(directory_path: str):
        audio_files = []
        for filename in glob.iglob(os.path.join(directory_path, "./**/*.mp3"), recursive=True):
            audio_files.append(os.path.abspath(filename))
        
        return audio_files


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
        return text.strip(" \n\"").replace("\"", "'")


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

