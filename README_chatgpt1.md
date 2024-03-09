# Chatnerds

## Introduction

Version 1:
Chatnerds is a CLI (Command-Line Interface) application designed to enable Q&A interactions with YouTube videos, podcasts, and documents offline. It leverages local LLMs (Large Language Models) and RAG (Retrieval-Augmented Generation) techniques to provide a comprehensive understanding of various subjects directly from your terminal. With Chatnerds, your documents and conversations remain secure on your system, ensuring privacy and control over your data.

Version 2:
Chatnerds is a unique CLI tool designed for deep dives into your private documents, with the added ability to explore YouTube videos and podcasts. It utilizes cutting-edge local LLMs and RAG techniques for an immersive Q&A experience. Born from a passion for learning about LLM and RAG, this project serves as a personal journey into the heart of data interaction and retrieval technologies.

Version 3:
Introducing Chatnerds: not just another CLI application, but a personal project aimed at revolutionizing the way we interact with private documents, YouTube videos, and podcasts. By harnessing the power of local LLMs and RAG, Chatnerds offers a bespoke Q&A experience, crafted from a desire to explore and understand the intricate workings of LLM and RAG technologies.



## Installation
### Prerequisites
- Python >=3.10

You can install Chatnerds directly using pip or by cloning the repository for a more development-focused setup.

### Using pip
```bash
pip install git+https://github.com/raulonlab/chatnerds.git
```

### Cloning the repository
```bash
Copy code
git clone https://github.com/raulonlab/chatnerds.git
cd chatnerds
poetry install
```

## Usage

Chatnerds allows you to create and manage "nerds," each specializing in different areas of knowledge. These nerds ingest information from various sources including documents, YouTube videos, and podcasts, which they then use to answer your questions.

### Creating and Activating a Nerd

```bash
chatnerds init [nerd_name]
chatnerds activate [nerd_name]
```

### Adding Sources
Place your documents in the appropriate directory and add YouTube video or podcast URLs to the designated sources files.

### Ingesting Sources
```bash
chatnerds download-sources
chatnerds transcribe-downloads
```
chatnerds study
```

### Interacting with Your Nerd
```bash
chatnerds chat
```

Or ask a direct question:

```bash

chatnerds chat "Your question here"
```

## Configuration
Chatnerds offers various configuration options through environment variables or a .env file. These include settings for nerds directory path, logging, transcription options, and more.

## Contributing
We welcome contributions from the community! If you're interested in helping to improve Chatnerds, please see our contributing guidelines for more information.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
