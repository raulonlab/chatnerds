# [ChatNerds](https://github.com/raulonlab/chatnerds)

CLI application to Q&A with YouTube videos, podcasts and documents offline using local LLMs and RAG (Retrieval-augmented generation) techniques. Your documents and conversations don't leaves your system.

## Installation

Install package (requires python >=3.10)
```shell
pip install git+https://github.com/raulonlab/chatnerds.git
```

Or creating an environment
```shell
pyenv shell 3.10
pipenv shell
pipenv install git+https://github.com/raulonlab/chatnerds.git
```

## Usage

ChatNerds works with a collection of sources called "nerd" allowing to chat separately in certain areas of knowledge. You can create a new nerd, add sources of an specific subject to it and chat with it.

For this example, we're going to create a new nerd called "stoicism" and work with it:

#### Create and manage nerds

Create a new nerd called "stoicism"
```shell
chatnerds nerd create stoicism
```
> This creates a new directory with the initial files in `./nerds/stoicism`

Activate it:
```shell
chatnerds nerd activate stoicism
```
> This writes the environment variable `ACTIVE_NERD=stoicism` in the file `.chatnerds.env` which will be loaded by the tool in the next commands.

There are other commands to manage the nerds. See `chatnerds nerd --help` for more information.

#### Add sources

Each nerd is able to process 3 types of sources: documents (pdf, docx, etc), youtube (playlist and video urls) and podcasts (xml feed urls). Add the sources for each type:

- **documents**: Copy your document files into the directory `./nerds/stoicism/source_documents`. The current supported formats with the loader class (from Langchain) used are the following:
	- `".pdf"`: PDFMinerLoader,
	- `".epub"`: UnstructuredEPubLoader,
	- `".md"`: UnstructuredMarkdownLoader,
	- `".txt"`: TextLoader,
	- `".doc"`: UnstructuredWordDocumentLoader,
	- `".docx"`: UnstructuredWordDocumentLoader,
	- `".enex"`: EverNoteLoader,
	- `".csv"`: CSVLoader,
	- `".html"`: UnstructuredHTMLLoader,
	- `".odt"`: UnstructuredODTLoader,
	- `".ppt"`: UnstructuredPowerPointLoader,
	- `".pptx"`: UnstructuredPowerPointLoader,

- **youtube**: Open `./nerds/stoicism/youtube.sources` and add the urls of the videos and playlists you want to add. One url per line. Lines starting with `# ` are ignored.
- **podcasts**: Open `./nerds/stoicism/podcast.sources` and add the urls of the podcasts you want to add. One url per line. Lines starting with `#` are ignored.

There are some free resources about "stoicism" in [docs/stoicism_resources.md](docs/stoicism_resources.md).

#### Download audio from youtube and podcasts sources

If you added youtube or podcast sources, start downloading the audios:
```shell
chatnerds download
```
> The downloads are saved in `./nerds/stoicism/downloads/...`

#### Transcribe audios to text

If you downloaded youtube or podcast sources, start transcribing the audios into text files:
```shell
chatnerds transcribe
```

#### Add sources to embeddings DB

Once you have documents in `source_documents` or `downloads/...`, add them to the embeddings DB:
```shell
chatnerds add
```
> The first time, it creates a new directory for the DB in `./nerds/stoicism/db/` and downloads the AI models (It takes a while)

#### Chat with the nerd

Start chatting with your nerd about stoicism:
```shell
chatnerds chat
```
> The first time, it needs to download the AI models and takes a while

#### Nerd model settings in config.yml

The file `./nerds/stoicism/config.yml` allows to override the default model settings used by the nerd. See the default configuration (with annotations) in [chatnerds/config.yml](chatnerds/config.yml). The new configuration is applied on every command. 

> Note that changing some of the configuration, like `embeddings` and `chroma`, will invalidate the current embeddings added in the database. To fix it, delete the directory `./nerds/stoicism/db` and start again

## Configuration with environment variables

The general behaviour of the application can be configured with environment variables or reading them from a `.env` file (optional). The available variables with their default values are:

```shell
# general options
NERDS_DIRECTORY_PATH=nerds  # (Default: "nerds") Path to nerds directory
LOG_FILE_LEVEL=NOTSET  # (Default: NOTSET) Logging level for the log file. Values: INFO, WARNING, ERROR, CRITICAL, NOTSET. If NOTSET, disable logging to file
LOG_FILE_PATH=logs/chatnerds.log  # (Default: "logs/chatnerds.log") Path to log file
VERBOSE=1  # (Default: 1) Amount of logs written to stdout (0: none, 1: medium, 2: full)

# transcription (Whisper) options
WHISPER_TRANSCRIPTION_MODEL_NAME=base  # (Default: "base") Name of the model to use for transcribing audios: tiny, base, small, medium, large
TRANSCRIPT_ADD_SUMMARY=False # (Default: False) Include a summary of the transcription in the output file

# youtube donwload options
YOUTUBE_GROUP_BY_AUTHOR=True  # (Default: True) Group downloaded videos by channel
YOUTUBE_SLEEP_SECONDS_BETWEEN_DOWNLOADS=3  # (Default: 3) Number of seconds to sleep between downloads
YOUTUBE_ADD_DATE_PREFIX=True  # (Default: True) Prefix all episodes with an ISO8602 formatted date of when they were published. Useful to ensure chronological ordering
YOUTUBE_SLUGIFY_PATHS=True  # (Default: True) Clean all folders and filename of potentially weird characters that might cause trouble with one or another target filesystem

# podcast donwload options
PODCAST_UPDATE_ARCHIVE=True  # (Default: True) Force the archiver to only update the feeds with newly added episodes. As soon as the first old episode found in the download directory, further downloading is interrupted
PODCAST_ADD_DATE_PREFIX=True  # (Default: True) Prefix all episodes with an ISO8602 formatted date of when they were published. Useful to ensure chronological ordering
PODCAST_SLUGIFY_PATHS=True  # (Default: True) Clean all folders and filename of potentially weird characters that might cause trouble with one or another target filesystem
PODCAST_GROUP_BY_AUTHOR=True  # (Default: True) Create a subdirectory for each feed (named with their titles) and put the episodes in there
PODCAST_MAXIMUM_EPISODE_COUNT=10  # (Default: 10) Only download the given number of episodes per podcast feed. Useful if you don't really need the entire backlog. Set 0 to disable limit
PODCAST_SHOW_PROGRESS_BAR=True  # (Default: True) Show a progress bar while downloading
```

The application also saves certain runtime settings in the file `.chatnerds.env` (for example, the active nerd). It's not necessary to touch this file since is handled automatically by the application.

## License

[MIT](LICENSE)
