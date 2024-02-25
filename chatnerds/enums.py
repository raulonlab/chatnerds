from enum import Enum


class LogColors:
    BOLD = "\033[1m"
    DEBUG = "\033[2m"
    NERD = "\033[92m"
    SOURCE = "\033[96m"
    BOLDNERD = "\033[1m\033[92m"
    BOLDSOURCE = "\033[1m\033[96m"

    # Other colors
    OK = "\033[92m"
    WARNING = "\033[95m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


class Emojis:
    NERD = "🧠"
    SOURCE = "📚"
    CHAT = "💬"
    OK = "✔"
    FAIL = "✘"


class SourceEnum(str, Enum):
    books = "books"
    youtube = "youtube"
    podcasts = "podcasts"


SOURCE_PATHS = {
    SourceEnum.books: "source_documents",
    SourceEnum.youtube: "downloads/youtube",
    SourceEnum.podcasts: "downloads/podcasts",
}
