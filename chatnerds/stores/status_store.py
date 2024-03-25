import logging
from typing import Any, List, Dict, Optional, Iterable
from pathlib import Path
import sqlite3
import json

DEFAULT_DATABASE_FILENAME = "nerd_status.sqlite"


# Reference: https://codereview.stackexchange.com/questions/182700/python-class-to-manage-a-table-in-sqlite
class StatusStore:
    connection = sqlite3.Connection = None
    # cursor: sqlite3.Cursor
    database_path: Path = None
    connect_kwargs: Dict[str, Any] = {}

    def __init__(
        self,
        store_directory_path: str | Path,
        **kwargs: Any,
    ):
        if not Path(store_directory_path).exists():
            raise FileNotFoundError(
                f"Store directory path not found at {store_directory_path}"
            )

        if sqlite3.threadsafety == 3:
            check_same_thread = False
        else:
            check_same_thread = True
            logging.warning(
                "SQLite is not thread safe. This may cause issues when running in a multi-threaded environment."
            )

        database_path = Path(store_directory_path, DEFAULT_DATABASE_FILENAME)
        database_exists = database_path.exists()

        self.database_path = database_path
        self.connect_kwargs = {
            "timeout": 300,
            "check_same_thread": check_same_thread,
            "isolation_level": None,
            **kwargs,
        }

        self.connect()

        # Create the database if it does not exist
        if not database_exists:
            self.migrate_up()

    def add_studied_document(
        self, id: str, source: str, page_content: str, metadata: Dict[str, Any]
    ):
        metadata_json = json.dumps(metadata, indent=4)

        self.execute(
            "INSERT INTO studied_documents (id, source, page_content, metadata) VALUES (?, ?, ?, ?)",
            (
                id,
                source,
                page_content,
                metadata_json,
            ),
        )

    def delete_studied_document(self, id: str):
        self.execute("DELETE FROM studied_documents WHERE id = ?", (id,))

    def iget_studied_document_ids(self) -> Iterable[List[str]]:
        cursor = self.query("SELECT id FROM studied_documents")

        rows = cursor.fetchall()

        for row in rows:
            yield row[0]

    def get_studied_document_ids(self) -> set[str]:
        return set(self.iget_studied_document_ids())

    def iget_studied_documents(self) -> Iterable[List[Dict[str, Any]]]:
        cursor = self.query(
            "SELECT id, source, page_content, metadata FROM studied_documents"
        )

        rows = cursor.fetchall()

        for row in rows:
            yield {
                "id": row[0],
                "source": row[1],
                "page_content": row[2],
                "metadata": json.loads(row[3]),
            }

    def get_studied_documents(self) -> List[Dict[str, Any]]:
        return list(self.iget_studied_documents())

    def get_studied_document(self, id: str) -> Dict[str, Any]:
        cursor = self.query(
            "SELECT id, source, page_content, metadata FROM studied_documents WHERE id = ?",
            (id,),
        )

        row = cursor.fetchone()

        return {
            "id": row[0],
            "source": row[1],
            "page_content": row[2],
            "metadata": json.loads(row[3]),
        }

    def migrate_up(self):
        self.execute(
            "CREATE TABLE IF NOT EXISTS studied_documents (\
                id TEXT PRIMARY KEY, \
                source TEXT, \
                page_content TEXT, \
                metadata TEXT, \
                created_at TEXT, \
                updated_at TEXT)"
        )

    def get_pragma_compile_options(self):
        cursor = self.query("SELECT * FROM pragma_compile_options")
        rows = cursor.fetchall()

        pragma_compile_options = {}
        for row in rows:
            row_parts = str(row[0]).split("=")
            if len(row_parts) > 1:
                pragma_compile_options[row_parts[0]] = (
                    row_parts[1] if len(row_parts) > 1 else None
                )

        return pragma_compile_options

    def connect(self):
        if self.connection is not None and isinstance(
            self.connection, sqlite3.Connection
        ):
            self.connection.close()

        self.connection = sqlite3.connect(
            self.database_path,
            **self.connect_kwargs,
        )

        # if self.cursor:
        #     self.cursor.close()

        # self.cursor = self.connection.cursor()

    def query(self, query: str, params: Optional[List[Any]] = None) -> sqlite3.Cursor:
        self.validate_connection()

        cursor = self.connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        return cursor

    def execute(self, query: str, params: Optional[List[Any]] = None) -> sqlite3.Cursor:
        self.validate_connection()

        try:
            if params:
                cursor = self.connection.execute(query, params)
            else:
                cursor = self.connection.execute(query)

            self.connection.commit()
            return cursor
        except Exception as e:
            self.connection.rollback()
            raise e

    def commit(self):
        self.validate_connection()
        self.connection.commit()

    def validate_connection(self):
        if not self.connection:
            raise Exception(
                "Database connection is not established. Ensure the instance is created in the same thread."
            )

    def close(self):
        # if self.cursor:
        #     try:
        #         self.cursor.close()
        #         self.cursor = None
        #     except:
        #         pass

        if self.connection:
            try:
                self.connection.close()
                self.connection = None
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        # if hasattr(self.__local, 'cursor') and self.__local.cursor:
        #     self.__local.cursor.close()

        if self.connection:
            if isinstance(exc_value, Exception):
                self.connection.rollback()
            else:
                self.connection.commit()

        self.close()

    def __del__(self):
        self.close()
