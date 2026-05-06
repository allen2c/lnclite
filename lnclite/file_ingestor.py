"""Directory crawling and text extraction from readable files.

Provides FilesIngestor for scanning trees while skipping binary and hidden paths.
"""

import asyncio
import logging
from pathlib import Path
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    TypeAlias,
    TypedDict,
    cast,
)

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BINARY_PROBE_CHUNK_SIZE: int = 1024

FileReader: TypeAlias = Callable[[Path], str] | Callable[[Path], Awaitable[str]]


class FileIngestorResult(TypedDict):
    path: str
    content: str


class FileIngestor:
    """
    A utility class to crawl directories and extract text content
    from readable files while filtering out binary and hidden data.
    """

    def __init__(self) -> None:
        self._custom_readers: dict[str, FileReader] = {}

    def register_reader(self, extension: str, reader_func: FileReader) -> None:
        """Registers a handler for a specific file extension (e.g., '.pdf')."""
        self._custom_readers[extension.lower()] = reader_func

    def ingest(self, dir_path: str) -> Generator[FileIngestorResult, None, None]:
        """Yield readable file contents from a directory tree."""
        root = Path(dir_path)

        for file_path in root.rglob("*"):
            # Ensure it is a file and not an excluded path
            if not file_path.is_file() or self._is_excluded(file_path):
                continue

            extension = file_path.suffix.lower()

            try:
                # 1. Use specialized reader if registered
                if extension in self._custom_readers:
                    reader = self._custom_readers[extension]
                    if asyncio.iscoroutinefunction(reader):
                        logger.warning(
                            "Skipping %s: async reader requires ingest_async.",
                            file_path,
                        )
                        continue
                    sync_reader = cast(Callable[[Path], str], reader)
                    content = sync_reader(file_path)
                    yield FileIngestorResult(path=str(file_path), content=content)

                # 2. Fallback to binary probe for generic text files
                elif not self._is_binary(file_path):
                    content = self._read_text(file_path)
                    yield FileIngestorResult(path=str(file_path), content=content)

            except Exception as e:
                logger.warning("Skipping %s due to error: %s", file_path, e)

    async def ingest_async(
        self, dir_path: str
    ) -> AsyncGenerator[FileIngestorResult, None]:
        """Async variant of ingest: walks the tree sync, reads in worker threads."""
        root = Path(dir_path)

        for file_path in root.rglob("*"):
            if not file_path.is_file() or self._is_excluded(file_path):
                continue

            extension = file_path.suffix.lower()

            try:
                if extension in self._custom_readers:
                    reader = self._custom_readers[extension]
                    is_coro_reader = asyncio.iscoroutinefunction(reader)
                    if is_coro_reader:
                        content = await reader(file_path)
                    else:
                        content = reader(file_path)
                    yield FileIngestorResult(path=str(file_path), content=content)
                elif not await asyncio.to_thread(self._is_binary, file_path):
                    content = await asyncio.to_thread(self._read_text, file_path)
                    yield FileIngestorResult(path=str(file_path), content=content)
            except Exception as e:
                logger.warning("Skipping %s due to error: %s", file_path, e)

    def _is_binary(
        self, file_path: Path, chunk_size: int = DEFAULT_BINARY_PROBE_CHUNK_SIZE
    ) -> bool:
        """Detect binary files with null-byte and UTF-8 probes."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(chunk_size)
                # Null bytes are standard in binary formats
                if b"\0" in chunk:
                    return True
                # Attempt decoding to verify it's a valid text format
                chunk.decode("utf-8")
                return False
        except (UnicodeDecodeError, Exception):
            return True

    def _is_excluded(self, path: Path) -> bool:
        """Checks if any part of the file path starts with '.' or '_'."""
        return any(part.startswith((".", "_")) for part in path.parts)

    def _read_text(self, file_path: Path) -> str:
        """Reads plain text files with encoding safety."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
