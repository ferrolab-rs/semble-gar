import logging
from pathlib import Path

from chonkie.chunker import CodeChunker

from semble.index.file_walker import language_for_path
from semble.types import Chunk

logger = logging.getLogger(__name__)


def chunk_file(file_path: Path) -> list[Chunk]:
    """Chunk a single file from disk."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    return chunk_source(source, str(file_path), language_for_path(file_path))


def chunk_source(source: str, file_path: str, language: str | None) -> list[Chunk]:
    """Chunk pre-read source text."""
    if not source.strip():
        return []
    total_lines = source.count("\n") + 1
    if language:
        return _chunk_with_chonkie(source, file_path, language, total_lines)
    return chunk_lines(source, file_path, language, total_lines=total_lines)


def chunk_lines(
    source: str,
    file_path: str,
    language: str | None = None,
    max_lines: int = 50,
    overlap_lines: int = 5,
    total_lines: int = 0,
) -> list[Chunk]:
    """Split source by line count with overlap."""
    lines = source.splitlines(keepends=True)
    if not lines:
        return []

    tl = total_lines or len(lines)
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        content = "".join(lines[start:end])
        if content.strip():
            chunks.append(
                Chunk(
                    content=content,
                    file_path=file_path,
                    start_line=start + 1,
                    end_line=end,
                    language=language,
                    file_total_lines=tl,
                )
            )
        start = end - overlap_lines if end < len(lines) else end

    return chunks


def _chunk_with_chonkie(source: str, file_path: str, language: str, total_lines: int = 0) -> list[Chunk]:
    """Chunk source with Chonkie and fall back to line chunks on failure."""
    tl = total_lines or source.count("\n") + 1
    try:
        code_chunker = CodeChunker(language=language, chunk_size=1500)
        raw_chunks = code_chunker.chunk(source)
    except Exception:
        logger.debug("Chonkie failed for language %r, falling back to line chunking", language, exc_info=True)
        return chunk_lines(source, file_path, language, total_lines=tl)

    if not raw_chunks:
        return chunk_lines(source, file_path, language, total_lines=tl)

    chunks: list[Chunk] = []
    for raw_chunk in raw_chunks:
        text = raw_chunk.text
        if not text.strip():
            continue
        end_index = max(raw_chunk.end_index - 1, raw_chunk.start_index)
        chunks.append(
            Chunk(
                content=text,
                file_path=file_path,
                start_line=source[: raw_chunk.start_index].count("\n") + 1,
                end_line=source[:end_index].count("\n") + 1,
                language=language,
                file_total_lines=tl,
            )
        )
    return chunks if chunks else chunk_lines(source, file_path, language, total_lines=tl)
