from __future__ import annotations

import json
import re

from semble.types import Chunk, GraphContext, SearchResult

_GIT_URL_SCHEMES = ("https://", "http://", "ssh://", "git://", "git+ssh://", "file://")
_SCP_GIT_URL_RE = re.compile(r"^[\w.-]+@[\w.-]+:(?!/)")


def _is_git_url(path: str) -> bool:
    """Return True if path looks like a remote git URL rather than a local path."""
    return path.startswith(_GIT_URL_SCHEMES) or _SCP_GIT_URL_RE.match(path) is not None


def _resolve_chunk(
    chunks: list[Chunk],
    file_path: str,
    line: int,
    *,
    file_mapping: dict[str, list[int]] | None = None,
) -> Chunk | None:
    """Return the chunk containing *line* in *file_path*, or None.

    When *file_mapping* (file → chunk indices) is provided, only chunks
    from the matching file are scanned instead of the entire list.
    """
    if file_mapping and file_path in file_mapping:
        candidates = [chunks[i] for i in file_mapping[file_path]]
    else:
        candidates = chunks

    fallback = None
    for chunk in candidates:
        if chunk.file_path == file_path and chunk.start_line <= line <= chunk.end_line:
            if line < chunk.end_line:
                return chunk
            if fallback is None:
                fallback = chunk
    return fallback


def _format_results(header: str, results: list[SearchResult]) -> str:
    """Render SearchResult objects as numbered, fenced code blocks."""
    lines: list[str] = [header, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r.chunk.location}  [score={r.score:.3f}]")
        lines.append("```")
        lines.append(r.chunk.content.strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _format_results_json(
    results: list[SearchResult],
    contexts: dict[str, GraphContext] | None = None,
    *,
    compact: bool = False,
) -> str:
    """Render SearchResult objects as JSON with relational metadata.

    When *compact* is True, the ``code`` field is omitted to save tokens.
    """
    contexts = contexts or {}
    output: list[dict] = []
    for r in results:
        ctx = contexts.get(r.chunk.location, GraphContext())
        entry: dict = {
            "file": r.chunk.file_path,
            "line": f"{r.chunk.start_line}-{r.chunk.end_line}",
            "file_total_lines": r.chunk.file_total_lines,
            "score": round(r.score, 4),
            "source": r.source.value,
            "context": {
                "called_by": ctx.called_by,
                "depends_on": ctx.depends_on,
            },
        }
        if not compact:
            entry["code"] = r.chunk.content
        output.append(entry)
    return json.dumps(output, ensure_ascii=False)


