from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from semble.index import SembleIndex
from semble.index.dense import load_model
from semble.types import Encoder
from semble.utils import _format_results, _format_results_json, _is_git_url, _resolve_chunk

_REPO_DESCRIPTION = (
    "Git URL (e.g. https://github.com/org/repo) or local path to index and search. "
    "Required when no default index was configured at startup. "
    "The index is cached after the first call, so repeat queries are fast."
)


def create_server(cache: _IndexCache, default_source: str | None = None) -> FastMCP:
    """Build and return a configured FastMCP server backed by the given cache."""
    server = FastMCP(
        "semble",
        instructions=(
            "Instant code search with graph-augmented retrieval for any local or GitHub repository. "
            "Use `search` to find code by intent, `trace_symbol` to navigate the call graph (callers/callees), "
            "`get_impact_radius` for recursive blast-radius analysis (all callers + subclasses up to N levels), "
            "`explore_graph` for a chunk's relational context, and `find_related` for semantically similar code. "
            "For external libraries, resolve the GitHub URL from your training knowledge and pass it as `repo`. "
            "Prefer these tools over Grep, Glob, or Read for any question about how code works."
        ),
    )

    @server.tool()
    async def search(
        query: Annotated[str, Field(description="Natural language or code query.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        mode: Annotated[
            Literal["hybrid", "semantic", "bm25"],
            Field(description="Search mode. 'hybrid' is best for most queries."),
        ] = "hybrid",
        top_k: Annotated[int, Field(description="Number of results to return.", ge=1)] = 5,
        filter_languages: Annotated[
            list[str] | None,
            Field(description="Restrict results to these language codes (e.g. ['python', 'rust'])."),
        ] = None,
        filter_paths: Annotated[
            list[str] | None,
            Field(description="Restrict results to these repo-relative file paths."),
        ] = None,
        compact: Annotated[
            bool,
            Field(description="If true, omit code content to save tokens. Use for broad exploration."),
        ] = False,
    ) -> str:
        """Search a codebase with a natural-language or code query.

        Pass a git URL or local path as `repo` to index it on demand; indexes are cached for the session.
        Use this to find where something is implemented, understand a library, or locate related code.
        """
        source = repo or default_source
        if not source:
            return (
                "No repo specified and no default index. "
                "Pass a git URL (https://github.com/...) or local path as `repo`."
            )
        try:
            index = await cache.get(source)
        except Exception as exc:
            return f"Failed to index {source!r}: {exc}"
        results = index.search(query, top_k=top_k, mode=mode,
                               filter_languages=filter_languages, filter_paths=filter_paths)
        if not results:
            return "No results found."
        contexts = index.get_context_for_results(results)
        syms = _collect_symbols(index, results)
        return _format_results_json(results, contexts, compact=compact, symbols=syms)

    @server.tool()
    async def find_related(
        file_path: Annotated[
            str,
            Field(description="Path to the file as stored in the index (use file_path from a search result)."),
        ],
        line: Annotated[int, Field(description="Line number (1-indexed).")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        top_k: Annotated[int, Field(description="Number of similar chunks to return.", ge=1)] = 5,
        compact: Annotated[
            bool,
            Field(description="If true, omit code content to save tokens."),
        ] = False,
    ) -> str:
        """Find code chunks semantically similar to a specific location in a file.

        Use after `search` to explore related implementations or callers.
        Pass file_path and line from a prior search result.
        """
        source = repo or default_source
        if not source:
            return (
                "No repo specified and no default index. "
                "Pass a git URL (https://github.com/...) or local path as `repo`."
            )
        try:
            index = await cache.get(source)
        except Exception as exc:
            return f"Failed to index {source!r}: {exc}"
        chunk = _resolve_chunk(index.chunks, file_path, line, file_mapping=index._file_mapping)
        if chunk is None:
            return (
                f"No chunk found at {file_path}:{line}. "
                "Make sure the file is indexed and the line number is within a known chunk."
            )
        results = index.find_related(chunk, top_k=top_k)
        if not results:
            return f"No related chunks found for {file_path}:{line}."
        contexts = index.get_context_for_results(results)
        syms = _collect_symbols(index, results)
        return _format_results_json(results, contexts, compact=compact, symbols=syms)

    @server.tool()
    async def get_impact_radius(
        symbol: Annotated[str, Field(description="Function or class name to analyze.")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
        depth: Annotated[int, Field(description="Recursion depth for caller/inheritance traversal (1-10).", ge=1, le=10)] = 3,
    ) -> str:
        """Recursive blast-radius analysis: find all callers and subclasses of a symbol.

        Traverses calls and inheritance edges up to `depth` levels. Returns
        the full impact tree and a flat list of impacted files. Use before
        modifying a function or class to understand the blast radius.
        """
        source = repo or default_source
        if not source:
            return json.dumps({"error": "No repo specified."})
        try:
            index = await cache.get(source)
        except Exception as exc:
            return json.dumps({"error": f"Failed to index {source!r}: {exc}"})
        result = index.get_impact_radius(symbol, depth=depth)
        return json.dumps(result, ensure_ascii=False)

    @server.tool()
    async def trace_symbol(
        symbol: Annotated[str, Field(description="Function or class name to trace (e.g. 'resolve_alpha').")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
    ) -> str:
        """Trace a symbol through the code graph: who calls it, what it calls, who imports it.

        Returns a compact subgraph with centrality scores. Use after search
        to understand how a function fits into the codebase without reading files.
        """
        source = repo or default_source
        if not source:
            return json.dumps({"error": "No repo specified and no default index."})
        try:
            index = await cache.get(source)
        except Exception as exc:
            return json.dumps({"error": f"Failed to index {source!r}: {exc}"})
        result = index.trace_symbol(symbol)
        if not result.get("found"):
            return json.dumps({"error": f"Symbol {symbol!r} not found in graph."})
        return json.dumps(result, ensure_ascii=False)

    @server.tool()
    async def explore_graph(
        file_path: Annotated[
            str,
            Field(description="File path as returned by a search result."),
        ],
        line: Annotated[int, Field(description="Line number within the file (1-indexed).")],
        repo: Annotated[str | None, Field(description=_REPO_DESCRIPTION)] = None,
    ) -> str:
        """Explore the code relationship graph for a specific location.

        Returns the call chain: what calls this code (called_by) and what it
        depends on (depends_on), plus the symbols defined at this location.
        Useful after `search` to understand how a chunk fits into the codebase.
        """
        source = repo or default_source
        if not source:
            return (
                "No repo specified and no default index. "
                "Pass a git URL (https://github.com/...) or local path as `repo`."
            )
        try:
            index = await cache.get(source)
        except Exception as exc:
            return f"Failed to index {source!r}: {exc}"

        chunk = _resolve_chunk(index.chunks, file_path, line, file_mapping=index._file_mapping)
        if chunk is None:
            return f"No chunk found at {file_path}:{line}."

        ctx = index.get_context_for_chunk(chunk)
        symbols = index.get_symbols_for_chunk(chunk)

        return json.dumps({
            "location": chunk.location,
            "symbols": symbols,
            "called_by": ctx.called_by,
            "depends_on": ctx.depends_on,
        }, ensure_ascii=False)

    return server


def _collect_symbols(index: SembleIndex, results: list) -> dict[str, list[dict]]:
    """Return ``{chunk_id: [symbol_dicts]}`` for results that have symbols."""
    symbols: dict[str, list[dict]] = {}
    for r in results:
        syms = index.get_symbols_for_chunk(r.chunk)
        if syms:
            symbols[r.chunk.location] = syms
    return symbols


async def serve(path: str | None = None, ref: str | None = None) -> None:
    """Start an MCP stdio server, optionally pre-indexing a default source."""
    model = await asyncio.to_thread(load_model)
    cache = _IndexCache(model=model)
    if path:
        await cache.get(path, ref=ref)

    server = create_server(cache, default_source=path)
    await server.run_stdio_async()


class _IndexCache:
    """Cache of indexed repos and local paths for the lifetime of the MCP server process."""

    def __init__(self, model: Encoder) -> None:
        """Initialise an empty cache with a shared embedding model."""
        self._model = model
        self._tasks: dict[str, asyncio.Task[SembleIndex]] = {}

    async def get(self, source: str, ref: str | None = None) -> SembleIndex:
        """Return an index for the requested source, building and caching it on first access."""
        is_git = _is_git_url(source)
        cache_key = (f"{source}@{ref}" if ref else source) if is_git else str(Path(source).resolve())

        if cache_key not in self._tasks:
            if is_git:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(SembleIndex.from_git, source, ref=ref, model=self._model)
                )
            else:
                self._tasks[cache_key] = asyncio.create_task(
                    asyncio.to_thread(SembleIndex.from_path, cache_key, model=self._model)
                )
        task = self._tasks[cache_key]
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:  # pragma: no cover
            if task.done():
                self._tasks.pop(cache_key, None)
            raise
        except Exception:
            # Build failed: evict so the next caller can retry.
            self._tasks.pop(cache_key, None)
            raise
