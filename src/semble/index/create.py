import contextlib
from pathlib import Path

import bm25s
from vicinity.backends.basic import BasicArgs

from semble.index.chunker import chunk_source
from semble.index.dense import SelectableBasicBackend, embed_chunks
from semble.index.file_walker import filter_extensions, language_for_path, walk_files
from semble.index.graph_extractor import extract_edges_from_file, extract_symbols_from_file
from semble.index.graph_store import GraphStore
from semble.index.sparse import enrich_for_bm25
from semble.tokens import tokenize
from semble.types import Chunk, Encoder


def create_index_from_path(
    path: Path,
    model: Encoder,
    extensions: frozenset[str] | None = None,
    ignore: frozenset[str] | None = None,
    include_text_files: bool = False,
    display_root: Path | None = None,
) -> tuple[bm25s.BM25, SelectableBasicBackend, list[Chunk], GraphStore]:
    """Create an index from a resolved directory, optionally storing chunk paths relative to display_root.

    :param path: Resolved absolute path to index.
    :param model: The model to use for indexing.
    :param extensions: File extensions to include.
    :param ignore: Directory names to skip.
    :param include_text_files: If True, also index non-code text files (.md, .yaml, .json, etc.).
    :param display_root: If set, chunk file paths are stored relative to this root.
    :raises ValueError: if no items were found, no index can be created.
    :return: A bm25 index, vicinity index, list of chunks, and the graph store.
    """
    extensions = filter_extensions(extensions, include_text_files=include_text_files)
    graph_store = GraphStore()
    chunks: list[Chunk] = []
    edge_data: list = []  # RawFileEdges from pass 1, resolved in pass 2

    # Walk and chunk (same pass as before, source freed after loop iteration).
    for file_path in walk_files(path, extensions, ignore):
        language = language_for_path(file_path)
        with contextlib.suppress(OSError):
            source = file_path.read_text(encoding="utf-8", errors="replace")
            chunk_path = file_path.relative_to(display_root) if display_root else file_path
            file_chunks = chunk_source(source, str(chunk_path), language)
            chunks.extend(file_chunks)
            if language and file_chunks:
                # Pass 1: parse once, store symbols, collect raw edges.
                raw = extract_symbols_from_file(source, str(chunk_path), language, file_chunks, graph_store)
                if raw is not None:
                    edge_data.append(raw)
            # source is freed here — no full-copy retained.

    # Pass 2: resolve edges using the complete symbol table (no re-parse).
    for raw in edge_data:
        extract_edges_from_file(raw, graph_store)

    if chunks:
        embeddings = embed_chunks(model, chunks)
        bm25_index = bm25s.BM25()
        bm25_index.index(
            [tokenize(enrich_for_bm25(chunk)) for chunk in chunks],
            show_progress=False,
        )
        args = BasicArgs()
        semantic_index = SelectableBasicBackend(embeddings, args)
    else:
        raise ValueError(f"No supported files found under {path}.")

    return bm25_index, semantic_index, chunks, graph_store
