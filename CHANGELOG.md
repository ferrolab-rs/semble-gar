# Changelog

## 1.0.0 — 2026-05-04

Initial release of Semble-GAR, a fork of [MinishLab/semble](https://github.com/MinishLab/semble)
adding Graph-Augmented Retrieval (SQLite + tree-sitter).

### Added

- **Graph-Augmented Retrieval** — SQLite-backed code relationship graph with zero new dependencies
- **`GraphStore`** — symbols + edges with cross-file resolution, import scoping, wildcard expansion
- **`GraphExtractor`** — tree-sitter AST extraction (functions, classes, calls, imports) with silent vector-only fallback
- **Graph centrality boost** — RRF ranking boosted by code graph degree centrality (`_GRAPH_BOOST=0.4`)
- **`trace_symbol` MCP tool** — traverse call graph by function/class name with centrality scores
- **`explore_graph` MCP tool** — relational context (called_by/depends_on) for any chunk
- **`search` enhancements** — `filter_languages`, `filter_paths`, `compact` mode
- **`file_total_lines`** — each result shows total file size so agents detect partial chunks
- **`symbols` in search results** — functions/classes defined in each chunk, chainable to `trace_symbol`
- **Ablation framework** — `benchmarks/ablation.py` compares GAR vs baseline on upstream benchmarks
- **Comprehensive tests** — 129 tests (116 upstream + 13 GAR integration), 0 regressions

### Changed

- MCP output format: markdown → JSON with `called_by`/`depends_on` context
- `create_index_from_path` returns 4-tuple (bm25, vicinity, chunks, graph_store)
- Two-pass indexing: symbols first, then cross-file edge resolution

### Fixed

- Aliased imports (`from X import Y as Z`) correctly resolve through alias chain
- Wildcard imports (`from X import *`) expand to all module symbols
- Module-level calls captured via `*module*` pseudo-symbol
- `_build_import_map` uses single JOIN instead of N sub-queries
- `get_graph_centrality` uses direct COUNT queries (no longer calls heavy `get_relational_context`)
- Thread-safe SQLite (`check_same_thread=False`) + context manager support
- `file://` removed from git URL schemes (security)
- `git clone` has 120s timeout

### Known limitations

- No persistent index cache (index lost on MCP restart)
- Centrality boost calibrated empirically; full NDCG@10 ablation pending upstream run
- `_scan_non_candidates` scans all chunks (upstream, not GAR-specific)
