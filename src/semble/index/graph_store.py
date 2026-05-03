from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from semble.types import GraphContext

logger = logging.getLogger(__name__)


class GraphStore:
    """SQLite-backed graph of code symbols and their relations.

    Thread-safe for MCP usage: uses ``check_same_thread=False`` so the
    connection can be shared across the indexing thread and the async
    event-loop thread.  Call :meth:`close` (or use as a context manager)
    to release the connection.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()
        self._import_map_cache: dict[str, list[str]] | None = None

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                file TEXT NOT NULL,
                chunk_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                source_id INTEGER,
                target_id INTEGER,
                type TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_chunk ON symbols(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        """)

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def insert_symbol(self, name: str, type_: str, file: str, chunk_id: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO symbols (name, type, file, chunk_id) VALUES (?, ?, ?, ?)",
            (name, type_, file, chunk_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def insert_edge(self, source_id: int, target_id: int, type_: str) -> None:
        self.conn.execute(
            "INSERT INTO edges (source_id, target_id, type) VALUES (?, ?, ?)",
            (source_id, target_id, type_),
        )
        self.conn.commit()

    def ensure_import_symbol(self, file_path: str) -> int:
        """Return the symbol id for the ``*import*`` pseudo-symbol of *file_path*, creating it if needed."""
        rows = self.conn.execute(
            "SELECT id FROM symbols WHERE name = '*import*' AND file = ?",
            (file_path,),
        ).fetchall()
        if rows:
            return rows[0][0]
        cur = self.conn.execute(
            "INSERT INTO symbols (name, type, file, chunk_id) VALUES ('*import*', 'import', ?, ?)",
            (file_path, f"{file_path}:0-0"),
        )
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_chunk_ids_for_symbols(self, symbol_names: list[str]) -> set[str]:
        if not symbol_names:
            return set()
        placeholders = ",".join("?" * len(symbol_names))
        rows = self.conn.execute(
            f"SELECT DISTINCT chunk_id FROM symbols WHERE name IN ({placeholders})",
            symbol_names,
        ).fetchall()
        return {row[0] for row in rows}

    def get_symbols_by_chunk(self, chunk_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, type FROM symbols WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchall()
        return [{"id": row[0], "name": row[1], "type": row[2]} for row in rows]

    def lookup_symbol_ids(self, name: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM symbols WHERE name = ?",
            (name,),
        ).fetchall()
        return [row[0] for row in rows]

    def lookup_symbol_ids_scoped(self, name: str, module_source: str) -> list[int]:
        """Like lookup_symbol_ids but only matches symbols whose file belongs to *module_source*."""
        module_end = module_source.split(".")[-1]
        patterns = _module_file_patterns(module_end)
        clauses = " OR ".join("file LIKE ?" for _ in patterns)
        rows = self.conn.execute(
            f"SELECT id FROM symbols WHERE name = ? AND ({clauses})",
            (name, *patterns),
        ).fetchall()
        return [row[0] for row in rows]

    def get_module_symbols(self, module_source: str) -> list[str]:
        """Return all symbol names defined in files belonging to *module_source*."""
        module_end = module_source.split(".")[-1]
        patterns = _module_file_patterns(module_end)
        clauses = " OR ".join("file LIKE ?" for _ in patterns)
        rows = self.conn.execute(
            f"SELECT DISTINCT name FROM symbols WHERE name NOT IN ('*import*', '*module*') AND ({clauses})",
            patterns,
        ).fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    # Relational context & centrality
    # ------------------------------------------------------------------

    def get_relational_context(self, chunk_ids: list[str]) -> dict[str, GraphContext]:
        """For each chunk_id, return what OTHER chunks call it and what it depends on."""
        if not chunk_ids:
            return {}

        placeholders = ",".join("?" * len(chunk_ids))
        symbol_rows = self.conn.execute(
            f"SELECT id, chunk_id FROM symbols WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()

        symbol_to_chunk: dict[int, str] = {row[0]: row[1] for row in symbol_rows}
        contexts: dict[str, GraphContext] = {cid: GraphContext() for cid in chunk_ids}
        all_symbol_ids = list(symbol_to_chunk.keys())
        if not all_symbol_ids:
            return contexts

        sid_placeholders = ",".join("?" * len(all_symbol_ids))

        # Lazy cache: chunk_id lookup for any symbol.
        _chunk_cache: dict[int, str] = {}

        def _cid(sid: int) -> str | None:
            if sid in symbol_to_chunk:
                return symbol_to_chunk[sid]
            if sid in _chunk_cache:
                return _chunk_cache[sid]
            row = self.conn.execute("SELECT chunk_id FROM symbols WHERE id = ?", (sid,)).fetchone()
            if row:
                _chunk_cache[sid] = row[0]
                return row[0]
            return None

        # Expand *import* pseudo-symbols: map them → real chunks of the same file.
        # Cached lazily: only built once, and only for files that have import edges.
        import_map = self._build_import_map()

        # --- called_by ---
        incoming = self.conn.execute(
            f"SELECT e.source_id, e.target_id FROM edges e "
            f"WHERE e.target_id IN ({sid_placeholders})",
            all_symbol_ids,
        ).fetchall()
        for source_sid, target_sid in incoming:
            target_chunk = _cid(target_sid)
            source_chunk = _cid(source_sid)
            if not target_chunk or not source_chunk or source_chunk == target_chunk:
                continue
            if source_chunk in import_map:
                # Exclude self-references when expanding *import* pseudo-symbols.
                contexts[target_chunk].called_by.extend(
                    c for c in import_map[source_chunk] if c != target_chunk
                )
            else:
                contexts[target_chunk].called_by.append(source_chunk)

        # --- depends_on ---
        outgoing = self.conn.execute(
            f"SELECT e.source_id, e.target_id FROM edges e "
            f"WHERE e.source_id IN ({sid_placeholders})",
            all_symbol_ids,
        ).fetchall()
        for source_sid, target_sid in outgoing:
            source_chunk = _cid(source_sid)
            target_chunk = _cid(target_sid)
            if not source_chunk or not target_chunk or source_chunk == target_chunk:
                continue
            if target_chunk in import_map:
                contexts[source_chunk].depends_on.extend(
                    c for c in import_map[target_chunk] if c != source_chunk
                )
            else:
                contexts[source_chunk].depends_on.append(target_chunk)

        # Deduplicate + filter pseudo-chunks.
        for ctx in contexts.values():
            ctx.called_by[:] = sorted(c for c in set(ctx.called_by) if not c.endswith(":0-0"))
            ctx.depends_on[:] = sorted(c for c in set(ctx.depends_on) if not c.endswith(":0-0"))

        return contexts

    def get_graph_centrality(self, chunk_ids: list[str]) -> dict[str, float]:
        """Return normalised degree centrality for each chunk_id."""
        if not chunk_ids:
            return {}

        contexts = self.get_relational_context(chunk_ids)
        raw: dict[str, float] = {}
        degrees: list[int] = []
        for cid, ctx in contexts.items():
            d = len(ctx.called_by) + len(ctx.depends_on)
            raw[cid] = float(d)
            degrees.append(d)

        max_degree = max(degrees) if degrees else 0
        if max_degree == 0:
            return {cid: 0.0 for cid in chunk_ids}
        return {cid: raw[cid] / float(max_degree) for cid in chunk_ids}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_import_map(self) -> dict[str, list[str]]:
        """Return ``{*import*_chunk_id: [real_chunk_ids_for_that_file]}``.

        Built lazily on first call and cached (the import symbols don't change
        after indexing).
        """
        if self._import_map_cache is None:
            mapping: dict[str, list[str]] = {}
            import_rows = self.conn.execute(
                "SELECT file, chunk_id FROM symbols WHERE name = '*import*'",
            ).fetchall()
            for file, cid in import_rows:
                real = self.conn.execute(
                    "SELECT DISTINCT chunk_id FROM symbols WHERE file = ? AND name NOT IN ('*import*', '*module*')",
                    (file,),
                ).fetchall()
                mapping[cid] = [r[0] for r in real]
            self._import_map_cache = mapping
        return self._import_map_cache

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def _module_file_patterns(module_end: str) -> list[str]:
    """Return LIKE patterns that match files belonging to *module_end*.

    Uses ``_`` (SQL single-char wildcard) for the path separator so both
    ``/`` and ``\\`` are matched regardless of platform.
    """
    return [
        f"{module_end}_%",       # ranking_% matches ranking/foo or ranking\foo
        f"%_{module_end}.py",    # %_ranking.py matches .../ranking.py or ...\ranking.py
        f"%_{module_end}_%",     # %_ranking_% matches .../ranking/... or ...\ranking\...
        f"{module_end}.py",      # ranking.py (no dir prefix)
    ]
