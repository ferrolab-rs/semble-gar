from __future__ import annotations

import logging
from dataclasses import dataclass, field

from semble.index.graph_store import GraphStore
from semble.types import Chunk

logger = logging.getLogger(__name__)

# Language names as expected by tree-sitter-language-pack.
_TREE_SITTER_LANG: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "kotlin": "kotlin",
    "ruby": "ruby",
    "c": "c",
    "cpp": "cpp",
    "csharp": "c_sharp",
    "swift": "swift",
    "scala": "scala",
    "elixir": "elixir",
    "dart": "dart",
    "lua": "lua",
    "bash": "bash",
    "zig": "zig",
    "haskell": "haskell",
    "php": "php",
    "sql": "sql",
}


@dataclass(slots=True)
class RawFileEdges:
    """Pre-extracted edges for a single file, held in memory between pass 1 and pass 2."""

    file_path: str
    raw_edges: list[dict] = field(default_factory=list)
    import_map: dict[str, str] = field(default_factory=dict)
    wildcard_modules: list[str] = field(default_factory=list)
    alias_to_original: dict[str, str] = field(default_factory=dict)


def extract_symbols_from_file(
    source: str, file_path: str, language: str, chunks: list[Chunk], store: GraphStore
) -> RawFileEdges | None:
    """Extract and store symbols from a file AST (pass 1).

    Parses the source once and returns *RawFileEdges* for use in pass 2,
    or *None* if extraction failed (parse error, unsupported language, etc.).
    """
    symbols, raw_edges, import_map, wildcard_modules, alias_map = _extract_graph_data(source, language)
    if symbols is None:
        return None

    chunk_id_by_line = _build_line_to_chunk_map(chunks)
    for sym in symbols:
        cid = _chunk_for_line(sym["line"], chunk_id_by_line)
        if cid is not None:
            store.insert_symbol(sym["name"], sym["type"], file_path, cid)

    return RawFileEdges(
        file_path=file_path,
        raw_edges=raw_edges,
        import_map=import_map,
        wildcard_modules=wildcard_modules,
        alias_to_original=alias_map,
    )


def extract_edges_from_file(data: RawFileEdges, store: GraphStore) -> None:
    """Resolve pre-extracted edges against the complete symbol table (pass 2).

    Does not re-parse — the raw edges and import map from pass 1 are reused.
    """
    local_symbols: dict[str, list[int]] = {}
    for sym in _get_stored_symbols(store, data.file_path):
        local_symbols.setdefault(sym["name"], []).append(sym["id"])

    # Expand wildcard imports and alias→original chain.
    import_map = dict(data.import_map)
    alias_to_original = dict(data.alias_to_original)
    for module_source in data.wildcard_modules:
        for name in store.get_module_symbols(module_source):
            if name not in import_map:
                import_map[name] = module_source

    if not data.raw_edges:
        return

    # Module-level calls use a per-file pseudo-symbol as caller.
    module_sym_id: int | None = None
    import_sym_id: int | None = None

    for edge in data.raw_edges:
        src_ids = local_symbols.get(edge["source"], [])
        tgt_ids = local_symbols.get(edge["target"], [])
        target_name: str = edge["target"]

        # Module-level callers: use *module* pseudo-symbol for the file.
        if edge["type"] == "calls" and edge["source"] == "*module*" and not src_ids:
            if module_sym_id is None:
                module_sym_id = _ensure_module_symbol(store, data.file_path)
            src_ids = [module_sym_id]

        # Resolve aliases: ``from X import Y as Z`` → call to Z should find Y.
        resolve_name = alias_to_original.get(target_name, target_name)

        # Cross-file fallback: scoped lookup (import-aware) first, then global.
        if not tgt_ids:
            module_source = import_map.get(target_name) or import_map.get(resolve_name)
            if module_source:
                tgt_ids = store.lookup_symbol_ids_scoped(resolve_name, module_source)
            if not tgt_ids:
                tgt_ids = store.lookup_symbol_ids(resolve_name)
        if not src_ids:
            src_ids = store.lookup_symbol_ids(edge["source"])

        # Import edges: wire from the file's *import* pseudo-symbol.
        if edge["type"] == "imports" and not src_ids:
            if import_sym_id is None:
                import_sym_id = store.ensure_import_symbol(data.file_path)
            src_ids = [import_sym_id]

        for src_id in src_ids:
            for tgt_id in tgt_ids:
                if src_id != tgt_id:
                    store.insert_edge(src_id, tgt_id, edge["type"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_graph_data(source: str, language: str):
    """Single tree-sitter parse returning ``(symbols, raw_edges, import_map, wildcard_modules, alias_map)``.

    Returns ``(None, ...)`` on any failure — the graph stays empty for this file
    and search falls back to vector-only.
    """
    ts_lang = _TREE_SITTER_LANG.get(language)
    if not ts_lang:
        return None, None, None, None, None

    try:
        import tree_sitter_language_pack as tsl  # noqa: PLC0415
    except ImportError:
        logger.debug("tree-sitter-language-pack not available, graph extraction disabled")
        return None, None, None, None, None

    try:
        parser = tsl.get_parser(ts_lang)
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        logger.debug("Tree-sitter parse failed for language %r", ts_lang, exc_info=True)
        return None, None, None, None, None

    try:
        symbols = _extract_symbols(tree, source, ts_lang)
    except Exception:
        logger.debug("Symbol extraction failed", exc_info=True)
        symbols = []

    try:
        raw_edges, import_map, wildcard_modules, alias_map = _extract_edges_and_imports(tree, source, ts_lang)
    except Exception:
        logger.debug("Edge extraction failed", exc_info=True)
        raw_edges, import_map, wildcard_modules, alias_map = [], {}, [], {}

    return symbols, raw_edges, import_map, wildcard_modules, alias_map


def _get_stored_symbols(store: GraphStore, file_path: str) -> list[dict]:
    rows = store.conn.execute(
        "SELECT id, name FROM symbols WHERE file = ? AND name NOT IN ('*import*', '*module*')",
        (file_path,),
    ).fetchall()
    return [{"id": row[0], "name": row[1]} for row in rows]


def _ensure_module_symbol(store: GraphStore, file_path: str) -> int:
    """Return the symbol id for the ``*module*`` pseudo-symbol of *file_path*."""
    rows = store.conn.execute(
        "SELECT id FROM symbols WHERE name = '*module*' AND file = ?",
        (file_path,),
    ).fetchall()
    if rows:
        return rows[0][0]
    cur = store.conn.execute(
        "INSERT INTO symbols (name, type, file, chunk_id) VALUES ('*module*', 'module', ?, ?)",
        (file_path, f"{file_path}:0-0"),
    )
    store.conn.commit()
    return cur.lastrowid


def _build_line_to_chunk_map(chunks: list[Chunk]) -> list[tuple[int, int, str]]:
    return sorted((c.start_line, c.end_line, c.location) for c in chunks)


def _chunk_for_line(line: int, line_map: list[tuple[int, int, str]]) -> str | None:
    lo, hi = 0, len(line_map) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, cid = line_map[mid]
        if start <= line <= end:
            return cid
        if line < start:
            hi = mid - 1
        else:
            lo = mid + 1
    return None


def _extract_symbols(tree, source: str, lang: str) -> list[dict]:
    symbols: list[dict] = []
    _walk_symbols(tree.root_node, source, lang, symbols)
    return symbols


def _walk_symbols(node, source: str, lang: str, symbols: list[dict]) -> None:
    kind = node.type

    if kind in _FUNC_NODE_TYPES:
        name = _child_text(node, _FUNC_NAME_FIELDS)
        if name:
            symbols.append({"name": name, "type": "function", "line": node.start_point[0] + 1})
    elif kind in _CLASS_NODE_TYPES:
        name = _child_text(node, _CLASS_NAME_FIELDS)
        if name:
            symbols.append({"name": name, "type": "class", "line": node.start_point[0] + 1})

    for child in node.children:
        _walk_symbols(child, source, lang, symbols)


# --- Node type tables ---

_FUNC_NODE_TYPES: frozenset[str] = frozenset({
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
    "function_item",
    "arrow_function",
    "function_expression",
    "constructor_declaration",
    "destructor_declaration",
})

_CLASS_NODE_TYPES: frozenset[str] = frozenset({
    "class_definition",
    "class_declaration",
    "struct_item",
    "interface_declaration",
    "trait_item",
    "enum_declaration",
    "impl_item",
    "type_alias_declaration",
})

_CALL_NODE_TYPES: frozenset[str] = frozenset({
    "call",
    "call_expression",
    "method_invocation",
    "function_call",
})

_IMPORT_NODE_TYPES: frozenset[str] = frozenset({
    "import_statement",
    "import_declaration",
    "import_from_statement",
    "use_declaration",
    "require_call",
    "include_statement",
    "using_directive",
})

_FUNC_NAME_FIELDS: tuple[str, ...] = ("name", "declarator")
_CLASS_NAME_FIELDS: tuple[str, ...] = ("name",)
_CALL_NAME_FIELDS: tuple[str, ...] = ("function", "method", "name")
_IMPORT_NAME_FIELDS: tuple[str, ...] = ("source", "path", "module", "name")

_DEF_NODE_TYPES: frozenset[str] = _FUNC_NODE_TYPES | _CLASS_NODE_TYPES
_DEF_NAME_FIELDS: tuple[str, ...] = _FUNC_NAME_FIELDS + _CLASS_NAME_FIELDS


def _extract_edges_and_imports(tree, source: str, lang: str) -> tuple[list[dict], dict[str, str], list[str], dict[str, str]]:
    edges: list[dict] = []
    import_map: dict[str, str] = {}
    wildcard_modules: list[str] = []
    alias_map: dict[str, str] = {}
    _walk_edges(tree.root_node, source, edges, import_map, wildcard_modules, alias_map)
    return edges, import_map, wildcard_modules, alias_map


def _walk_edges(node, source: str, edges: list[dict], import_map: dict[str, str], wildcard_modules: list[str], alias_map: dict[str, str]) -> None:
    kind = node.type

    if kind in _CALL_NODE_TYPES:
        callee = _child_text(node, _CALL_NAME_FIELDS)
        if callee and _is_meaningful_symbol(callee):
            caller = _enclosing_function(node)
            if callee != caller:
                edges.append({"source": caller, "target": callee, "type": "calls"})

    elif kind in _IMPORT_NODE_TYPES:
        module_path, names, is_wildcard, aliases = _extract_import_info(node)
        if is_wildcard and module_path:
            wildcard_modules.append(module_path)
        for n in names:
            if n and _is_meaningful_symbol(n):
                edges.append({"source": "*import*", "target": n, "type": "imports"})
                if module_path:
                    import_map[n] = module_path
        # Store alias→original mappings for call-edge resolution.
        alias_map.update(aliases)

    for child in node.children:
        _walk_edges(child, source, edges, import_map, wildcard_modules, alias_map)


def _extract_import_info(node) -> tuple[str | None, list[str], bool, dict[str, str]]:
    """Extract module source and imported names from an import statement.

    Returns ``(module_path, [names], is_wildcard, aliases)`` where *aliases*
    maps local alias names to original names (e.g. ``{"lfn": "long_function_name"}``).
    """
    module_path: str | None = None
    names: list[str] = []
    dotted_names: list[str] = []
    is_wildcard = False
    aliases: dict[str, str] = {}

    for child in node.children:
        if child.type == "dotted_name":
            text = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
            if text:
                dotted_names.append(text)
        elif child.type == "aliased_import":
            alias = child.child_by_field_name("alias")
            original = child.child_by_field_name("name")
            alias_text: str | None = None
            original_text: str | None = None
            if alias:
                alias_text = alias.text.decode("utf-8") if isinstance(alias.text, bytes) else alias.text
                if alias_text:
                    names.append(alias_text)
            if original:
                original_text = original.text.decode("utf-8") if isinstance(original.text, bytes) else original.text
                if original_text:
                    names.append(original_text)
            if alias_text and original_text:
                aliases[alias_text] = original_text
            elif not alias and not original:
                first = next((c for c in child.children if c.type == "dotted_name"), None)
                if first:
                    text = first.text.decode("utf-8") if isinstance(first.text, bytes) else first.text
                    if text:
                        names.append(text)
        elif child.type == "wildcard_import":
            is_wildcard = True

    if node.type == "import_from_statement":
        if dotted_names:
            module_path = dotted_names[0]
            names.extend(dotted_names[1:] if len(dotted_names) > 1 else [])
    else:
        names.extend(dotted_names)

    return module_path, names, is_wildcard, aliases


_BUILTINS: frozenset[str] = frozenset({
    "len", "range", "print", "int", "str", "float", "bool", "list", "dict",
    "set", "tuple", "type", "isinstance", "issubclass", "hasattr", "getattr",
    "setattr", "delattr", "min", "max", "sum", "sorted", "enumerate", "zip",
    "map", "filter", "any", "all", "abs", "round", "pow", "divmod", "chr",
    "ord", "hex", "oct", "bin", "repr", "id", "object", "super", "open",
    "iter", "next", "input", "format", "bytes", "bytearray", "memoryview",
    "slice", "reversed", "complex", "staticmethod", "classmethod", "property",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
    "RuntimeError", "OSError", "FileNotFoundError", "NotADirectoryError",
    "True", "False", "None",
})


def _is_meaningful_symbol(name: str) -> bool:
    if name in _BUILTINS:
        return False
    base = name.split(".")[-1] if "." in name else name
    if base in _BUILTINS:
        return False
    return len(base) >= 2


def _child_text(node, field_names: tuple[str, ...]) -> str | None:
    for name in field_names:
        child = node.child_by_field_name(name)
        if child is not None:
            text = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
            if text:
                return text
    for child in node.children:
        if child.type in ("identifier", "attribute") or child.type.endswith("_identifier"):
            text = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
            if text:
                return text
    return None


def _enclosing_function(node) -> str:
    """Return the enclosing function/class name, or ``*module*`` for top-level calls."""
    current = node.parent
    while current is not None:
        if current.type in _DEF_NODE_TYPES:
            name = _child_text(current, _DEF_NAME_FIELDS)
            if name:
                return name
        current = current.parent
    return "*module*"
